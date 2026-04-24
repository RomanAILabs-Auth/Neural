// Copyright (c) 2026 Daniel Harding - RomanAILabs
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// engine/src/llama_bridge.c - Phase 7-EG native libllama bridge.
//
// The bridge is intentionally tiny: it owns backend selection, timing, and
// the dispatch into either the deterministic stub or a registered callback.
// All actual model inference happens in the backend; the bridge merely
// shapes the call so the native ladder (engine/src/ladder_native.c) does
// not care which backend is engaged.
//
// Threading: the registered callback and backend selection are global state
// guarded by a single critical section. Callers that want concurrent runs
// must serialize at the application layer (the GGUF runner already does
// today). This matches the existing nrlpy._core threading model.

#include "nrl/llama_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static CRITICAL_SECTION g_bridge_lock;
static int32_t g_bridge_lock_initialized = 0;
static void bridge_lock_init(void) {
    if (!g_bridge_lock_initialized) {
        InitializeCriticalSection(&g_bridge_lock);
        g_bridge_lock_initialized = 1;
    }
}
static void bridge_lock(void) { bridge_lock_init(); EnterCriticalSection(&g_bridge_lock); }
static void bridge_unlock(void) { LeaveCriticalSection(&g_bridge_lock); }
static double bridge_now_seconds(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <pthread.h>
#include <time.h>
static pthread_mutex_t g_bridge_lock = PTHREAD_MUTEX_INITIALIZER;
static void bridge_lock(void) { pthread_mutex_lock(&g_bridge_lock); }
static void bridge_unlock(void) { pthread_mutex_unlock(&g_bridge_lock); }
static double bridge_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static nrl_llama_backend g_backend = NRL_LLAMA_BACKEND_STUB;
static nrl_llama_callback_fn g_callback = NULL;
static void *g_callback_user_data = NULL;

NRL_API nrl_v1_status nrl_v1_llama_set_backend(nrl_llama_backend backend) {
    if (backend != NRL_LLAMA_BACKEND_STUB
        && backend != NRL_LLAMA_BACKEND_CALLBACK) {
        return NRL_ERR_ARGS;
    }
    bridge_lock();
    g_backend = backend;
    bridge_unlock();
    return NRL_OK;
}

NRL_API nrl_llama_backend nrl_v1_llama_get_backend(void) {
    bridge_lock();
    nrl_llama_backend b = g_backend;
    bridge_unlock();
    return b;
}

NRL_API nrl_v1_status nrl_v1_llama_set_callback(
    nrl_llama_callback_fn cb,
    void *user_data) {
    bridge_lock();
    g_callback = cb;
    g_callback_user_data = user_data;
    bridge_unlock();
    return NRL_OK;
}

NRL_API const char *nrl_v1_llama_backend_name(void) {
    nrl_llama_backend b = nrl_v1_llama_get_backend();
    switch (b) {
        case NRL_LLAMA_BACKEND_STUB:     return "stub";
        case NRL_LLAMA_BACKEND_CALLBACK: return "callback";
        default:                         return "unknown";
    }
}

/* Deterministic stub backend.
 *
 * Mirrors the Python NRL_INFERENCE=stub backend: emits a fixed reply
 * derived from the prompt so unit tests can exercise the ladder without
 * loading a real model. The reply is chosen to be byte-identical to what
 * the Python stub produces for the same prompt: "<prompt> [stub]".
 *
 * Token count is the number of whitespace-separated runs in the reply
 * text, which matches the Python stub's accounting. */
static nrl_v1_status bridge_stub_run(
    const nrl_llama_request *request,
    nrl_llama_response *response) {
    if (request == NULL || response == NULL || response->text_buf == NULL) {
        return NRL_ERR_ARGS;
    }
    const char *prompt = request->prompt_utf8 ? request->prompt_utf8 : "";
    const size_t prompt_len = strlen(prompt);
    const char suffix[] = " [stub]";
    const size_t suffix_len = sizeof(suffix) - 1;
    const size_t total = prompt_len + suffix_len;

    if (response->text_buf_capacity == 0) {
        response->text_byte_len = total;
        response->text_truncated = 1;
        response->tokens_emitted = 0;
        return NRL_OK;
    }
    const size_t writable = response->text_buf_capacity - 1;
    const size_t to_copy = (total <= writable) ? total : writable;
    if (prompt_len <= writable) {
        memcpy(response->text_buf, prompt, prompt_len);
        const size_t left = writable - prompt_len;
        const size_t suff = (suffix_len <= left) ? suffix_len : left;
        memcpy(response->text_buf + prompt_len, suffix, suff);
    } else {
        memcpy(response->text_buf, prompt, to_copy);
    }
    response->text_buf[to_copy] = '\0';
    response->text_byte_len = total;
    response->text_truncated = (total > writable) ? 1 : 0;

    /* Whitespace-separated token count over the *full* would-be reply. */
    int32_t tokens = 0;
    int32_t in_word = 0;
    for (size_t i = 0; i < prompt_len; ++i) {
        char c = prompt[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            tokens += 1;
        }
    }
    /* The "[stub]" suffix is always one extra token. */
    tokens += 1;
    if (tokens < 1) tokens = 1;
    response->tokens_emitted = tokens;
    return NRL_OK;
}

NRL_API nrl_v1_status nrl_v1_llama_run(
    const nrl_llama_request *request,
    nrl_llama_response *response) {
    if (request == NULL || response == NULL) {
        return NRL_ERR_ARGS;
    }
    response->text_byte_len = 0;
    response->text_truncated = 0;
    response->tokens_emitted = 0;
    response->wall_seconds = 0.0;

    bridge_lock();
    nrl_llama_backend backend = g_backend;
    nrl_llama_callback_fn cb = g_callback;
    void *user_data = g_callback_user_data;
    bridge_unlock();

    const double t0 = bridge_now_seconds();
    nrl_v1_status rc = NRL_ERR_INTERNAL;
    if (backend == NRL_LLAMA_BACKEND_STUB) {
        rc = bridge_stub_run(request, response);
    } else if (backend == NRL_LLAMA_BACKEND_CALLBACK) {
        if (cb == NULL) {
            rc = NRL_ERR_ARGS;
        } else {
            rc = cb(user_data, request, response);
        }
    } else {
        rc = NRL_ERR_INTERNAL;
    }
    const double t1 = bridge_now_seconds();
    response->wall_seconds = (t1 > t0) ? (t1 - t0) : 0.0;
    return rc;
}
