// Copyright (c) 2026 Daniel Harding - RomanAILabs
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// nrl/llama_bridge.h - Phase 7-EG native libllama bridge ABI.
//
// The bridge is the single point in the native ladder where actual model
// inference is performed. It exposes a stable C ABI so that the native
// resolver (engine/src/ladder_native.c) does not need to know whether it is
// driving a stub backend, a Python-registered callback, or a Phase-8-EG
// direct ``libllama.dll`` link.
//
// Backends:
//   * NRL_LLAMA_BACKEND_STUB     - deterministic in-process stub mirroring
//                                  the Python NRL_INFERENCE=stub backend.
//                                  Used by tests and by hosts without a
//                                  registered libllama linkage.
//   * NRL_LLAMA_BACKEND_CALLBACK - calls the user-registered callback for
//                                  every turn. Today the Python runner sets
//                                  this callback to its existing libllama
//                                  driver (llama-cpp-python). In Phase 8-EG
//                                  the same surface is rebound to direct
//                                  libllama linkage with no API change.
//
// All strings passed across this ABI are UTF-8 byte buffers; ownership rules
// are documented per call.

#ifndef NRL_LLAMA_BRIDGE_H_
#define NRL_LLAMA_BRIDGE_H_

#include <stddef.h>
#include <stdint.h>

#include "nrl/nrl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nrl_llama_backend {
    NRL_LLAMA_BACKEND_STUB = 0,
    NRL_LLAMA_BACKEND_CALLBACK = 1
} nrl_llama_backend;

/* Inputs the bridge needs for a single turn. ``model_utf8`` and
 * ``prompt_utf8`` MUST be NUL-terminated UTF-8 buffers owned by the caller. */
typedef struct nrl_llama_request {
    const char *model_utf8;
    const char *prompt_utf8;
    int32_t max_tokens;
    int32_t seed;
    int32_t n_ctx;
    int32_t n_threads;
    int32_t n_batch;
    float temperature;
    float top_p;
    int32_t top_k;
    float repeat_penalty;
} nrl_llama_request;

/* Per-turn output. ``text_buf`` is caller-owned and ``text_buf_capacity``
 * bytes long. The bridge writes at most ``capacity - 1`` bytes plus a NUL.
 * On overflow ``text_truncated`` is set to 1 and ``text_byte_len`` reports
 * the bytes the bridge would have written if the buffer were unbounded. */
typedef struct nrl_llama_response {
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
    int32_t tokens_emitted;
    double wall_seconds;
} nrl_llama_response;

/* User-supplied callback for the CALLBACK backend. ``user_data`` is the
 * cookie passed to nrl_v1_llama_set_callback. The callback MUST fill
 * ``response->text_byte_len`` and ``response->tokens_emitted`` and return
 * NRL_OK on success. Any non-NRL_OK return is propagated. */
typedef nrl_v1_status (*nrl_llama_callback_fn)(
    void *user_data,
    const nrl_llama_request *request,
    nrl_llama_response *response);

/* Process-wide bridge state. Backend selection is global by design: a single
 * GGUF runner owns one backend at a time; tests reset between runs. */
NRL_API nrl_v1_status nrl_v1_llama_set_backend(nrl_llama_backend backend);
NRL_API nrl_llama_backend nrl_v1_llama_get_backend(void);

/* Register the CALLBACK-backend callback. ``cb`` may be NULL to clear it.
 * Setting a callback does not switch the active backend; call
 * nrl_v1_llama_set_backend(NRL_LLAMA_BACKEND_CALLBACK) to engage it. */
NRL_API nrl_v1_status nrl_v1_llama_set_callback(
    nrl_llama_callback_fn cb,
    void *user_data);

/* Single-turn inference. Returns NRL_ERR_ARGS if the bridge is set to
 * CALLBACK but no callback is registered. Always populates wall_seconds. */
NRL_API nrl_v1_status nrl_v1_llama_run(
    const nrl_llama_request *request,
    nrl_llama_response *response);

/* Identifier string for the active backend. Returned pointer is static. */
NRL_API const char *nrl_v1_llama_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif /* NRL_LLAMA_BRIDGE_H_ */
