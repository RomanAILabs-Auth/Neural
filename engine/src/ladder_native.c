// Copyright (c) 2026 Daniel Harding - RomanAILabs
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// engine/src/ladder_native.c - Phase 7-EG native Resolution Ladder.
//
// Implements the §4.2 ladder dispatch in C. The selection order MUST stay
// byte-identical to nrlpy.gguf.run_gguf:
//
//   R0  muscle memory hit       (cache_hit==True path in Python)
//   R1  ZPM nullspace hit       (zpm_hit_meta path in Python)
//   R2  omega native (active)   (lane-gated, requires r2_shadow_enabled,
//                                Stage-VI must already have passed when
//                                the caller built the candidate)
//   R5  libllama bridge call    (the §4.2 fallback)
//
// R3/R4 slots exist on the input struct so future phases (and the §4.2
// release-gate auditor) can attach pre-computed candidates without ABI
// churn. They are intentionally never selected here.
//
// Honest accounting: ``wall_seconds`` on the output is the rung's own
// wall-clock contribution. For R0/R1/R2 we use the caller-provided
// ``precomputed_wall_s`` (the time their Python computation already paid).
// For R5 we use the bridge's reported wall_seconds. This matches the
// per-rung accounting the Python ladder reports in TpsReport today.

#include "nrl/ladder_native.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static double ladder_now_seconds(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double ladder_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

NRL_API int32_t nrl_v1_lane_allows_r2_active(int32_t lane) {
    return (lane == NRL_COHERENCE_LANE_FAST_BALANCED
            || lane == NRL_COHERENCE_LANE_MAX_THROUGHPUT) ? 1 : 0;
}

NRL_API const char *nrl_v1_ladder_rung_name(int32_t rung) {
    switch (rung) {
        case NRL_LADDER_RUNG_R0_MUSCLE_MEMORY: return "r0_muscle_memory";
        case NRL_LADDER_RUNG_R1_ZPM:           return "r1_zpm_nullspace";
        case NRL_LADDER_RUNG_R2_OMEGA_ACTIVE:  return "r2_omega_native_resolve";
        case NRL_LADDER_RUNG_R3_PREFILL:       return "r3_prefill_cache";
        case NRL_LADDER_RUNG_R4_LAYER_SKIP:    return "r4_layer_skip";
        case NRL_LADDER_RUNG_R5_LIBLLAMA:      return "r5_libllama";
        default:                               return "unknown";
    }
}

NRL_API int32_t nrl_v1_coherence_lane_from_str(const char *name) {
    if (name == NULL) return -1;
    if (strcmp(name, "fast-stable") == 0)    return NRL_COHERENCE_LANE_FAST_STABLE;
    if (strcmp(name, "fast-balanced") == 0)  return NRL_COHERENCE_LANE_FAST_BALANCED;
    if (strcmp(name, "max-throughput") == 0) return NRL_COHERENCE_LANE_MAX_THROUGHPUT;
    return -1;
}

NRL_API const char *nrl_v1_coherence_lane_name(int32_t lane) {
    switch (lane) {
        case NRL_COHERENCE_LANE_FAST_STABLE:    return "fast-stable";
        case NRL_COHERENCE_LANE_FAST_BALANCED:  return "fast-balanced";
        case NRL_COHERENCE_LANE_MAX_THROUGHPUT: return "max-throughput";
        default:                                return "unknown";
    }
}

/* Copy ``src`` (UTF-8, NUL-terminated) into ``out``'s output buffer.
 * Updates text_byte_len / text_truncated. Always NUL-terminates when the
 * buffer has at least one byte of capacity. */
static void ladder_copy_text(
    nrl_ladder_outputs *out,
    const char *src) {
    const char *s = src ? src : "";
    const size_t n = strlen(s);
    out->text_byte_len = n;
    if (out->text_buf == NULL || out->text_buf_capacity == 0) {
        out->text_truncated = (n > 0) ? 1 : 0;
        return;
    }
    const size_t cap = out->text_buf_capacity;
    const size_t writable = cap - 1;
    const size_t to_copy = (n <= writable) ? n : writable;
    if (to_copy > 0) {
        memcpy(out->text_buf, s, to_copy);
    }
    out->text_buf[to_copy] = '\0';
    out->text_truncated = (n > writable) ? 1 : 0;
}

NRL_API nrl_v1_status nrl_v1_ladder_resolve(
    const nrl_ladder_inputs *inputs,
    nrl_ladder_outputs *outputs) {
    if (inputs == NULL || outputs == NULL) {
        return NRL_ERR_ARGS;
    }
    outputs->served_rung = -1;
    outputs->tokens_emitted = 0;
    outputs->wall_seconds = 0.0;
    outputs->text_byte_len = 0;
    outputs->text_truncated = 0;
    outputs->served_class = 0;

    /* R0 muscle memory --------------------------------------------------- */
    if (inputs->r0.available && inputs->r0.tokens > 0) {
        outputs->served_rung = NRL_LADDER_RUNG_R0_MUSCLE_MEMORY;
        outputs->tokens_emitted = inputs->r0.tokens;
        outputs->wall_seconds = inputs->r0.precomputed_wall_s;
        outputs->served_class = 1;
        ladder_copy_text(outputs, inputs->r0.text_utf8);
        return NRL_OK;
    }

    /* R1 ZPM nullspace --------------------------------------------------- */
    if (inputs->r1.available && inputs->r1.tokens > 0) {
        outputs->served_rung = NRL_LADDER_RUNG_R1_ZPM;
        outputs->tokens_emitted = inputs->r1.tokens;
        outputs->wall_seconds = inputs->r1.precomputed_wall_s;
        outputs->served_class = 1;
        ladder_copy_text(outputs, inputs->r1.text_utf8);
        return NRL_OK;
    }

    /* R2 omega native resolve (active) ----------------------------------- */
    /* Active R2 is gated by lane + r2_shadow_enabled; the caller is
     * responsible for setting available=0 if Stage-VI failed. We re-check
     * the lane gate so the C ABI cannot be tricked into serving R2 on
     * fast-stable, even if a buggy caller passes available=1. */
    if (inputs->r2_active.available
        && inputs->r2_active.tokens > 0
        && inputs->r2_shadow_enabled
        && nrl_v1_lane_allows_r2_active(inputs->coherence_lane)) {
        outputs->served_rung = NRL_LADDER_RUNG_R2_OMEGA_ACTIVE;
        outputs->tokens_emitted = inputs->r2_active.tokens;
        outputs->wall_seconds = inputs->r2_active.precomputed_wall_s;
        outputs->served_class = 2;
        ladder_copy_text(outputs, inputs->r2_active.text_utf8);
        return NRL_OK;
    }

    /* R5 libllama bridge ------------------------------------------------- */
    nrl_llama_response resp;
    memset(&resp, 0, sizeof(resp));
    resp.text_buf = outputs->text_buf;
    resp.text_buf_capacity = outputs->text_buf_capacity;
    const double t0 = ladder_now_seconds();
    nrl_v1_status rc = nrl_v1_llama_run(&inputs->r5_request, &resp);
    const double t1 = ladder_now_seconds();
    outputs->served_rung = NRL_LADDER_RUNG_R5_LIBLLAMA;
    outputs->served_class = 3;
    outputs->tokens_emitted = resp.tokens_emitted;
    outputs->text_byte_len = resp.text_byte_len;
    outputs->text_truncated = resp.text_truncated;
    /* Prefer the bridge-reported wall (it has callback-internal scope) and
     * fall back to our own measurement on misbehaving callbacks. */
    outputs->wall_seconds = (resp.wall_seconds > 0.0)
        ? resp.wall_seconds : (t1 - t0);
    return rc;
}
