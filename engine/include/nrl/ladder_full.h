// Copyright (c) 2026 Daniel Harding - RomanAILabs
//
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
//
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// nrl/ladder_full.h - Phase 8-EG full-native hot path ABI.
//
// Phase 7-EG moved rung *dispatch* and the libllama bridge call into C, while
// leaving R0/R1/R2 candidate computation in Python for byte-parity reasons.
// Phase 8-EG removes the remaining Python hops from the common fast path:
//
//   * R0 (muscle memory)  - file read + FNV-1a64 keying now runs in C.
//   * R1 (ZPM nullspace)  - 256-bit anchor + on-disk index scan + Stage VI
//                           verify now run in C.
//   * R2 (omega native)   - dispatched through a registered callback using
//                           the same proven pattern as the libllama bridge
//                           so the C ladder owns control flow and timing.
//                           Phase 9-EG replaces this callback with a direct
//                           C port of ``_run_omega_probe``; the header and
//                           ABI will not change.
//   * R5 (libllama)       - unchanged from Phase 7-EG bridge.
//
// All strings crossing the ABI are UTF-8 byte buffers. Callers own every
// pointer they pass in; the implementation never allocates memory the
// caller must free. Output buffers are caller-owned (sized by the caller).
//
// The full-turn API is intentionally one call per turn: it bundles the
// muscle-memory probe, ZPM probe, optional R2 probe, and R5 bridge call
// into a single C entry point so Python never sits in the hot loop. It
// *does not* own the sha256 / prompt shaping / evidence serialization
// steps, which stay in the Python runner.

#ifndef NRL_LADDER_FULL_H_
#define NRL_LADDER_FULL_H_

#include <stddef.h>
#include <stdint.h>

#include "nrl/ladder_native.h"
#include "nrl/llama_bridge.h"
#include "nrl/nrl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================== *
 * R0 muscle memory — direct, caller-scoped                                  *
 * ======================================================================== */

/* Muscle-memory probe request. Must fully determine the FNV-1a64 key; every
 * field is serialized into the key blob in the same order
 * ``nrlpy.gguf._muscle_memory_key`` uses (separator = ``\x1f``). Strings
 * must be NUL-terminated UTF-8. */
typedef struct nrl_mm_lookup_request {
    const char *root_dir_utf8;         /* base dir: $NRL_ROOT/cache/mm or cwd/cache/mm */
    const char *model_sha256_utf8;     /* manifest.model_sha256 (or "unknown") */
    const char *prompt_utf8;           /* manifest.prompt */
    const char *sampler_fingerprint_utf8; /* manifest.sampler_fingerprint() */
    int32_t seed;
    int32_t max_tokens;
    int32_t muscle_memory_on;          /* 0 => lookup is a no-op miss (mode=off). */
} nrl_mm_lookup_request;

/* Muscle-memory probe result. ``text_buf`` is caller-owned; on hit the
 * file body is copied verbatim. ``hit`` is 0 on miss / mode-off / bad
 * file header / decode error. ``key_fnv1a64`` is always populated so
 * Python can persist the key even on a miss (used for later stores). */
typedef struct nrl_mm_lookup_result {
    int32_t hit;
    int32_t tokens_emitted;
    uint64_t key_fnv1a64;
    double wall_seconds;
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
} nrl_mm_lookup_result;

/* Run a native muscle-memory probe. Never raises; returns NRL_OK with
 * ``result->hit == 0`` on every miss. */
NRL_API nrl_v1_status nrl_v1_mm_lookup(
    const nrl_mm_lookup_request *request,
    nrl_mm_lookup_result *result);

/* ======================================================================== *
 * R1 ZPM nullspace — anchor + on-disk index + Stage VI                      *
 * ======================================================================== */

/* Subset of manifest fields that feed ``_zpm_anchor_bytes`` in the exact
 * byte order Python uses (separator = 0x1e, RS). Swapping order or adding
 * fields here WILL invalidate every stored unity state on disk, so this
 * struct is a hard ABI commitment. */
typedef struct nrl_zpm_lookup_request {
    const char *index_path_utf8;   /* $NRL_ROOT/cache/zpm/<sha>/index.bin */
    const char *model_sha256_utf8;
    const char *prompt_utf8;
    const char *sampler_fingerprint_utf8;
    int32_t seed;
    int32_t max_tokens;
    int32_t threshold_bits;        /* manifest.zpm_threshold_bits */
    int32_t enabled;               /* manifest.zpm_nullspace */
} nrl_zpm_lookup_request;

/* ZPM probe result. On hit the stored reply text is copied into
 * ``text_buf`` and ``tokens_emitted`` is populated with the stored
 * token count. ``distance_bits`` is the nearest-neighbor Hamming
 * distance (0 == exact); it is always populated so callers can log
 * auditing numbers even on a miss. */
typedef struct nrl_zpm_lookup_result {
    int32_t hit;
    int32_t tokens_emitted;
    int32_t exact;                 /* distance_bits == 0 */
    int32_t within_threshold;      /* distance_bits <= threshold_bits */
    int32_t distance_bits;
    int32_t stored_entry_index;    /* -1 on miss */
    uint64_t state[4];             /* the 256-bit query anchor */
    double wall_seconds;
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
} nrl_zpm_lookup_result;

/* Run a native ZPM probe. Anchors, loads the on-disk index, performs
 * nullspace search + Stage VI verify, all without entering Python. */
NRL_API nrl_v1_status nrl_v1_zpm_lookup(
    const nrl_zpm_lookup_request *request,
    nrl_zpm_lookup_result *result);

/* ======================================================================== *
 * R2 bridge callback — same proven pattern as the libllama bridge           *
 * ======================================================================== */

/* The C ladder calls this once per turn when R2 is eligible. The callback
 * must fill ``available`` (0 on miss / lane disallow / Stage-VI fail),
 * ``tokens_emitted`` on hit, and copy the served text into ``text_buf``.
 * Returning a non-NRL_OK status is propagated unchanged. */
typedef struct nrl_r2_probe_request {
    int32_t coherence_lane;
    int32_t r2_shadow_enabled;
    int32_t zpm_threshold_bits;
    int32_t omega_iterations;
    int32_t omega_candidates;
    double omega_budget_ms;
    const char *model_sha256_utf8;
    const uint8_t *intent_anchor_bytes;
    size_t intent_anchor_len;
} nrl_r2_probe_request;

typedef struct nrl_r2_probe_response {
    int32_t available;
    int32_t tokens_emitted;
    int32_t stored_entry_index;
    int32_t distance_bits;
    double wall_seconds;
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
} nrl_r2_probe_response;

typedef nrl_v1_status (*nrl_r2_probe_callback_fn)(
    void *user_data,
    const nrl_r2_probe_request *request,
    nrl_r2_probe_response *response);

/* Install / clear the R2 probe callback. When cleared, R2 is always
 * reported as a miss and the ladder falls through to R5. */
NRL_API nrl_v1_status nrl_v1_r2_set_callback(
    nrl_r2_probe_callback_fn cb,
    void *user_data);

/* True iff an R2 callback is currently registered. */
NRL_API int32_t nrl_v1_r2_has_callback(void);

/* ======================================================================== *
 * Full-turn orchestrator                                                    *
 * ======================================================================== */

/* Inputs for a single full-native turn. Mirrors ``_run_gguf_native`` but
 * with the candidate-computation fields inlined so Python never sits in
 * the hot path. */
typedef struct nrl_full_turn_request {
    /* R0 muscle memory. */
    nrl_mm_lookup_request mm;
    /* R1 ZPM nullspace. */
    nrl_zpm_lookup_request zpm;
    /* Lane + flags shared with Phase 7-EG ABI. */
    int32_t coherence_lane;
    int32_t r2_shadow_enabled;
    int32_t zpm_threshold_bits;
    int32_t omega_iterations;
    int32_t omega_candidates;
    double omega_budget_ms;
    /* Anchor bytes the Python runner computed once before the turn.
     * R2 uses this; R0/R1 derive their keys from the request fields
     * above so they are independent of ``intent_anchor_bytes``. */
    const uint8_t *intent_anchor_bytes;
    size_t intent_anchor_len;
    /* R5 libllama bridge request (unchanged from Phase 7-EG). */
    nrl_llama_request r5_request;
} nrl_full_turn_request;

/* Output of a single full-native turn. ``served_rung`` matches the
 * Phase 7-EG rung identifiers; ``text_buf`` is caller-owned and receives
 * the chosen rung's reply. Per-rung sub-reports are populated so the
 * Python runner can write the same evidence banner with zero behavioral
 * change. */
typedef struct nrl_full_turn_result {
    int32_t served_rung;
    int32_t tokens_emitted;
    double wall_seconds;
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
    /* Sub-reports (populated even when another rung served, so the
     * Python evidence log keeps its honest-accounting fields). */
    nrl_mm_lookup_result mm_report;
    nrl_zpm_lookup_result zpm_report;
    int32_t r2_available;
    int32_t r2_tokens_emitted;
    int32_t r2_stored_entry_index;
    int32_t r2_distance_bits;
    double r2_wall_seconds;
    /* Populated when R5 served; mirrors nrl_llama_response. */
    int32_t r5_invoked;
    int32_t r5_tokens_emitted;
    double r5_wall_seconds;
} nrl_full_turn_result;

/* Drive one full turn through the native hot path. Decision order:
 *
 *   1. R0 muscle memory hit    -> serve, skip R1..R5.
 *   2. R1 ZPM nullspace hit    -> serve, skip R2..R5.
 *   3. R2 probe (if eligible)  -> serve on non-demoted hit, else fall through.
 *   4. R5 libllama bridge call -> always serves if R0..R2 missed.
 *
 * ``request->mm.muscle_memory_on == 0`` skips R0. ``request->zpm.enabled == 0``
 * skips R1. R2 is skipped when no callback is registered or when the lane
 * gate denies it. The caller is responsible for building per-rung output
 * buffers; the result's ``text_buf_capacity`` must be large enough to hold
 * the served reply for the rung that wins. */
NRL_API nrl_v1_status nrl_v1_ladder_run_turn(
    const nrl_full_turn_request *request,
    nrl_full_turn_result *result);

#ifdef __cplusplus
}
#endif

#endif /* NRL_LADDER_FULL_H_ */
