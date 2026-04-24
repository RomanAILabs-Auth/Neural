// Copyright (c) 2026 Daniel Harding - RomanAILabs
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// nrl/ladder_native.h - Phase 7-EG native Resolution Ladder ABI.
//
// Implements the §4.2 Resolution Ladder (R0..R5) in C. The ladder is a
// thin dispatcher: the deterministic R0/R1/R2 candidate computation
// continues to live in nrlpy.lmo / nrlpy.gguf for the parity-gate window,
// because byte-identical evidence is the Phase 7-EG release contract. The
// native ladder receives pre-computed candidates as *inputs* and chooses
// which rung to ratify; on R5 it drives the libllama bridge directly.
//
// All wide strings are UTF-8. Pointers passed in are caller-owned for the
// duration of the call. The native ladder never allocates memory the caller
// must free; output buffers are caller-owned (sized by the caller).
//
// Rung identifiers (kept as small integers so the ABI is stable across
// language bindings):
//   NRL_LADDER_RUNG_R0_MUSCLE_MEMORY = 0
//   NRL_LADDER_RUNG_R1_ZPM           = 1
//   NRL_LADDER_RUNG_R2_OMEGA_ACTIVE  = 2
//   NRL_LADDER_RUNG_R3_PREFILL       = 3   (skeleton today)
//   NRL_LADDER_RUNG_R4_LAYER_SKIP    = 4   (skeleton today)
//   NRL_LADDER_RUNG_R5_LIBLLAMA      = 5

#ifndef NRL_LADDER_NATIVE_H_
#define NRL_LADDER_NATIVE_H_

#include <stddef.h>
#include <stdint.h>

#include "nrl/llama_bridge.h"
#include "nrl/nrl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NRL_LADDER_RUNG_R0_MUSCLE_MEMORY 0
#define NRL_LADDER_RUNG_R1_ZPM           1
#define NRL_LADDER_RUNG_R2_OMEGA_ACTIVE  2
#define NRL_LADDER_RUNG_R3_PREFILL       3
#define NRL_LADDER_RUNG_R4_LAYER_SKIP    4
#define NRL_LADDER_RUNG_R5_LIBLLAMA      5

/* Coherence lanes (mirror nrlpy.lmo.COHERENCE_LANES). Numeric ABI. */
#define NRL_COHERENCE_LANE_FAST_STABLE     0
#define NRL_COHERENCE_LANE_FAST_BALANCED   1
#define NRL_COHERENCE_LANE_MAX_THROUGHPUT  2

/* Pre-computed rung candidate. ``available == 0`` means the rung had no
 * candidate this turn (cache miss / lane disallow / Stage-VI fail / etc.).
 * ``text_utf8`` is a NUL-terminated UTF-8 buffer owned by the caller for
 * the duration of nrl_v1_ladder_resolve. */
typedef struct nrl_ladder_candidate {
    int32_t available;
    int32_t tokens;
    const char *text_utf8;       /* may be NULL when available==0 */
    double precomputed_wall_s;   /* wall already spent computing this candidate */
} nrl_ladder_candidate;

/* Inputs to a single ladder turn. */
typedef struct nrl_ladder_inputs {
    int32_t coherence_lane;            /* one of NRL_COHERENCE_LANE_* */
    int32_t r2_shadow_enabled;         /* mirrors GgufManifest.r2_shadow_enabled */
    nrl_ladder_candidate r0;           /* muscle memory */
    nrl_ladder_candidate r1;           /* zpm nullspace */
    nrl_ladder_candidate r2_active;    /* omega native resolve (active mode) */
    nrl_llama_request r5_request;      /* used when rungs R0..R4 all miss */
} nrl_ladder_inputs;

/* Output of a single ladder turn. ``text_buf`` is caller-owned and at least
 * ``text_buf_capacity`` bytes; the chosen rung's reply (or libllama's
 * stream) is copied here. Truncation is reported but never silently
 * corrupted. */
typedef struct nrl_ladder_outputs {
    int32_t served_rung;
    int32_t tokens_emitted;
    double wall_seconds;
    char *text_buf;
    size_t text_buf_capacity;
    size_t text_byte_len;
    int32_t text_truncated;
    /* Stable ABI-numbered status. 0 == NRL_OK.
     *   1 == cache hit (R0/R1)
     *   2 == R2 active served (omega native resolve)
     *   3 == R5 served via bridge */
    int32_t served_class;
} nrl_ladder_outputs;

/* Drive one turn through the §4.2 Resolution Ladder.
 *
 * Decision order, mirrors nrlpy.gguf.run_gguf exactly:
 *   1. R0 muscle memory hit   -> served_rung = R0, no bridge call
 *   2. R1 ZPM nullspace hit   -> served_rung = R1, no bridge call
 *   3. R2 active service      -> served_rung = R2, no bridge call (gated by
 *                                lane + r2_shadow_enabled, candidate must
 *                                report available==1)
 *   4. fall through to R5     -> served_rung = R5, bridge call via
 *                                nrl_v1_llama_run.
 *
 * R3/R4 are reserved skeleton rungs in this phase; the ladder does not
 * select them. They are present in the ABI so future phases can attach
 * candidate slots without bumping the soname.
 *
 * Returns NRL_OK on success. Any bridge error is propagated unchanged. */
NRL_API nrl_v1_status nrl_v1_ladder_resolve(
    const nrl_ladder_inputs *inputs,
    nrl_ladder_outputs *outputs);

/* Lane-allow predicate (mirrors nrlpy.lmo.lane_allows_r2_active). */
NRL_API int32_t nrl_v1_lane_allows_r2_active(int32_t lane);

/* String identifier for a rung; returned pointer is static. */
NRL_API const char *nrl_v1_ladder_rung_name(int32_t rung);

/* Parse / format a coherence-lane label. Returns -1 on unknown lane. */
NRL_API int32_t nrl_v1_coherence_lane_from_str(const char *name);
NRL_API const char *nrl_v1_coherence_lane_name(int32_t lane);

#ifdef __cplusplus
}
#endif

#endif /* NRL_LADDER_NATIVE_H_ */
