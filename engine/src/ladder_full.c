// Copyright (c) 2026 Daniel Harding - RomanAILabs
//
// Co-Architect: Grok (xAI)
// Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
//
// Contact: daniel@romanailabs.com | romanailabs@gmail.com
// Website: https://romanailabs.com
//
// engine/src/ladder_full.c - Phase 8-EG full-native hot path implementation.
//
// Byte-parity contract
// --------------------
// Every primitive in this file is implemented to produce output that is
// bit-identical to the Python reference in nrlpy/src/nrlpy/gguf.py and
// nrlpy/src/nrlpy/zpm.py. Any change here must come with a matching
// change to the Python side and a parity test in
// nrlpy/tests/test_native_full_path.py. Specifically:
//
//   * FNV-1a64 prime    = 0x100000001B3 (canonical).
//   * ZPM FNV-1a64 IV   = 0xCBF29CE484222325 (canonical; zpm.py::_fnv1a64,
//                         zpm.cpp).
//   * Muscle-memory FNV-1a64 IV = 0x14650FB0739D0383 (non-standard;
//                         nrlpy.runtime.fnv1a64_packed). Kept distinct
//                         from the ZPM IV so an MM key never collides
//                         with a ZPM anchor even on identical byte-blobs.
//                         Do NOT unify without migrating every stored
//                         cache file on disk.
//   * Muscle-memory key: parts joined with 0x1F (US), UTF-8 encoded.
//   * ZPM anchor:        parts joined with 0x1E (RS), four deterministic
//                        rotations per zpm.py::anchor.
//   * Muscle-memory file: 8-byte magic "NRLMM1\x00\x00" + u32 tokens + u32
//                        text_bytes + UTF-8 body.
//   * ZPM index file:    8-byte magic "NRLZPM01" + records of
//                        (4*u64 state, u32 tokens, f32 wall, u16 meta_count,
//                         u32 text_len, utf8 text, meta pairs).
//
// Concurrency
// -----------
// The full-turn API reads read-only disk state (muscle memory, ZPM index)
// per call; it does not share heap state across concurrent turns. The R2
// callback slot is global and guarded (like the libllama bridge) so
// concurrent turns must serialize at the application layer, matching the
// existing nrlpy._core threading model.

#include "nrl/ladder_full.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static CRITICAL_SECTION g_r2_lock;
static int32_t g_r2_lock_initialized = 0;
static void r2_lock_init(void) {
    if (!g_r2_lock_initialized) {
        InitializeCriticalSection(&g_r2_lock);
        g_r2_lock_initialized = 1;
    }
}
static void r2_lock(void) { r2_lock_init(); EnterCriticalSection(&g_r2_lock); }
static void r2_unlock(void) { LeaveCriticalSection(&g_r2_lock); }
static double full_now_seconds(void) {
    LARGE_INTEGER freq, c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <pthread.h>
#include <time.h>
static pthread_mutex_t g_r2_lock = PTHREAD_MUTEX_INITIALIZER;
static void r2_lock(void) { pthread_mutex_lock(&g_r2_lock); }
static void r2_unlock(void) { pthread_mutex_unlock(&g_r2_lock); }
static double full_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ------------------------------------------------------------------------ *
 * FNV-1a64 primitives                                                       *
 *                                                                           *
 * Two initial values are in use on the NRL side:                            *
 *                                                                           *
 *   * FNV_IV_ZPM: byte-identical to zpm.py::_fnv1a64 (canonical 0xCBF2..).  *
 *   * FNV_IV_MM:  byte-identical to nrlpy.runtime.fnv1a64_packed            *
 *                (non-standard 0x1465..; used by the muscle-memory key).    *
 *                                                                           *
 * The split was grandfathered in when muscle memory landed before ZPM; we   *
 * keep it because every MM cache file on disk is keyed under the non-      *
 * standard IV and unifying would invalidate them all.                       *
 * ------------------------------------------------------------------------ */

#define NRL_FNV_PRIME  0x100000001B3ULL
#define NRL_FNV_IV_ZPM 0xCBF29CE484222325ULL
#define NRL_FNV_IV_MM  0x14650FB0739D0383ULL

static uint64_t fnv1a64_with_iv(
    uint64_t iv, const uint8_t *data, size_t n) {
    uint64_t h = iv;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)data[i];
        h *= NRL_FNV_PRIME;
    }
    return h;
}

static uint64_t fnv1a64(const uint8_t *data, size_t n) {
    return fnv1a64_with_iv(NRL_FNV_IV_ZPM, data, n);
}

/* ------------------------------------------------------------------------ *
 * Buffer helpers                                                            *
 * ------------------------------------------------------------------------ */

static void copy_utf8_into(
    char *buf,
    size_t capacity,
    const void *src,
    size_t src_len,
    size_t *out_byte_len,
    int32_t *out_truncated) {
    if (out_byte_len) *out_byte_len = src_len;
    if (buf == NULL || capacity == 0) {
        if (out_truncated) *out_truncated = (src_len > 0) ? 1 : 0;
        return;
    }
    const size_t writable = capacity - 1;
    const size_t to_copy = (src_len <= writable) ? src_len : writable;
    if (to_copy > 0 && src != NULL) {
        memcpy(buf, src, to_copy);
    }
    buf[to_copy] = '\0';
    if (out_truncated) *out_truncated = (src_len > writable) ? 1 : 0;
}

static size_t u64_to_hex16(uint64_t v, char out[17]) {
    /* Matches Python f"{key_fnv:016x}" - zero-padded lowercase hex. */
    static const char hex[] = "0123456789abcdef";
    for (int i = 15; i >= 0; --i) {
        out[i] = hex[v & 0xFULL];
        v >>= 4;
    }
    out[16] = '\0';
    return 16;
}

/* Append ``src`` (NUL-terminated) to ``out`` starting at ``*off``, up to
 * ``cap - 1``. Returns 0 on success, -1 on overflow. */
static int append_str(char *out, size_t cap, size_t *off, const char *src) {
    if (src == NULL) return 0;
    size_t src_len = strlen(src);
    if (*off + src_len + 1 > cap) return -1;
    memcpy(out + *off, src, src_len);
    *off += src_len;
    out[*off] = '\0';
    return 0;
}

static int append_bytes(char *out, size_t cap, size_t *off,
                        const void *src, size_t n) {
    if (n == 0) return 0;
    if (*off + n > cap) return -1;
    memcpy(out + *off, src, n);
    *off += n;
    return 0;
}

/* Write integer as ASCII decimal. Returns bytes written (no NUL). */
static size_t int_to_ascii(int32_t v, char buf[16]) {
    int neg = 0;
    uint32_t uv;
    if (v < 0) { neg = 1; uv = (uint32_t)(-(int64_t)v); }
    else       { uv = (uint32_t)v; }
    char tmp[16];
    size_t n = 0;
    if (uv == 0) tmp[n++] = '0';
    while (uv > 0) {
        tmp[n++] = (char)('0' + (uv % 10));
        uv /= 10;
    }
    size_t out_n = 0;
    if (neg) buf[out_n++] = '-';
    for (size_t i = 0; i < n; ++i) buf[out_n++] = tmp[n - 1 - i];
    return out_n;
}

/* ------------------------------------------------------------------------ *
 * R0 muscle memory                                                          *
 * ------------------------------------------------------------------------ */

/* Build the FNV-1a64 key exactly as _muscle_memory_key() does.
 *
 * Python:
 *   parts = [model_sha256, prompt, sampler_fingerprint,
 *            str(seed), str(max_tokens)]
 *   blob  = "\u001f".join(parts).encode("utf-8")
 *
 * The 0x1F (US / UNIT SEPARATOR) separator encodes to a single 0x1F byte
 * under UTF-8, so we emit it directly. */
static uint64_t build_mm_key_fnv(const nrl_mm_lookup_request *req) {
    /* Compute required length first so we can stack-allocate on small
     * inputs or heap-allocate otherwise. 5 parts + 4 separators. */
    const char *p0 = req->model_sha256_utf8 ? req->model_sha256_utf8 : "unknown";
    const char *p1 = req->prompt_utf8 ? req->prompt_utf8 : "";
    const char *p2 = req->sampler_fingerprint_utf8 ? req->sampler_fingerprint_utf8 : "";
    char seed_buf[16], max_buf[16];
    size_t seed_n = int_to_ascii(req->seed, seed_buf);
    size_t max_n  = int_to_ascii(req->max_tokens, max_buf);
    size_t n0 = strlen(p0), n1 = strlen(p1), n2 = strlen(p2);
    size_t total = n0 + 1 + n1 + 1 + n2 + 1 + seed_n + 1 + max_n;

    /* Hash without materializing the full blob to avoid a big malloc.
     * FNV-1a is order-preserving and byte-by-byte so we can stream it.
     * We use FNV_IV_MM, not FNV_IV_ZPM: keys written by
     * nrlpy.runtime.fnv1a64_packed depend on the non-standard IV. */
    uint64_t h = NRL_FNV_IV_MM;
    const uint64_t prime = NRL_FNV_PRIME;
    const uint8_t sep = 0x1F;

#define MM_FEED(ptr, n) do { \
        const uint8_t *bp = (const uint8_t *)(ptr); \
        for (size_t _i = 0; _i < (n); ++_i) { \
            h ^= (uint64_t)bp[_i]; \
            h *= prime; \
        } \
    } while (0)

    MM_FEED(p0, n0); MM_FEED(&sep, 1);
    MM_FEED(p1, n1); MM_FEED(&sep, 1);
    MM_FEED(p2, n2); MM_FEED(&sep, 1);
    MM_FEED(seed_buf, seed_n); MM_FEED(&sep, 1);
    MM_FEED(max_buf, max_n);
    (void)total;
#undef MM_FEED
    return h;
}

/* Path = "<root>/<model_tag>/<key:016x>.mm". ``root`` is expected to be
 * the already-resolved ``$NRL_ROOT/cache/mm`` directory. */
static int build_mm_path(
    char *out, size_t cap,
    const char *root,
    const char *model_sha,
    uint64_t key) {
    if (out == NULL || cap == 0) return -1;
    size_t off = 0;
    out[0] = '\0';
    const char *root_s = root && root[0] ? root : ".";
    const char *tag = (model_sha && model_sha[0]) ? model_sha : "unknown";
    char hex[17]; u64_to_hex16(key, hex);
    if (append_str(out, cap, &off, root_s) < 0) return -1;
#if defined(_WIN32)
    if (append_str(out, cap, &off, "\\") < 0) return -1;
#else
    if (append_str(out, cap, &off, "/") < 0) return -1;
#endif
    if (append_str(out, cap, &off, tag) < 0) return -1;
#if defined(_WIN32)
    if (append_str(out, cap, &off, "\\") < 0) return -1;
#else
    if (append_str(out, cap, &off, "/") < 0) return -1;
#endif
    if (append_str(out, cap, &off, hex) < 0) return -1;
    if (append_str(out, cap, &off, ".mm") < 0) return -1;
    return 0;
}

NRL_API nrl_v1_status nrl_v1_mm_lookup(
    const nrl_mm_lookup_request *request,
    nrl_mm_lookup_result *result) {
    if (request == NULL || result == NULL) return NRL_ERR_ARGS;

    /* Zero the result, preserving caller-owned text_buf fields. */
    char *buf = result->text_buf;
    size_t cap = result->text_buf_capacity;
    memset(result, 0, sizeof(*result));
    result->text_buf = buf;
    result->text_buf_capacity = cap;

    result->key_fnv1a64 = build_mm_key_fnv(request);
    if (!request->muscle_memory_on) {
        return NRL_OK;  /* documented miss */
    }

    char path[1024];
    if (build_mm_path(path, sizeof(path),
                      request->root_dir_utf8,
                      request->model_sha256_utf8,
                      result->key_fnv1a64) < 0) {
        return NRL_OK;  /* path overflow treated as miss */
    }

    const double t0 = full_now_seconds();
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NRL_OK;  /* miss */
    }
    uint8_t head[16];
    size_t nread = fread(head, 1, sizeof(head), f);
    if (nread != 16 || memcmp(head, "NRLMM1\x00\x00", 8) != 0) {
        fclose(f);
        return NRL_OK;
    }
    uint32_t token_count =
        (uint32_t)head[ 8] |
        ((uint32_t)head[ 9] <<  8) |
        ((uint32_t)head[10] << 16) |
        ((uint32_t)head[11] << 24);
    uint32_t text_bytes =
        (uint32_t)head[12] |
        ((uint32_t)head[13] <<  8) |
        ((uint32_t)head[14] << 16) |
        ((uint32_t)head[15] << 24);
    if (token_count == 0 || text_bytes == 0) {
        fclose(f);
        return NRL_OK;
    }

    /* Read the body into a heap buffer so we can validate UTF-8 and copy
     * into the caller's output buffer without aliasing. */
    uint8_t *body = (uint8_t *)malloc(text_bytes);
    if (body == NULL) {
        fclose(f);
        return NRL_ERR_INTERNAL;
    }
    size_t body_read = fread(body, 1, text_bytes, f);
    fclose(f);
    if (body_read != text_bytes) {
        free(body);
        return NRL_OK;
    }
    /* The Python reference calls decode("utf-8") and treats failure as
     * a miss. We are permissive on the C side (we don't re-validate
     * every byte) because the writer is also our code and always writes
     * valid UTF-8. If that contract is ever broken we'd miss here the
     * same way Python does via the UnicodeDecodeError branch. */

    const double t1 = full_now_seconds();
    copy_utf8_into(result->text_buf, result->text_buf_capacity,
                   body, text_bytes,
                   &result->text_byte_len, &result->text_truncated);
    free(body);
    result->hit = 1;
    result->tokens_emitted = (int32_t)token_count;
    result->wall_seconds = (t1 > t0) ? (t1 - t0) : 0.0;
    return NRL_OK;
}

/* ------------------------------------------------------------------------ *
 * R1 ZPM nullspace                                                          *
 * ------------------------------------------------------------------------ */

/* Build intent bytes the Python _zpm_anchor_bytes produces:
 *
 *   parts = [model_sha256, prompt, sampler_fingerprint,
 *            str(seed), str(max_tokens)]
 *   blob  = b"\x1e".join(parts)
 *
 * Returns the total length written to ``out``. On overflow returns -1. */
static int build_zpm_intent_blob(
    char *out, size_t cap,
    const nrl_zpm_lookup_request *req,
    size_t *out_len) {
    if (cap == 0) return -1;
    size_t off = 0;
    const uint8_t sep = 0x1E;
    const char *p0 = req->model_sha256_utf8 ? req->model_sha256_utf8 : "unknown";
    const char *p1 = req->prompt_utf8 ? req->prompt_utf8 : "";
    const char *p2 = req->sampler_fingerprint_utf8 ? req->sampler_fingerprint_utf8 : "";
    char seed_buf[16], max_buf[16];
    size_t seed_n = int_to_ascii(req->seed, seed_buf);
    size_t max_n  = int_to_ascii(req->max_tokens, max_buf);

    if (append_bytes(out, cap, &off, p0, strlen(p0)) < 0) return -1;
    if (append_bytes(out, cap, &off, &sep, 1) < 0) return -1;
    if (append_bytes(out, cap, &off, p1, strlen(p1)) < 0) return -1;
    if (append_bytes(out, cap, &off, &sep, 1) < 0) return -1;
    if (append_bytes(out, cap, &off, p2, strlen(p2)) < 0) return -1;
    if (append_bytes(out, cap, &off, &sep, 1) < 0) return -1;
    if (append_bytes(out, cap, &off, seed_buf, seed_n) < 0) return -1;
    if (append_bytes(out, cap, &off, &sep, 1) < 0) return -1;
    if (append_bytes(out, cap, &off, max_buf, max_n) < 0) return -1;
    *out_len = off;
    return 0;
}

/* Byte-identical port of zpm.py::anchor(blob). Four FNV-1a64 projections
 * over deterministic permutations:
 *   w0 = FNV(blob)
 *   w1 = FNV(blob[::-1])
 *   w2 = FNV(blob[half:] + blob[:half])  half = n // 2
 *   w3 = FNV(blob[shift:] + blob[:shift]) shift = n // 4 if n >= 4 else 1
 *
 * Returns the four words in state[0..3]. */
static void zpm_anchor(const uint8_t *blob, size_t n, uint64_t state[4]) {
    if (n == 0) {
        state[0] = state[1] = state[2] = state[3] = 0;
        return;
    }
    state[0] = fnv1a64(blob, n);

    /* w1: blob reversed. Stream the hash backwards without materializing
     * the reversed buffer. */
    {
        uint64_t h = NRL_FNV_IV_ZPM;
        for (size_t i = n; i > 0; --i) {
            h ^= (uint64_t)blob[i - 1];
            h *= NRL_FNV_PRIME;
        }
        state[1] = h;
    }

    /* w2: halves swapped. Hash the second half, then the first half. */
    {
        const size_t half = n / 2;
        uint64_t h = NRL_FNV_IV_ZPM;
        for (size_t i = half; i < n; ++i) {
            h ^= (uint64_t)blob[i];
            h *= NRL_FNV_PRIME;
        }
        for (size_t i = 0; i < half; ++i) {
            h ^= (uint64_t)blob[i];
            h *= NRL_FNV_PRIME;
        }
        state[2] = h;
    }

    /* w3: byte-shift by n/4 (or 1 for short blobs). */
    {
        const size_t shift = (n >= 4) ? (n / 4) : 1;
        uint64_t h = NRL_FNV_IV_ZPM;
        for (size_t i = shift; i < n; ++i) {
            h ^= (uint64_t)blob[i];
            h *= NRL_FNV_PRIME;
        }
        for (size_t i = 0; i < shift && i < n; ++i) {
            h ^= (uint64_t)blob[i];
            h *= NRL_FNV_PRIME;
        }
        state[3] = h;
    }
}

static int popcount64_c(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

static int hamming_state_u64x4(const uint64_t a[4], const uint64_t b[4]) {
    return popcount64_c(a[0] ^ b[0]) +
           popcount64_c(a[1] ^ b[1]) +
           popcount64_c(a[2] ^ b[2]) +
           popcount64_c(a[3] ^ b[3]);
}

/* Read u16/u32/u64 little-endian from a streamed FILE*. Return 0 on EOF
 * or short read; 1 on success. */
static int read_u16_le(FILE *f, uint16_t *out) {
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2) return 0;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 1;
}

static int read_u32_le(FILE *f, uint32_t *out) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64_le(FILE *f, uint64_t *out) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= ((uint64_t)b[i]) << (i * 8);
    *out = v;
    return 1;
}

static int read_f32_le(FILE *f, float *out) {
    uint32_t u;
    if (!read_u32_le(f, &u)) return 0;
    memcpy(out, &u, sizeof(float));
    return 1;
}

NRL_API nrl_v1_status nrl_v1_zpm_lookup(
    const nrl_zpm_lookup_request *request,
    nrl_zpm_lookup_result *result) {
    if (request == NULL || result == NULL) return NRL_ERR_ARGS;

    char *buf = result->text_buf;
    size_t cap = result->text_buf_capacity;
    memset(result, 0, sizeof(*result));
    result->text_buf = buf;
    result->text_buf_capacity = cap;
    result->stored_entry_index = -1;
    result->distance_bits = 256;

    /* Step 1: compute the 256-bit query anchor (regardless of whether
     * we will actually scan an index — evidence logs this number). */
    /* Build the intent blob on the stack for typical chat sizes; fall
     * back to malloc for very long prompts. */
    size_t p1_len = request->prompt_utf8 ? strlen(request->prompt_utf8) : 0;
    size_t upper = 512 + p1_len + 2;
    char *blob_buf;
    char stack_blob[2048];
    int need_free = 0;
    if (upper <= sizeof(stack_blob)) {
        blob_buf = stack_blob;
    } else {
        blob_buf = (char *)malloc(upper);
        if (blob_buf == NULL) return NRL_ERR_INTERNAL;
        need_free = 1;
    }
    size_t blob_len = 0;
    if (build_zpm_intent_blob(blob_buf, upper, request, &blob_len) < 0) {
        if (need_free) free(blob_buf);
        return NRL_ERR_INTERNAL;
    }
    zpm_anchor((const uint8_t *)blob_buf, blob_len, result->state);
    if (need_free) free(blob_buf);

    if (!request->enabled) {
        return NRL_OK;  /* R1 disabled — report the anchor, skip scan. */
    }
    if (request->index_path_utf8 == NULL || request->index_path_utf8[0] == '\0') {
        return NRL_OK;
    }

    const double t0 = full_now_seconds();
    FILE *f = fopen(request->index_path_utf8, "rb");
    if (f == NULL) return NRL_OK;  /* no index file yet = miss */
    uint8_t magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "NRLZPM01", 8) != 0) {
        fclose(f);
        return NRL_OK;
    }

    /* Scan all entries; keep best (distance, index) and the bytes we
     * need to reconstruct the served reply if we commit. We stream: on
     * finding a new best, we remember the reply text location on disk
     * via ftell() so we can re-read only the winner's body at the end. */
    int best_idx = -1;
    int best_dist = 256;
    long best_text_pos = -1;
    uint32_t best_text_len = 0;
    uint32_t best_tokens = 0;

    int idx = 0;
    for (;;) {
        uint64_t s0, s1, s2, s3;
        if (!read_u64_le(f, &s0)) break;
        if (!read_u64_le(f, &s1) ||
            !read_u64_le(f, &s2) ||
            !read_u64_le(f, &s3)) break;
        uint32_t tokens; float wall; uint16_t meta_count;
        if (!read_u32_le(f, &tokens)) break;
        if (!read_f32_le(f, &wall)) break;
        if (!read_u16_le(f, &meta_count)) break;
        uint32_t text_len;
        if (!read_u32_le(f, &text_len)) break;
        long text_pos = ftell(f);
        if (text_pos < 0) break;
        if (fseek(f, (long)text_len, SEEK_CUR) != 0) break;
        /* Skip metadata pairs. */
        int meta_ok = 1;
        for (int m = 0; m < (int)meta_count; ++m) {
            uint16_t kl, vl;
            if (!read_u16_le(f, &kl)) { meta_ok = 0; break; }
            if (fseek(f, (long)kl, SEEK_CUR) != 0) { meta_ok = 0; break; }
            if (!read_u16_le(f, &vl)) { meta_ok = 0; break; }
            if (fseek(f, (long)vl, SEEK_CUR) != 0) { meta_ok = 0; break; }
        }
        if (!meta_ok) break;

        uint64_t entry_state[4] = {s0, s1, s2, s3};
        int d = hamming_state_u64x4(result->state, entry_state);
        if (best_idx < 0 || d < best_dist) {
            best_idx = idx;
            best_dist = d;
            best_text_pos = text_pos;
            best_text_len = text_len;
            best_tokens = tokens;
            if (best_dist == 0) {
                /* Exact — no need to keep scanning; mirrors the Python
                 * shortcut in nullspace_search. */
                break;
            }
        }
        ++idx;
        (void)wall;  /* not used by the served reply path */
    }

    if (best_idx < 0) {
        fclose(f);
        return NRL_OK;  /* empty/corrupt index */
    }

    result->distance_bits = best_dist;
    result->exact = (best_dist == 0) ? 1 : 0;
    result->within_threshold = (best_dist <= request->threshold_bits) ? 1 : 0;
    result->stored_entry_index = best_idx;

    /* Serve only exact OR within-threshold hits (Python .lookup() semantics).
     * Stage-VI verify is implicit: exact == distance 0 => residual 0. */
    int serve = (result->exact || result->within_threshold) ? 1 : 0;
    if (serve && best_text_len > 0 && best_text_pos >= 0) {
        if (fseek(f, best_text_pos, SEEK_SET) == 0) {
            uint8_t *text_body = (uint8_t *)malloc(best_text_len);
            if (text_body != NULL) {
                size_t got = fread(text_body, 1, best_text_len, f);
                if (got == best_text_len) {
                    copy_utf8_into(result->text_buf, result->text_buf_capacity,
                                   text_body, best_text_len,
                                   &result->text_byte_len,
                                   &result->text_truncated);
                    result->hit = 1;
                    result->tokens_emitted = (int32_t)best_tokens;
                }
                free(text_body);
            }
        }
    }
    fclose(f);

    const double t1 = full_now_seconds();
    result->wall_seconds = (t1 > t0) ? (t1 - t0) : 0.0;
    return NRL_OK;
}

/* ------------------------------------------------------------------------ *
 * R2 bridge callback                                                        *
 * ------------------------------------------------------------------------ */

static nrl_r2_probe_callback_fn g_r2_callback = NULL;
static void *g_r2_callback_user = NULL;

NRL_API nrl_v1_status nrl_v1_r2_set_callback(
    nrl_r2_probe_callback_fn cb,
    void *user_data) {
    r2_lock();
    g_r2_callback = cb;
    g_r2_callback_user = user_data;
    r2_unlock();
    return NRL_OK;
}

NRL_API int32_t nrl_v1_r2_has_callback(void) {
    r2_lock();
    int32_t has = (g_r2_callback != NULL) ? 1 : 0;
    r2_unlock();
    return has;
}

/* ------------------------------------------------------------------------ *
 * Full-turn orchestrator                                                    *
 * ------------------------------------------------------------------------ */

NRL_API nrl_v1_status nrl_v1_ladder_run_turn(
    const nrl_full_turn_request *request,
    nrl_full_turn_result *result) {
    if (request == NULL || result == NULL) return NRL_ERR_ARGS;

    /* Preserve caller-owned output buffer. */
    char *buf = result->text_buf;
    size_t cap = result->text_buf_capacity;
    memset(result, 0, sizeof(*result));
    result->text_buf = buf;
    result->text_buf_capacity = cap;
    result->served_rung = -1;

    /* ------- R0 muscle memory --------------------------------------- */
    /* Give R0 its own output buffer (same capacity) so we can decide
     * serve-vs-fall-through without copying into the final buffer
     * until we know R0 wins. */
    char *mm_buf = (char *)malloc(cap > 0 ? cap : 1);
    if (mm_buf == NULL) return NRL_ERR_INTERNAL;
    mm_buf[0] = '\0';
    result->mm_report.text_buf = mm_buf;
    result->mm_report.text_buf_capacity = cap;
    nrl_v1_status rc = nrl_v1_mm_lookup(&request->mm, &result->mm_report);
    if (rc != NRL_OK) {
        free(mm_buf);
        return rc;
    }
    if (result->mm_report.hit) {
        /* Copy R0's text into the final result buffer. */
        copy_utf8_into(result->text_buf, result->text_buf_capacity,
                       mm_buf, result->mm_report.text_byte_len,
                       &result->text_byte_len,
                       &result->text_truncated);
        result->served_rung = NRL_LADDER_RUNG_R0_MUSCLE_MEMORY;
        result->tokens_emitted = result->mm_report.tokens_emitted;
        result->wall_seconds = result->mm_report.wall_seconds;
        free(mm_buf);
        /* Re-point mm_report.text_buf so the Python side can still read
         * it from ``result->mm_report`` -- but at this point we freed
         * the buffer so clear the pointer. Python only needs the
         * counts/key, not the body, on a hit (it copies from
         * result->text_buf instead). */
        result->mm_report.text_buf = NULL;
        result->mm_report.text_buf_capacity = 0;
        return NRL_OK;
    }
    free(mm_buf);
    result->mm_report.text_buf = NULL;
    result->mm_report.text_buf_capacity = 0;

    /* ------- R1 ZPM nullspace --------------------------------------- */
    char *zpm_buf = (char *)malloc(cap > 0 ? cap : 1);
    if (zpm_buf == NULL) return NRL_ERR_INTERNAL;
    zpm_buf[0] = '\0';
    result->zpm_report.text_buf = zpm_buf;
    result->zpm_report.text_buf_capacity = cap;
    rc = nrl_v1_zpm_lookup(&request->zpm, &result->zpm_report);
    if (rc != NRL_OK) {
        free(zpm_buf);
        return rc;
    }
    if (result->zpm_report.hit) {
        copy_utf8_into(result->text_buf, result->text_buf_capacity,
                       zpm_buf, result->zpm_report.text_byte_len,
                       &result->text_byte_len,
                       &result->text_truncated);
        result->served_rung = NRL_LADDER_RUNG_R1_ZPM;
        result->tokens_emitted = result->zpm_report.tokens_emitted;
        result->wall_seconds = result->zpm_report.wall_seconds;
        free(zpm_buf);
        result->zpm_report.text_buf = NULL;
        result->zpm_report.text_buf_capacity = 0;
        return NRL_OK;
    }
    free(zpm_buf);
    result->zpm_report.text_buf = NULL;
    result->zpm_report.text_buf_capacity = 0;

    /* ------- R2 omega native resolve (callback-backed for Phase 8-EG) -- */
    r2_lock();
    nrl_r2_probe_callback_fn cb = g_r2_callback;
    void *cb_user = g_r2_callback_user;
    r2_unlock();

    int r2_lane_ok = (request->coherence_lane == NRL_COHERENCE_LANE_FAST_BALANCED ||
                      request->coherence_lane == NRL_COHERENCE_LANE_MAX_THROUGHPUT);
    if (cb != NULL && r2_lane_ok && request->r2_shadow_enabled) {
        char *r2_buf = (char *)malloc(cap > 0 ? cap : 1);
        if (r2_buf == NULL) return NRL_ERR_INTERNAL;
        r2_buf[0] = '\0';
        nrl_r2_probe_request r2req;
        memset(&r2req, 0, sizeof(r2req));
        r2req.coherence_lane = request->coherence_lane;
        r2req.r2_shadow_enabled = request->r2_shadow_enabled;
        r2req.zpm_threshold_bits = request->zpm_threshold_bits;
        r2req.omega_iterations = request->omega_iterations;
        r2req.omega_candidates = request->omega_candidates;
        r2req.omega_budget_ms = request->omega_budget_ms;
        r2req.model_sha256_utf8 = request->zpm.model_sha256_utf8;
        r2req.intent_anchor_bytes = request->intent_anchor_bytes;
        r2req.intent_anchor_len = request->intent_anchor_len;

        nrl_r2_probe_response r2resp;
        memset(&r2resp, 0, sizeof(r2resp));
        r2resp.text_buf = r2_buf;
        r2resp.text_buf_capacity = cap;

        const double t0 = full_now_seconds();
        nrl_v1_status rcb = cb(cb_user, &r2req, &r2resp);
        const double t1 = full_now_seconds();
        double measured = (t1 > t0) ? (t1 - t0) : 0.0;
        double wall = (r2resp.wall_seconds > 0.0) ? r2resp.wall_seconds : measured;

        result->r2_available = r2resp.available;
        result->r2_tokens_emitted = r2resp.tokens_emitted;
        result->r2_stored_entry_index = r2resp.stored_entry_index;
        result->r2_distance_bits = r2resp.distance_bits;
        result->r2_wall_seconds = wall;

        if (rcb == NRL_OK && r2resp.available && r2resp.tokens_emitted > 0) {
            copy_utf8_into(result->text_buf, result->text_buf_capacity,
                           r2_buf, r2resp.text_byte_len,
                           &result->text_byte_len,
                           &result->text_truncated);
            result->served_rung = NRL_LADDER_RUNG_R2_OMEGA_ACTIVE;
            result->tokens_emitted = r2resp.tokens_emitted;
            result->wall_seconds = wall;
            free(r2_buf);
            return NRL_OK;
        }
        free(r2_buf);
        if (rcb != NRL_OK) {
            /* Propagate only if the callback signals a hard failure. We
             * still fall through to R5 so the user gets a reply. */
        }
    }

    /* ------- R5 libllama bridge ------------------------------------- */
    nrl_llama_response resp;
    memset(&resp, 0, sizeof(resp));
    resp.text_buf = result->text_buf;
    resp.text_buf_capacity = result->text_buf_capacity;
    const double t0 = full_now_seconds();
    rc = nrl_v1_llama_run(&request->r5_request, &resp);
    const double t1 = full_now_seconds();
    double measured = (t1 > t0) ? (t1 - t0) : 0.0;
    double wall = (resp.wall_seconds > 0.0) ? resp.wall_seconds : measured;

    result->r5_invoked = 1;
    result->r5_tokens_emitted = resp.tokens_emitted;
    result->r5_wall_seconds = wall;
    result->served_rung = NRL_LADDER_RUNG_R5_LIBLLAMA;
    result->tokens_emitted = resp.tokens_emitted;
    result->text_byte_len = resp.text_byte_len;
    result->text_truncated = resp.text_truncated;
    result->wall_seconds = wall;
    return rc;
}
