// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * test_runtime.c - sanity tests for runtime skeleton.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <stdio.h>
#include <string.h>

#include "nrl/nrl.h"

typedef struct nrl_omega_stats {
    uint64_t executed_updates;
    uint64_t baseline_updates;
    uint64_t active_sublattices;
    uint64_t total_sublattices;
    uint64_t pruned_sublattices;
} nrl_omega_stats;

extern nrl_v1_status nrl_braincore_int4_scalar(uint8_t *packed_potentials,
                                                const uint8_t *packed_inputs,
                                                size_t neuron_count,
                                                size_t iterations,
                                                uint8_t threshold);
extern nrl_v1_status nrl_braincore_int4_avx2(uint8_t *packed_potentials,
                                              const uint8_t *packed_inputs,
                                              size_t neuron_count,
                                              size_t iterations,
                                              uint8_t threshold);
extern nrl_v1_status nrl_braincore_int4_zpm_static(uint8_t *packed_potentials,
                                                    const uint8_t *packed_inputs,
                                                    size_t neuron_count,
                                                    size_t iterations,
                                                    uint8_t threshold,
                                                    uint64_t *executed_updates_out,
                                                    uint64_t *baseline_updates_out);
extern nrl_v1_status nrl_braincore_int4_omega_virtual(
    uint8_t *packed_potentials,
    const uint8_t *packed_inputs,
    size_t neuron_count,
    size_t iterations,
    uint8_t threshold,
    size_t sublattice_neurons,
    uint32_t wake_mod,
    uint64_t prune_zero_byte_threshold,
    size_t min_active_sublattices,
    uint32_t active_kernel_mode,
    nrl_omega_stats *stats_out);

/* One logical timestep: read-only src potentials -> write-only dst (double-buffer step). */
static void braincore_int4_step_doublebuf(const uint8_t *src,
                                         uint8_t *dst,
                                         const uint8_t *packed_inputs,
                                         size_t byte_count,
                                         uint8_t threshold) {
    for (size_t i = 0; i < byte_count; ++i) {
        const uint8_t p = src[i];
        const uint8_t in = packed_inputs[i];
        uint8_t lo = (uint8_t)((p & 0x0fu) + (in & 0x0fu));
        uint8_t hi = (uint8_t)(((p >> 4) & 0x0fu) + ((in >> 4) & 0x0fu));
        if (lo > 15u) {
            lo = 15u;
        }
        if (hi > 15u) {
            hi = 15u;
        }
        if (lo >= threshold) {
            lo = 0u;
        }
        if (hi >= threshold) {
            hi = 0u;
        }
        dst[i] = (uint8_t)(lo | (uint8_t)(hi << 4));
    }
}

/* Same as braincore_int4_step_doublebuf but reverse index order (must match for independence). */
static void braincore_int4_step_doublebuf_reverse(const uint8_t *src,
                                                  uint8_t *dst,
                                                  const uint8_t *packed_inputs,
                                                  size_t byte_count,
                                                  uint8_t threshold) {
    if (byte_count == 0) {
        return;
    }
    for (size_t k = 0; k < byte_count; ++k) {
        const size_t i = byte_count - 1u - k;
        const uint8_t p = src[i];
        const uint8_t in = packed_inputs[i];
        uint8_t lo = (uint8_t)((p & 0x0fu) + (in & 0x0fu));
        uint8_t hi = (uint8_t)(((p >> 4) & 0x0fu) + ((in >> 4) & 0x0fu));
        if (lo > 15u) {
            lo = 15u;
        }
        if (hi > 15u) {
            hi = 15u;
        }
        if (lo >= threshold) {
            lo = 0u;
        }
        if (hi >= threshold) {
            hi = 0u;
        }
        dst[i] = (uint8_t)(lo | (uint8_t)(hi << 4));
    }
}

static void braincore_int4_reference(uint8_t *packed_potentials,
                                     const uint8_t *packed_inputs,
                                     size_t neuron_count,
                                     size_t iterations,
                                     uint8_t threshold) {
    const size_t byte_count = neuron_count >> 1;
    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < byte_count; ++i) {
            const uint8_t p = packed_potentials[i];
            const uint8_t in = packed_inputs[i];
            uint8_t lo = (uint8_t)((p & 0x0fu) + (in & 0x0fu));
            uint8_t hi = (uint8_t)(((p >> 4) & 0x0fu) + ((in >> 4) & 0x0fu));
            if (lo > 15u) lo = 15u;
            if (hi > 15u) hi = 15u;
            if (lo >= threshold) lo = 0u;
            if (hi >= threshold) hi = 0u;
            packed_potentials[i] = (uint8_t)(lo | (uint8_t)(hi << 4));
        }
    }
}

static int test_init_and_version(void) {
    if (nrl_v1_init() != NRL_OK) {
        return 1;
    }
    const char *v = nrl_v1_version();
    if (v == NULL || strstr(v, "nrl") == NULL) {
        return 2;
    }
    return 0;
}

static int test_variant_reporting(void) {
    const char *name = nrl_v1_active_variant("braincore_int4");
    if (name == NULL) {
        return 1;
    }
    if (strcmp(name, "scalar_ref") != 0 &&
        strcmp(name, "avx2") != 0) {
        return 2;
    }
    return 0;
}

static int test_braincore_int4_matches_reference(void) {
    enum { NEURONS = 32, BYTES = NEURONS / 2 };
    uint8_t in[BYTES];
    uint8_t a[BYTES];
    uint8_t b[BYTES];

    for (size_t i = 0; i < BYTES; ++i) {
        in[i] = (uint8_t)(0x11u + (uint8_t)i);
        a[i] = (uint8_t)(0x08u + (uint8_t)i);
        b[i] = a[i];
    }

    braincore_int4_reference(a, in, NEURONS, 7, 12);
    if (nrl_v1_braincore_int4(b, in, NEURONS, 7, 12) != NRL_OK) {
        return 1;
    }
    if (memcmp(a, b, BYTES) != 0) {
        return 2;
    }
    return 0;
}

static int test_braincore_double_buffer_matches_inplace(void) {
    enum { NEURONS = 96, BYTES = NEURONS / 2, ITERS = 11 };
    uint8_t in[BYTES];
    uint8_t gold[BYTES];
    uint8_t buf0[BYTES];
    uint8_t buf1[BYTES];
    uint8_t *cur = buf0;
    uint8_t *nxt = buf1;

    for (size_t i = 0; i < BYTES; ++i) {
        in[i] = (uint8_t)((i * 3u) & 0x77u);
        gold[i] = (uint8_t)(0x0fu ^ (uint8_t)i);
        buf0[i] = gold[i];
    }
    memcpy(buf1, buf0, BYTES);

    braincore_int4_reference(gold, in, NEURONS, ITERS, 9);

    cur = buf0;
    nxt = buf1;
    for (size_t it = 0; it < ITERS; ++it) {
        braincore_int4_step_doublebuf((const uint8_t *)cur, nxt, in, BYTES, 9);
        {
            uint8_t *tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    }
    /* After ITERS ping-pong steps, latest potentials sit in ``cur``. */
    return memcmp(gold, cur, BYTES) == 0 ? 0 : 1;
}

static int test_braincore_intra_step_scan_order_independent(void) {
    enum { BYTES = 73 };
    uint8_t src[BYTES];
    uint8_t d1[BYTES];
    uint8_t d2[BYTES];
    uint8_t in[BYTES];
    for (size_t i = 0; i < BYTES; ++i) {
        src[i] = (uint8_t)(i * 11u + 3u);
        in[i] = (uint8_t)((i * 5u) & 0x7fu);
    }
    braincore_int4_step_doublebuf(src, d1, in, BYTES, 7);
    braincore_int4_step_doublebuf_reverse(src, d2, in, BYTES, 7);
    return memcmp(d1, d2, BYTES) == 0 ? 0 : 1;
}

static int test_braincore_int4_deterministic(void) {
    enum { NEURONS = 64, BYTES = NEURONS / 2 };
    uint8_t in[BYTES];
    uint8_t a[BYTES];
    uint8_t b[BYTES];

    for (size_t i = 0; i < BYTES; ++i) {
        in[i] = (uint8_t)(i & 0x7fu);
        a[i] = (uint8_t)(i ^ 0x55u);
        b[i] = a[i];
    }

    if (nrl_v1_braincore_int4(a, in, NEURONS, 19, 10) != NRL_OK) {
        return 1;
    }
    if (nrl_v1_braincore_int4(b, in, NEURONS, 19, 10) != NRL_OK) {
        return 2;
    }
    return memcmp(a, b, BYTES) == 0 ? 0 : 3;
}

static int test_braincore_packed_bytes(void) {
    if (nrl_v1_braincore_packed_bytes(0) != 0u) {
        return 1;
    }
    if (nrl_v1_braincore_packed_bytes(15) != 0u) {
        return 2;
    }
    if (nrl_v1_braincore_packed_bytes(16) != 8u) {
        return 3;
    }
    return 0;
}

static int test_braincore_int4_argchecks(void) {
    uint8_t buf[8] = {0};
    if (nrl_v1_braincore_int4(NULL, buf, 16, 1, 8) != NRL_ERR_ARGS) return 1;
    if (nrl_v1_braincore_int4(buf, NULL, 16, 1, 8) != NRL_ERR_ARGS) return 2;
    if (nrl_v1_braincore_int4(buf, buf, 0, 1, 8) != NRL_ERR_ARGS) return 3;
    if (nrl_v1_braincore_int4(buf, buf, 15, 1, 8) != NRL_ERR_ARGS) return 4;
    if (nrl_v1_braincore_int4(buf, buf, 16, 0, 8) != NRL_ERR_ARGS) return 5;
    if (nrl_v1_braincore_int4(buf, buf, 16, 1, 0) != NRL_ERR_ARGS) return 6;
    if (nrl_v1_braincore_int4(buf, buf, 16, 1, 16) != NRL_ERR_ARGS) return 7;
    return 0;
}

static int test_avx2_parity_lock(void) {
    if ((nrl_v1_cpu_features() & NRL_CPU_AVX2) == 0) {
        return 0;
    }

    enum { NEURONS = 124, BYTES = NEURONS / 2 };
    uint8_t in[BYTES];
    uint8_t scalar_out[BYTES];
    uint8_t avx2_out[BYTES];

    for (size_t i = 0; i < BYTES; ++i) {
        in[i] = (uint8_t)((i * 13u) & 0x77u);
        scalar_out[i] = (uint8_t)(0x33u ^ (uint8_t)i);
        avx2_out[i] = scalar_out[i];
    }

    if (nrl_braincore_int4_scalar(scalar_out, in, NEURONS, 31, 8) != NRL_OK) {
        return 1;
    }
    if (nrl_braincore_int4_avx2(avx2_out, in, NEURONS, 31, 8) != NRL_OK) {
        return 2;
    }
    if (memcmp(scalar_out, avx2_out, BYTES) != 0) {
        return 3;
    }

    for (size_t i = 0; i < BYTES; ++i) {
        scalar_out[i] = (uint8_t)(0x22u + (uint8_t)i);
        avx2_out[i] = scalar_out[i];
    }
    if (nrl_braincore_int4_scalar(scalar_out, in, NEURONS, 17, 12) != NRL_OK) {
        return 4;
    }
    if (nrl_braincore_int4_avx2(avx2_out, in, NEURONS, 17, 12) != NRL_OK) {
        return 5;
    }
    if (memcmp(scalar_out, avx2_out, BYTES) != 0) {
        return 6;
    }

    return 0;
}

static int test_zpm_static_parity_and_stats(void) {
    enum { NEURONS = 190, BYTES = NEURONS / 2 };
    uint8_t in[BYTES];
    uint8_t scalar_out[BYTES];
    uint8_t zpm_out[BYTES];
    uint64_t executed_updates = 0;
    uint64_t baseline_updates = 0;

    for (size_t i = 0; i < BYTES; ++i) {
        in[i] = (uint8_t)((i * 7u) & 0x77u);
        scalar_out[i] = (uint8_t)(0x44u ^ (uint8_t)i);
        zpm_out[i] = scalar_out[i];
    }

    if (nrl_braincore_int4_scalar(scalar_out, in, NEURONS, 257, 9) != NRL_OK) {
        return 1;
    }
    if (nrl_braincore_int4_zpm_static(zpm_out, in, NEURONS, 257, 9,
                                      &executed_updates,
                                      &baseline_updates) != NRL_OK) {
        return 2;
    }
    if (memcmp(scalar_out, zpm_out, BYTES) != 0) {
        return 3;
    }
    if (executed_updates != (uint64_t)NEURONS) {
        return 4;
    }
    if (baseline_updates != (uint64_t)NEURONS * 257ull) {
        return 5;
    }
    return 0;
}

static int test_omega_virtual_stats(void) {
    enum { NEURONS = 1024, BYTES = NEURONS / 2 };
    uint8_t in[BYTES];
    uint8_t out[BYTES];
    nrl_omega_stats stats = {0};
    nrl_omega_stats stats2 = {0};

    memset(in, 0, BYTES);
    for (size_t i = 0; i < BYTES; i += 64) {
        in[i] = 0x77u;
    }
    memset(out, 0, BYTES);
    if (nrl_braincore_int4_omega_virtual(out, in, NEURONS, 2048, 8,
                                         128, 128, 0, 1, 0, &stats) != NRL_OK) {
        return 1;
    }
    if (stats.baseline_updates != (uint64_t)NEURONS * 2048ull) {
        return 2;
    }
    if (stats.executed_updates > stats.baseline_updates) {
        return 3;
    }
    if (stats.active_sublattices > stats.total_sublattices) {
        return 4;
    }

    memset(out, 0, BYTES);
    if (nrl_braincore_int4_omega_virtual(out, in, NEURONS, 2048, 8,
                                         128, 128, 0, 1, 0, &stats2) != NRL_OK) {
        return 5;
    }
    if (memcmp(&stats, &stats2, sizeof(stats)) != 0) {
        return 6;
    }
    return 0;
}

int main(void) {
    int rc = 0;
    rc = test_init_and_version();
    if (rc != 0) {
        fprintf(stderr, "test_init_and_version failed: %d\n", rc);
        return rc;
    }

    rc = test_variant_reporting();
    if (rc != 0) {
        fprintf(stderr, "test_variant_reporting failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_int4_matches_reference();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_int4_matches_reference failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_int4_deterministic();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_int4_deterministic failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_double_buffer_matches_inplace();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_double_buffer_matches_inplace failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_intra_step_scan_order_independent();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_intra_step_scan_order_independent failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_packed_bytes();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_packed_bytes failed: %d\n", rc);
        return rc;
    }

    rc = test_braincore_int4_argchecks();
    if (rc != 0) {
        fprintf(stderr, "test_braincore_int4_argchecks failed: %d\n", rc);
        return rc;
    }

    rc = test_avx2_parity_lock();
    if (rc != 0) {
        fprintf(stderr, "test_avx2_parity_lock failed: %d\n", rc);
        return rc;
    }

    rc = test_zpm_static_parity_and_stats();
    if (rc != 0) {
        fprintf(stderr, "test_zpm_static_parity_and_stats failed: %d\n", rc);
        return rc;
    }

    rc = test_omega_virtual_stats();
    if (rc != 0) {
        fprintf(stderr, "test_omega_virtual_stats failed: %d\n", rc);
        return rc;
    }

    puts("test_runtime: OK");
    return 0;
}
