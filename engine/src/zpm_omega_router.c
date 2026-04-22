/*
 * zpm_omega_router.c - fractal routed virtual compute lane.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <string.h>
#include <stdlib.h>

#include "runtime_internal.h"

static uint64_t signature_block(const uint8_t *p, size_t n) {
    uint64_t x = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        x ^= (uint64_t)p[i];
        x *= 1099511628211ull;
    }
    return x;
}

static uint64_t count_nonzero_bytes(const uint8_t *p, size_t n) {
    uint64_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c += p[i] != 0 ? 1u : 0u;
    }
    return c;
}

nrl_v1_status nrl_braincore_int4_omega_virtual(
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
    nrl_omega_stats *stats_out) {
    nrl_omega_stats stats = {0};
    const size_t safe_sublattice_neurons =
        sublattice_neurons < 64 ? 64 : (sublattice_neurons & ~(size_t)1u);
    const size_t sub_bytes = safe_sublattice_neurons >> 1;
    const size_t total_bytes = neuron_count >> 1;
    const uint32_t safe_wake_mod = wake_mod == 0 ? 1u : wake_mod;
    const size_t total_sublattices =
        (total_bytes + sub_bytes - 1u) / sub_bytes;
    uint8_t *wake_flags = NULL;
    uint8_t *prune_flags = NULL;
    size_t active_count = 0;
    const uint32_t kernel_mode = active_kernel_mode > 1u ? 0u : active_kernel_mode;
    const uint32_t has_avx2 = (nrl_v1_cpu_features() & NRL_CPU_AVX2) != 0u;

    if (sub_bytes == 0 || total_bytes == 0) {
        return NRL_ERR_ARGS;
    }
    if (total_sublattices == 0) {
        return NRL_ERR_ARGS;
    }

    wake_flags = (uint8_t *)calloc(total_sublattices, 1u);
    prune_flags = (uint8_t *)calloc(total_sublattices, 1u);
    if (wake_flags == NULL || prune_flags == NULL) {
        free(wake_flags);
        free(prune_flags);
        return NRL_ERR_ALLOC;
    }

    for (size_t s = 0; s < total_sublattices; ++s) {
        const size_t off = s * sub_bytes;
        const size_t block_bytes =
            (off + sub_bytes <= total_bytes) ? sub_bytes : (total_bytes - off);
        const uint8_t *in_block = packed_inputs + off;
        const uint64_t nonzero = count_nonzero_bytes(in_block, block_bytes);

        stats.total_sublattices += 1;
        stats.baseline_updates += (uint64_t)(block_bytes << 1) * (uint64_t)iterations;

        if (nonzero <= prune_zero_byte_threshold) {
            stats.pruned_sublattices += 1;
            prune_flags[s] = 1u;
            continue;
        }

        const uint64_t sig = signature_block(in_block, block_bytes);
        const uint64_t block_index = s;
        const int gate_by_index = (block_index % (uint64_t)safe_wake_mod) == 0u;
        const int gate_by_signature = (uint32_t)(sig % (uint64_t)safe_wake_mod) == 0u;
        if (gate_by_index || gate_by_signature) {
            wake_flags[s] = 1u;
            active_count += 1u;
        }
    }

    if (min_active_sublattices > total_sublattices) {
        min_active_sublattices = total_sublattices;
    }
    if (active_count < min_active_sublattices) {
        for (size_t s = 0; s < total_sublattices && active_count < min_active_sublattices; ++s) {
            if (prune_flags[s] != 0u || wake_flags[s] != 0u) {
                continue;
            }
            wake_flags[s] = 1u;
            active_count += 1u;
        }
    }

    for (size_t s = 0; s < total_sublattices; ++s) {
        const size_t off = s * sub_bytes;
        const size_t block_bytes =
            (off + sub_bytes <= total_bytes) ? sub_bytes : (total_bytes - off);
        const size_t block_neurons = block_bytes << 1;
        const uint8_t *in_block = packed_inputs + off;
        uint8_t *pot_block = packed_potentials + off;
        uint64_t executed = 0;
        uint64_t baseline = 0;
        nrl_v1_status rc = NRL_OK;

        if (wake_flags[s] == 0u) {
            continue;
        }

        if (kernel_mode == 0u) {
            rc = nrl_braincore_int4_zpm_static(
                pot_block, in_block, block_neurons, iterations, threshold, &executed,
                &baseline);
        } else if (has_avx2 != 0u) {
            rc = nrl_braincore_int4_avx2(
                pot_block, in_block, block_neurons, iterations, threshold);
            executed = (uint64_t)block_neurons * (uint64_t)iterations;
            baseline = executed;
        } else {
            rc = nrl_braincore_int4_scalar(
                pot_block, in_block, block_neurons, iterations, threshold);
            executed = (uint64_t)block_neurons * (uint64_t)iterations;
            baseline = executed;
        }

        if (rc != NRL_OK) {
            free(wake_flags);
            free(prune_flags);
            return rc;
        }

        stats.active_sublattices += 1;
        stats.executed_updates += executed;
        (void)baseline;
    }

    free(wake_flags);
    free(prune_flags);
    if (stats_out != NULL) {
        *stats_out = stats;
    }
    return NRL_OK;
}
