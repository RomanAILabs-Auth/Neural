/*
 * zpm_int4_static.c - exact static-input accelerator for INT4 braincore.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <string.h>

#include "runtime_internal.h"

/* One-step transition for a 4-bit state under fixed 4-bit input and threshold. */
static uint8_t step_transition(uint8_t state, uint8_t input, uint8_t threshold) {
    uint8_t next = (uint8_t)(state + input);
    if (next > 15u) {
        next = 15u;
    }
    if (next >= threshold) {
        next = 0u;
    }
    return next;
}

/*
 * Build exact k-step transition map for a fixed input value.
 * states are in [0,15], so this is tiny and cache-resident.
 */
static void build_k_map(uint8_t input, uint8_t threshold, size_t iterations,
                        uint8_t out_map[16]) {
    uint8_t power_maps[64][16];
    uint8_t acc[16];
    size_t bits = 0;
    size_t k = iterations;

    for (uint8_t x = 0; x < 16u; ++x) {
        power_maps[0][x] = step_transition(x, input, threshold);
        acc[x] = x;
    }

    while (k != 0) {
        ++bits;
        k >>= 1;
    }
    if (bits == 0) {
        bits = 1;
    }

    for (size_t b = 1; b < bits; ++b) {
        for (uint8_t x = 0; x < 16u; ++x) {
            power_maps[b][x] = power_maps[b - 1][power_maps[b - 1][x]];
        }
    }

    for (size_t b = 0; b < bits; ++b) {
        if (((iterations >> b) & 1u) == 0u) {
            continue;
        }
        for (uint8_t x = 0; x < 16u; ++x) {
            acc[x] = power_maps[b][acc[x]];
        }
    }

    memcpy(out_map, acc, 16);
}

nrl_v1_status nrl_braincore_int4_zpm_static(uint8_t *packed_potentials,
                                            const uint8_t *packed_inputs,
                                            size_t neuron_count,
                                            size_t iterations,
                                            uint8_t threshold,
                                            uint64_t *executed_updates_out,
                                            uint64_t *baseline_updates_out) {
    uint8_t maps[16][16];
    const size_t byte_count = neuron_count >> 1;

    for (uint8_t input = 0; input < 16u; ++input) {
        build_k_map(input, threshold, iterations, maps[input]);
    }

    for (size_t i = 0; i < byte_count; ++i) {
        const uint8_t p = packed_potentials[i];
        const uint8_t in = packed_inputs[i];
        const uint8_t in_lo = (uint8_t)(in & 0x0fu);
        const uint8_t in_hi = (uint8_t)((in >> 4) & 0x0fu);
        const uint8_t st_lo = (uint8_t)(p & 0x0fu);
        const uint8_t st_hi = (uint8_t)((p >> 4) & 0x0fu);
        const uint8_t out_lo = maps[in_lo][st_lo];
        const uint8_t out_hi = maps[in_hi][st_hi];
        packed_potentials[i] = (uint8_t)(out_lo | (uint8_t)(out_hi << 4));
    }

    if (executed_updates_out != NULL) {
        *executed_updates_out = (uint64_t)neuron_count;
    }
    if (baseline_updates_out != NULL) {
        *baseline_updates_out = (uint64_t)neuron_count * (uint64_t)iterations;
    }

    return NRL_OK;
}
