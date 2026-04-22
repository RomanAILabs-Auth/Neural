/*
 * braincore_int4_scalar.c - scalar reference kernel for packed INT4 braincore.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include "runtime_internal.h"

nrl_v1_status nrl_braincore_int4_scalar(uint8_t *packed_potentials,
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

            packed_potentials[i] = (uint8_t)(lo | (uint8_t)(hi << 4));
        }
    }

    return NRL_OK;
}
