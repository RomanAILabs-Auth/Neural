/*
 * capi.c - public C ABI for NRL runtime.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <string.h>

#include "runtime_internal.h"

bool g_nrl_initialized = false;

nrl_v1_status nrl_v1_init(void) {
    if (g_nrl_initialized) {
        return NRL_OK;
    }

    nrl_runtime_detect_features();
    nrl_runtime_bind_dispatch();
    g_nrl_initialized = true;
    return NRL_OK;
}

const char *nrl_v1_active_variant(const char *kernel_name) {
    if (kernel_name == NULL) {
        return "unknown";
    }
    if (!g_nrl_initialized) {
        (void)nrl_v1_init();
    }
    if (strcmp(kernel_name, "braincore_int4") == 0) {
        return g_nrl_dispatch.braincore_int4_variant;
    }
    return "unknown";
}

size_t nrl_v1_braincore_packed_bytes(size_t neuron_count) {
    if (neuron_count == 0u || (neuron_count & 1u) != 0u) {
        return 0u;
    }
    return neuron_count / 2u;
}

nrl_v1_status nrl_braincore_int4_stub(uint8_t *packed_potentials,
                                      const uint8_t *packed_inputs,
                                      size_t neuron_count,
                                      size_t iterations,
                                      uint8_t threshold) {
    if (packed_potentials == NULL || packed_inputs == NULL || neuron_count == 0 ||
        iterations == 0 || threshold == 0) {
        return NRL_ERR_ARGS;
    }
    return NRL_ERR_INTERNAL;
}

nrl_v1_status nrl_v1_braincore_int4(uint8_t *packed_potentials,
                                    const uint8_t *packed_inputs,
                                    size_t neuron_count,
                                    size_t iterations,
                                    uint8_t threshold) {
    if (packed_potentials == NULL || packed_inputs == NULL) {
        return NRL_ERR_ARGS;
    }
    if (neuron_count == 0 || (neuron_count & 1u) != 0u) {
        return NRL_ERR_ARGS;
    }
    if (iterations == 0) {
        return NRL_ERR_ARGS;
    }
    if (threshold == 0 || threshold > 15u) {
        return NRL_ERR_ARGS;
    }

    if (!g_nrl_initialized) {
        const nrl_v1_status st = nrl_v1_init();
        if (st != NRL_OK) {
            return st;
        }
    }
    return g_nrl_dispatch.braincore_int4(packed_potentials, packed_inputs,
                                         neuron_count, iterations, threshold);
}
