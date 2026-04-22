/*
 * runtime_internal.h - internal runtime contracts.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#ifndef NRL_RUNTIME_INTERNAL_H_
#define NRL_RUNTIME_INTERNAL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "nrl/nrl.h"

typedef nrl_v1_status (*nrl_braincore_int4_fn)(uint8_t *packed_potentials,
                                                const uint8_t *packed_inputs,
                                                size_t neuron_count,
                                                size_t iterations,
                                                uint8_t threshold);

typedef struct nrl_dispatch_table {
    const char *braincore_int4_variant;
    nrl_braincore_int4_fn braincore_int4;
} nrl_dispatch_table;

typedef struct nrl_omega_stats {
    uint64_t executed_updates;
    uint64_t baseline_updates;
    uint64_t active_sublattices;
    uint64_t total_sublattices;
    uint64_t pruned_sublattices;
} nrl_omega_stats;

extern nrl_dispatch_table g_nrl_dispatch;
extern bool g_nrl_initialized;

void nrl_runtime_detect_features(void);
void nrl_runtime_bind_dispatch(void);
nrl_v1_status nrl_braincore_int4_stub(uint8_t *packed_potentials,
                                      const uint8_t *packed_inputs,
                                      size_t neuron_count,
                                      size_t iterations,
                                      uint8_t threshold);
nrl_v1_status nrl_braincore_int4_scalar(uint8_t *packed_potentials,
                                        const uint8_t *packed_inputs,
                                        size_t neuron_count,
                                        size_t iterations,
                                        uint8_t threshold);
nrl_v1_status nrl_braincore_int4_avx2(uint8_t *packed_potentials,
                                      const uint8_t *packed_inputs,
                                      size_t neuron_count,
                                      size_t iterations,
                                      uint8_t threshold);
nrl_v1_status nrl_braincore_int4_zpm_static(uint8_t *packed_potentials,
                                            const uint8_t *packed_inputs,
                                            size_t neuron_count,
                                            size_t iterations,
                                            uint8_t threshold,
                                            uint64_t *executed_updates_out,
                                            uint64_t *baseline_updates_out);
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
    nrl_omega_stats *stats_out);

#endif /* NRL_RUNTIME_INTERNAL_H_ */
