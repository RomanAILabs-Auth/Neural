/*
 * nrl.h - NRL Engine v1 Public C ABI
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#ifndef NRL_V1_H_
#define NRL_V1_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define NRL_API __declspec(dllexport)
#else
#define NRL_API __attribute__((visibility("default")))
#endif

typedef enum nrl_v1_status {
    NRL_OK = 0,
    NRL_ERR_ARGS = 1,
    NRL_ERR_ALLOC = 2,
    NRL_ERR_CPU = 3,
    NRL_ERR_INTERNAL = 99
} nrl_v1_status;

#define NRL_VERSION_MAJOR 0
#define NRL_VERSION_MINOR 1
#define NRL_VERSION_PATCH 0
#define NRL_VERSION_STRING "0.1.0"

/* CPU feature bits */
#define NRL_CPU_SSE2    (1u << 0)
#define NRL_CPU_SSE42   (1u << 1)
#define NRL_CPU_AVX     (1u << 2)
#define NRL_CPU_AVX2    (1u << 3)
#define NRL_CPU_AVX512F (1u << 4)
#define NRL_CPU_BMI2    (1u << 5)
#define NRL_CPU_POPCNT  (1u << 6)
#define NRL_CPU_FMA     (1u << 7)

NRL_API nrl_v1_status nrl_v1_init(void);
NRL_API const char *nrl_v1_version(void);
NRL_API uint32_t nrl_v1_cpu_features(void);
NRL_API const char *nrl_v1_active_variant(const char *kernel_name);

/*
 * Packed INT4 lattice layout: neuron_count must be positive and even.
 * Each byte stores two 4-bit potentials (low nibble = neuron 2i, high = 2i+1).
 * Returns 0 if neuron_count is invalid.
 */
NRL_API size_t nrl_v1_braincore_packed_bytes(size_t neuron_count);

/* INT4 packed lattice update: 2 neurons per byte, threshold-reset semantics. */
NRL_API nrl_v1_status nrl_v1_braincore_int4(
    uint8_t *packed_potentials,
    const uint8_t *packed_inputs,
    size_t neuron_count,
    size_t iterations,
    uint8_t threshold);

#ifdef __cplusplus
}
#endif

#endif /* NRL_V1_H_ */
