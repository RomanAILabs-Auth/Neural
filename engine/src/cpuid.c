// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * cpuid.c - CPU feature detection for NRL runtime.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "nrl/nrl.h"

typedef struct cpuid_regs {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
} cpuid_regs;

static uint32_t g_features = 0;

#if defined(__x86_64__) || defined(_M_X64)
static void cpuid_query(uint32_t leaf, uint32_t subleaf, cpuid_regs *out) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    out->eax = (uint32_t)regs[0];
    out->ebx = (uint32_t)regs[1];
    out->ecx = (uint32_t)regs[2];
    out->edx = (uint32_t)regs[3];
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("cpuid"
                     : "=a"(out->eax), "=b"(out->ebx), "=c"(out->ecx),
                       "=d"(out->edx)
                     : "a"(leaf), "c"(subleaf));
#else
    memset(out, 0, sizeof(*out));
#endif
}

static uint64_t xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax;
    uint32_t edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
                     : "=a"(eax), "=d"(edx)
                     : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}

static bool os_supports_avx_state(void) {
    uint64_t xcr0 = xgetbv0();
    return (xcr0 & 0x6u) == 0x6u;
}

static bool os_supports_avx512_state(void) {
    uint64_t xcr0 = xgetbv0();
    return (xcr0 & 0xe6u) == 0xe6u;
}
#endif

void nrl_runtime_detect_features(void) {
    uint32_t features = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
    g_features = 0;
    (void)features;
    return;
#elif defined(__x86_64__) || defined(_M_X64)
    cpuid_regs r0 = {0};
    cpuid_regs r1 = {0};
    cpuid_regs r7 = {0};

    cpuid_query(0, 0, &r0);
    if (r0.eax < 1) {
        g_features = 0;
        return;
    }

    cpuid_query(1, 0, &r1);
    if ((r1.edx & (1u << 26)) != 0) {
        features |= NRL_CPU_SSE2;
    }
    if ((r1.ecx & (1u << 20)) != 0) {
        features |= NRL_CPU_SSE42;
    }
    if ((r1.ecx & (1u << 23)) != 0) {
        features |= NRL_CPU_POPCNT;
    }
    if ((r1.ecx & (1u << 12)) != 0) {
        features |= NRL_CPU_FMA;
    }

    const bool avx_hw = (r1.ecx & (1u << 28)) != 0;
    const bool osxsave = (r1.ecx & (1u << 27)) != 0;
    const bool avx_usable = avx_hw && osxsave && os_supports_avx_state();
    if (avx_usable) {
        features |= NRL_CPU_AVX;
    }

    if (r0.eax >= 7) {
        cpuid_query(7, 0, &r7);
        if ((r7.ebx & (1u << 8)) != 0) {
            features |= NRL_CPU_BMI2;
        }
        if (avx_usable && (r7.ebx & (1u << 5)) != 0) {
            features |= NRL_CPU_AVX2;
        }
        if (avx_usable && os_supports_avx512_state() &&
            (r7.ebx & (1u << 16)) != 0) {
            features |= NRL_CPU_AVX512F;
        }
    }
#else
    (void)features;
#endif

    g_features = features;
}

uint32_t nrl_v1_cpu_features(void) { return g_features; }
