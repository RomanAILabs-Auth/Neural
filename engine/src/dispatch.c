// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * dispatch.c - kernel variant binding for NRL runtime.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include "runtime_internal.h"

nrl_dispatch_table g_nrl_dispatch = {
    "unbound",
    nrl_braincore_int4_scalar,
};

void nrl_runtime_bind_dispatch(void) {
    const uint32_t features = nrl_v1_cpu_features();

    if ((features & NRL_CPU_AVX2) != 0) {
        g_nrl_dispatch.braincore_int4_variant = "avx2";
        g_nrl_dispatch.braincore_int4 = nrl_braincore_int4_avx2;
        return;
    }

    g_nrl_dispatch.braincore_int4_variant = "scalar_ref";
    g_nrl_dispatch.braincore_int4 = nrl_braincore_int4_scalar;
}
