// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * version.c - build/version banner for NRL.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include "nrl/nrl.h"

#define NRL_STR2(x) #x
#define NRL_STR(x) NRL_STR2(x)

#if defined(_MSC_VER)
#define NRL_COMPILER "msvc"
#elif defined(__clang__)
#define NRL_COMPILER "clang"
#elif defined(__GNUC__)
#define NRL_COMPILER "gcc"
#else
#define NRL_COMPILER "cc"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define NRL_ARCH "x86_64"
#elif defined(_M_ARM64) || defined(__aarch64__)
#define NRL_ARCH "aarch64"
#else
#define NRL_ARCH "unknown"
#endif

static const char *k_version =
    "nrl " NRL_VERSION_STRING " (" NRL_COMPILER ", " NRL_ARCH ")";

const char *nrl_v1_version(void) { return k_version; }
