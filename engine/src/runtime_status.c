// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * runtime_status.c - cross-platform process memory sampling.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 */
#include "nrl/runtime_status.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#else
#include <sys/resource.h>
#endif

void nrl_runtime_get_process_memory(nrl_process_memory *out) {
    if (out == NULL) {
        return;
    }
    out->current_bytes = 0;
    out->peak_bytes = 0;

#if defined(_WIN32)
    {
        PROCESS_MEMORY_COUNTERS pmc;
        memset(&pmc, 0, sizeof(pmc));
        pmc.cb = sizeof(pmc);
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            out->current_bytes = (size_t)pmc.WorkingSetSize;
            out->peak_bytes = (size_t)pmc.PeakWorkingSetSize;
        }
    }
#elif defined(__linux__)
    {
        FILE *fp = fopen("/proc/self/status", "r");
        if (fp != NULL) {
            char line[256];
            while (fgets(line, sizeof(line), fp) != NULL) {
                unsigned long kb = 0;
                if (strncmp(line, "VmRSS:", 6) == 0) {
                    if (sscanf(line, "VmRSS: %lu", &kb) == 1) {
                        out->current_bytes = (size_t)kb * 1024u;
                    }
                } else if (strncmp(line, "VmHWM:", 6) == 0) {
                    if (sscanf(line, "VmHWM: %lu", &kb) == 1) {
                        out->peak_bytes = (size_t)kb * 1024u;
                    }
                }
            }
            fclose(fp);
        }
        if (out->peak_bytes == 0) {
            struct rusage ru;
            memset(&ru, 0, sizeof(ru));
            if (getrusage(RUSAGE_SELF, &ru) == 0) {
                out->peak_bytes = (size_t)ru.ru_maxrss * 1024u;
            }
        }
    }
#else
    {
        struct rusage ru;
        memset(&ru, 0, sizeof(ru));
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
            out->current_bytes = (size_t)ru.ru_maxrss;
            out->peak_bytes = (size_t)ru.ru_maxrss;
#else
            out->current_bytes = (size_t)ru.ru_maxrss * 1024u;
            out->peak_bytes = (size_t)ru.ru_maxrss * 1024u;
#endif
        }
    }
#endif
}
