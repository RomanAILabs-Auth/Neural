// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * runtime_status.h - process memory and lightweight runtime introspection.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 */
#ifndef NRL_RUNTIME_STATUS_H_
#define NRL_RUNTIME_STATUS_H_

#include <stddef.h>

typedef struct nrl_process_memory {
    size_t current_bytes;
    size_t peak_bytes;
} nrl_process_memory;

/* RSS-style current and peak (best-effort per OS). Safe to call from CLI threads. */
void nrl_runtime_get_process_memory(nrl_process_memory *out);

#endif /* NRL_RUNTIME_STATUS_H_ */
