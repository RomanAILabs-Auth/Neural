/*
 * braincore_int4_avx2.c - AVX2 kernel for packed INT4 braincore.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <immintrin.h>

#include "runtime_internal.h"

nrl_v1_status nrl_braincore_int4_avx2(uint8_t *packed_potentials,
                                      const uint8_t *packed_inputs,
                                      size_t neuron_count,
                                      size_t iterations,
                                      uint8_t threshold) {
    const size_t byte_count = neuron_count >> 1;
    const size_t vec_bytes = byte_count & ~(size_t)31u;
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i v_thresh = _mm256_set1_epi8((char)(threshold & 0x0f));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i v_hi_bit = _mm256_set1_epi8(0x08);
    const int threshold_is_8 = ((threshold & 0x0f) == 8);

    for (size_t it = 0; it < iterations; ++it) {
        size_t i = 0;
        for (; i < vec_bytes; i += 32) {
            const __m256i v_pot =
                _mm256_loadu_si256((const __m256i *)&packed_potentials[i]);
            const __m256i v_in =
                _mm256_loadu_si256((const __m256i *)&packed_inputs[i]);

            __m256i lo_p = _mm256_and_si256(v_pot, nibble_mask);
            __m256i hi_p =
                _mm256_and_si256(_mm256_srli_epi16(v_pot, 4), nibble_mask);
            const __m256i lo_i = _mm256_and_si256(v_in, nibble_mask);
            const __m256i hi_i =
                _mm256_and_si256(_mm256_srli_epi16(v_in, 4), nibble_mask);

            lo_p = _mm256_add_epi8(lo_p, lo_i);
            hi_p = _mm256_add_epi8(hi_p, hi_i);
            lo_p = _mm256_min_epu8(lo_p, nibble_mask);
            hi_p = _mm256_min_epu8(hi_p, nibble_mask);

            if (threshold_is_8) {
                const __m256i keep_lo =
                    _mm256_cmpeq_epi8(_mm256_and_si256(lo_p, v_hi_bit), zero);
                const __m256i keep_hi =
                    _mm256_cmpeq_epi8(_mm256_and_si256(hi_p, v_hi_bit), zero);
                lo_p = _mm256_and_si256(lo_p, keep_lo);
                hi_p = _mm256_and_si256(hi_p, keep_hi);
            } else {
                const __m256i m_lo =
                    _mm256_cmpeq_epi8(_mm256_max_epu8(lo_p, v_thresh), lo_p);
                const __m256i m_hi =
                    _mm256_cmpeq_epi8(_mm256_max_epu8(hi_p, v_thresh), hi_p);
                lo_p = _mm256_andnot_si256(m_lo, lo_p);
                hi_p = _mm256_andnot_si256(m_hi, hi_p);
            }

            const __m256i packed =
                _mm256_or_si256(_mm256_and_si256(lo_p, nibble_mask),
                                _mm256_slli_epi16(
                                    _mm256_and_si256(hi_p, nibble_mask), 4));
            _mm256_storeu_si256((__m256i *)&packed_potentials[i], packed);
        }

        for (; i < byte_count; ++i) {
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
