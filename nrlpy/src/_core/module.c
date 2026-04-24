// Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
/*
 * nrlpy/_core/module.c - CPython extension binding to libnrl.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nrl/nrl.h"
#include "nrl/llama_bridge.h"
#include "nrl/ladder_native.h"
#include "nrl/ladder_full.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

static PyObject *NrlPyError = NULL;

static PyObject *py_init(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    nrl_v1_status rc = nrl_v1_init();
    if (rc != NRL_OK) {
        PyErr_SetString(NrlPyError, "nrl_v1_init failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString(nrl_v1_version());
}

static PyObject *py_features(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    const uint32_t f = nrl_v1_cpu_features();
    PyObject *d = PyDict_New();
    if (d == NULL) {
        return NULL;
    }
#define PUT(name, flag)                                                         \
    do {                                                                         \
        PyObject *v = (f & (flag)) ? Py_True : Py_False;                        \
        Py_INCREF(v);                                                            \
        if (PyDict_SetItemString(d, (name), v) < 0) {                           \
            Py_DECREF(v);                                                        \
            Py_DECREF(d);                                                        \
            return NULL;                                                         \
        }                                                                        \
        Py_DECREF(v);                                                            \
    } while (0)
    PUT("sse2", NRL_CPU_SSE2);
    PUT("sse42", NRL_CPU_SSE42);
    PUT("avx", NRL_CPU_AVX);
    PUT("avx2", NRL_CPU_AVX2);
    PUT("avx512f", NRL_CPU_AVX512F);
    PUT("bmi2", NRL_CPU_BMI2);
    PUT("popcnt", NRL_CPU_POPCNT);
    PUT("fma", NRL_CPU_FMA);
#undef PUT
    return d;
}

static PyObject *py_active_variant(PyObject *self, PyObject *args) {
    (void)self;
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    return PyUnicode_FromString(nrl_v1_active_variant(name));
}

static PyObject *py_braincore_packed_bytes(PyObject *self, PyObject *args) {
    (void)self;
    Py_ssize_t neurons = 0;
    if (!PyArg_ParseTuple(args, "n:braincore_packed_bytes", &neurons)) {
        return NULL;
    }
    if (neurons < 0) {
        PyErr_SetString(PyExc_ValueError, "neurons must be non-negative");
        return NULL;
    }
    const size_t sz = nrl_v1_braincore_packed_bytes((size_t)neurons);
    return PyLong_FromUnsignedLongLong((unsigned long long)sz);
}

static PyObject *py_braincore_int4_inplace(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    Py_buffer pot = {0};
    Py_buffer inp = {0};
    Py_ssize_t neurons = 0;
    Py_ssize_t iter = 0;
    unsigned char thresh = 12;
    static char *kwlist[] = {"potentials", "inputs", "neurons", "iterations", "threshold", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "w*y*nnB:braincore_int4_inplace", kwlist, &pot, &inp,
                                     &neurons, &iter, &thresh)) {
        return NULL;
    }

    if (neurons <= 0 || (neurons & 1) != 0) {
        PyBuffer_Release(&pot);
        PyBuffer_Release(&inp);
        PyErr_SetString(PyExc_ValueError, "neurons must be a positive even integer");
        return NULL;
    }
    if (iter <= 0) {
        PyBuffer_Release(&pot);
        PyBuffer_Release(&inp);
        PyErr_SetString(PyExc_ValueError, "iterations must be positive");
        return NULL;
    }
    if (thresh == 0 || thresh > 15) {
        PyBuffer_Release(&pot);
        PyBuffer_Release(&inp);
        PyErr_SetString(PyExc_ValueError, "threshold must be in [1, 15]");
        return NULL;
    }

    const size_t need = nrl_v1_braincore_packed_bytes((size_t)neurons);
    if (need == 0u || (size_t)pot.len < need || (size_t)inp.len < need) {
        PyBuffer_Release(&pot);
        PyBuffer_Release(&inp);
        PyErr_Format(PyExc_ValueError,
                     "packed buffer length mismatch: need %zu bytes per buffer, got %zd and %zd",
                     need, pot.len, inp.len);
        return NULL;
    }

    nrl_v1_status rc = NRL_ERR_INTERNAL;
    double t0 = 0.0;
    double t1 = 0.0;
#if defined(_WIN32)
    LARGE_INTEGER freq = {0};
    LARGE_INTEGER c0 = {0};
    LARGE_INTEGER c1 = {0};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c0);
    Py_BEGIN_ALLOW_THREADS
    rc = nrl_v1_braincore_int4((uint8_t *)pot.buf, (const uint8_t *)inp.buf, (size_t)neurons,
                               (size_t)iter, thresh);
    Py_END_ALLOW_THREADS
    QueryPerformanceCounter(&c1);
    t0 = (double)c0.QuadPart / (double)freq.QuadPart;
    t1 = (double)c1.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts0;
    struct timespec ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    Py_BEGIN_ALLOW_THREADS
    rc = nrl_v1_braincore_int4((uint8_t *)pot.buf, (const uint8_t *)inp.buf, (size_t)neurons,
                               (size_t)iter, thresh);
    Py_END_ALLOW_THREADS
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    t0 = (double)ts0.tv_sec + (double)ts0.tv_nsec * 1e-9;
    t1 = (double)ts1.tv_sec + (double)ts1.tv_nsec * 1e-9;
#endif

    PyBuffer_Release(&pot);
    PyBuffer_Release(&inp);
    if (rc != NRL_OK) {
        PyErr_SetString(NrlPyError, "braincore_int4_inplace failed");
        return NULL;
    }

    const double sec = (t1 > t0) ? (t1 - t0) : 0.0;
    return Py_BuildValue(
        "{s:s,s:s,s:n,s:n,s:i,s:d}",
        "kernel", "braincore_int4",
        "variant", nrl_v1_active_variant("braincore_int4"),
        "neurons", neurons,
        "iterations", iter,
        "threshold", (int)thresh,
        "seconds", sec);
}

static PyObject *py_braincore_int4(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    Py_ssize_t neurons = 8 * 1000 * 1000;
    Py_ssize_t iter = 1000;
    unsigned char thresh = 12;
    static char *kwlist[] = {"neurons", "iterations", "threshold", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|nnB:braincore_int4", kwlist,
                                     &neurons, &iter, &thresh)) {
        return NULL;
    }
    if (neurons <= 0 || iter <= 0) {
        PyErr_SetString(PyExc_ValueError, "neurons and iterations must be positive");
        return NULL;
    }
    if (thresh == 0 || thresh > 15) {
        PyErr_SetString(PyExc_ValueError, "threshold must be in [1, 15]");
        return NULL;
    }
    if ((neurons & 1) != 0) {
        neurons += 1;
    }

    const size_t byte_count = ((size_t)neurons) / 2u;
    uint8_t *pot = (uint8_t *)malloc(byte_count);
    uint8_t *inp = (uint8_t *)malloc(byte_count);
    if (pot == NULL || inp == NULL) {
        free(pot);
        free(inp);
        PyErr_NoMemory();
        return NULL;
    }
    memset(pot, 0, byte_count);
    for (size_t i = 0; i < byte_count; ++i) {
        inp[i] = (uint8_t)(((uint8_t)i * 37u) & 0x77u);
    }

    nrl_v1_status rc = NRL_ERR_INTERNAL;
    double t0 = 0.0;
    double t1 = 0.0;
#if defined(_WIN32)
    LARGE_INTEGER freq = {0};
    LARGE_INTEGER c0 = {0};
    LARGE_INTEGER c1 = {0};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c0);
    Py_BEGIN_ALLOW_THREADS
    rc = nrl_v1_braincore_int4(pot, inp, (size_t)neurons, (size_t)iter, thresh);
    Py_END_ALLOW_THREADS
    QueryPerformanceCounter(&c1);
    t0 = (double)c0.QuadPart / (double)freq.QuadPart;
    t1 = (double)c1.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts0;
    struct timespec ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    Py_BEGIN_ALLOW_THREADS
    rc = nrl_v1_braincore_int4(pot, inp, (size_t)neurons, (size_t)iter, thresh);
    Py_END_ALLOW_THREADS
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    t0 = (double)ts0.tv_sec + (double)ts0.tv_nsec * 1e-9;
    t1 = (double)ts1.tv_sec + (double)ts1.tv_nsec * 1e-9;
#endif

    free(pot);
    free(inp);
    if (rc != NRL_OK) {
        PyErr_SetString(NrlPyError, "braincore_int4 failed");
        return NULL;
    }

    const double sec = (t1 > t0) ? (t1 - t0) : 0.0;
    const double gps = sec > 0.0 ? ((double)neurons * (double)iter) / sec / 1e9 : 0.0;
    return Py_BuildValue(
        "{s:s,s:s,s:n,s:n,s:i,s:d,s:d}",
        "kernel", "braincore_int4",
        "variant", nrl_v1_active_variant("braincore_int4"),
        "neurons", neurons,
        "iterations", iter,
        "threshold", (int)thresh,
        "seconds", sec,
        "giga_neurons_per_sec", gps);
}

/* --------------------------------------------------------------------------
 * Phase 7-EG: native ladder + libllama bridge bindings.
 *
 * These add a Python-callable surface around the C resolution ladder so
 * nrlpy.native_ladder can drive R0..R5 dispatch from native code while
 * keeping the deterministic R0/R1/R2 candidate computation in Python (the
 * parity-gate contract). The libllama bridge is exposed in CALLBACK mode
 * with a Python callable as the backend; in Phase 8-EG this is rebound to
 * a direct libllama linkage with no surface change.
 * --------------------------------------------------------------------------*/

static PyObject *g_llama_callback_pyobj = NULL;  /* owned reference */

static nrl_v1_status nrlpy_llama_callback_thunk(
    void *user_data,
    const nrl_llama_request *request,
    nrl_llama_response *response) {
    (void)user_data;
    if (g_llama_callback_pyobj == NULL || request == NULL || response == NULL) {
        return NRL_ERR_ARGS;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    nrl_v1_status status = NRL_ERR_INTERNAL;
    PyObject *py_request = Py_BuildValue(
        "{s:s,s:s,s:i,s:i,s:i,s:i,s:i,s:f,s:f,s:i,s:f}",
        "model",         request->model_utf8 ? request->model_utf8 : "",
        "prompt",        request->prompt_utf8 ? request->prompt_utf8 : "",
        "max_tokens",    (int)request->max_tokens,
        "seed",          (int)request->seed,
        "n_ctx",         (int)request->n_ctx,
        "n_threads",     (int)request->n_threads,
        "n_batch",       (int)request->n_batch,
        "temperature",   (double)request->temperature,
        "top_p",         (double)request->top_p,
        "top_k",         (int)request->top_k,
        "repeat_penalty",(double)request->repeat_penalty);
    if (py_request == NULL) {
        PyGILState_Release(gstate);
        return NRL_ERR_ALLOC;
    }
    PyObject *result = PyObject_CallOneArg(g_llama_callback_pyobj, py_request);
    Py_DECREF(py_request);
    if (result == NULL) {
        PyErr_Clear();
        PyGILState_Release(gstate);
        return NRL_ERR_INTERNAL;
    }
    /* Expect a dict: {"text": str, "tokens": int}. */
    if (PyDict_Check(result)) {
        PyObject *py_text = PyDict_GetItemString(result, "text");
        PyObject *py_tokens = PyDict_GetItemString(result, "tokens");
        const char *text = "";
        Py_ssize_t text_len = 0;
        if (py_text != NULL && PyUnicode_Check(py_text)) {
            text = PyUnicode_AsUTF8AndSize(py_text, &text_len);
            if (text == NULL) text = "";
        }
        long tokens = 0;
        if (py_tokens != NULL && PyLong_Check(py_tokens)) {
            tokens = PyLong_AsLong(py_tokens);
        }
        if (response->text_buf != NULL && response->text_buf_capacity > 0) {
            const size_t cap = response->text_buf_capacity;
            const size_t writable = cap - 1;
            const size_t to_copy = ((size_t)text_len <= writable)
                ? (size_t)text_len : writable;
            if (to_copy > 0) {
                memcpy(response->text_buf, text, to_copy);
            }
            response->text_buf[to_copy] = '\0';
            response->text_byte_len = (size_t)text_len;
            response->text_truncated = ((size_t)text_len > writable) ? 1 : 0;
        } else {
            response->text_byte_len = (size_t)text_len;
            response->text_truncated = 1;
        }
        response->tokens_emitted = (int32_t)tokens;
        status = NRL_OK;
    } else {
        status = NRL_ERR_INTERNAL;
    }
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return status;
}

static PyObject *py_llama_set_backend(PyObject *self, PyObject *args) {
    (void)self;
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "s:llama_set_backend", &name)) {
        return NULL;
    }
    nrl_llama_backend backend;
    if (strcmp(name, "stub") == 0)          backend = NRL_LLAMA_BACKEND_STUB;
    else if (strcmp(name, "callback") == 0) backend = NRL_LLAMA_BACKEND_CALLBACK;
    else {
        PyErr_Format(PyExc_ValueError,
                     "unknown libllama backend %R; expected 'stub' or 'callback'",
                     PyUnicode_FromString(name));
        return NULL;
    }
    if (nrl_v1_llama_set_backend(backend) != NRL_OK) {
        PyErr_SetString(NrlPyError, "nrl_v1_llama_set_backend failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_llama_get_backend(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString(nrl_v1_llama_backend_name());
}

static PyObject *py_llama_set_callback(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cb = NULL;
    if (!PyArg_ParseTuple(args, "O:llama_set_callback", &cb)) {
        return NULL;
    }
    if (cb == Py_None) {
        Py_XDECREF(g_llama_callback_pyobj);
        g_llama_callback_pyobj = NULL;
        nrl_v1_llama_set_callback(NULL, NULL);
        Py_RETURN_NONE;
    }
    if (!PyCallable_Check(cb)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }
    Py_INCREF(cb);
    Py_XDECREF(g_llama_callback_pyobj);
    g_llama_callback_pyobj = cb;
    nrl_v1_llama_set_callback(nrlpy_llama_callback_thunk, NULL);
    Py_RETURN_NONE;
}

static PyObject *py_lane_allows_r2_active(PyObject *self, PyObject *args) {
    (void)self;
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "s:lane_allows_r2_active", &name)) {
        return NULL;
    }
    int32_t lane = nrl_v1_coherence_lane_from_str(name);
    if (lane < 0) {
        Py_RETURN_FALSE;
    }
    if (nrl_v1_lane_allows_r2_active(lane)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *py_ladder_rung_name(PyObject *self, PyObject *args) {
    (void)self;
    int rung = -1;
    if (!PyArg_ParseTuple(args, "i:ladder_rung_name", &rung)) {
        return NULL;
    }
    return PyUnicode_FromString(nrl_v1_ladder_rung_name((int32_t)rung));
}

/* The big one: drive a single ladder turn through the native dispatcher.
 *
 * Argument is one big dict with the layout the Python wrapper builds; this
 * keeps the binding signature stable across C-side ABI growth. */
static PyObject *py_ladder_resolve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *spec = NULL;
    if (!PyArg_ParseTuple(args, "O!:ladder_resolve", &PyDict_Type, &spec)) {
        return NULL;
    }

    nrl_ladder_inputs inputs;
    memset(&inputs, 0, sizeof(inputs));
    nrl_ladder_outputs outputs;
    memset(&outputs, 0, sizeof(outputs));

    /* Lane + flags. */
    PyObject *py_lane = PyDict_GetItemString(spec, "coherence_lane");
    const char *lane_name = "fast-stable";
    if (py_lane != NULL && PyUnicode_Check(py_lane)) {
        lane_name = PyUnicode_AsUTF8(py_lane);
        if (lane_name == NULL) lane_name = "fast-stable";
    }
    inputs.coherence_lane = nrl_v1_coherence_lane_from_str(lane_name);
    if (inputs.coherence_lane < 0) {
        inputs.coherence_lane = NRL_COHERENCE_LANE_FAST_STABLE;
    }
    PyObject *py_r2_enabled = PyDict_GetItemString(spec, "r2_shadow_enabled");
    inputs.r2_shadow_enabled = (py_r2_enabled != NULL
        && PyObject_IsTrue(py_r2_enabled)) ? 1 : 0;

    /* Helper macro to lift a candidate dict from spec[name] into a slot. */
#define LIFT_CANDIDATE(slot_name, slot_field) \
    do { \
        PyObject *cand = PyDict_GetItemString(spec, slot_name); \
        if (cand != NULL && PyDict_Check(cand)) { \
            PyObject *py_avail = PyDict_GetItemString(cand, "available"); \
            PyObject *py_tokens = PyDict_GetItemString(cand, "tokens"); \
            PyObject *py_text = PyDict_GetItemString(cand, "text"); \
            PyObject *py_wall = PyDict_GetItemString(cand, "wall_s"); \
            inputs.slot_field.available = (py_avail != NULL \
                && PyObject_IsTrue(py_avail)) ? 1 : 0; \
            inputs.slot_field.tokens = (py_tokens != NULL \
                && PyLong_Check(py_tokens)) \
                ? (int32_t)PyLong_AsLong(py_tokens) : 0; \
            inputs.slot_field.text_utf8 = (py_text != NULL \
                && PyUnicode_Check(py_text)) \
                ? PyUnicode_AsUTF8(py_text) : NULL; \
            inputs.slot_field.precomputed_wall_s = (py_wall != NULL \
                && PyFloat_Check(py_wall)) \
                ? PyFloat_AsDouble(py_wall) : 0.0; \
        } \
    } while (0)

    LIFT_CANDIDATE("r0", r0);
    LIFT_CANDIDATE("r1", r1);
    LIFT_CANDIDATE("r2_active", r2_active);
#undef LIFT_CANDIDATE

    /* R5 request. */
    PyObject *py_r5 = PyDict_GetItemString(spec, "r5_request");
    if (py_r5 != NULL && PyDict_Check(py_r5)) {
        PyObject *o;
        o = PyDict_GetItemString(py_r5, "model");
        inputs.r5_request.model_utf8 = (o != NULL && PyUnicode_Check(o))
            ? PyUnicode_AsUTF8(o) : "";
        o = PyDict_GetItemString(py_r5, "prompt");
        inputs.r5_request.prompt_utf8 = (o != NULL && PyUnicode_Check(o))
            ? PyUnicode_AsUTF8(o) : "";
        o = PyDict_GetItemString(py_r5, "max_tokens");
        inputs.r5_request.max_tokens = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "seed");
        inputs.r5_request.seed = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "n_ctx");
        inputs.r5_request.n_ctx = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "n_threads");
        inputs.r5_request.n_threads = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "n_batch");
        inputs.r5_request.n_batch = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "temperature");
        inputs.r5_request.temperature = (o != NULL && PyFloat_Check(o))
            ? (float)PyFloat_AsDouble(o) : 0.0f;
        o = PyDict_GetItemString(py_r5, "top_p");
        inputs.r5_request.top_p = (o != NULL && PyFloat_Check(o))
            ? (float)PyFloat_AsDouble(o) : 0.0f;
        o = PyDict_GetItemString(py_r5, "top_k");
        inputs.r5_request.top_k = (o != NULL && PyLong_Check(o))
            ? (int32_t)PyLong_AsLong(o) : 0;
        o = PyDict_GetItemString(py_r5, "repeat_penalty");
        inputs.r5_request.repeat_penalty = (o != NULL && PyFloat_Check(o))
            ? (float)PyFloat_AsDouble(o) : 0.0f;
    }

    /* Allocate an output buffer sized from spec["text_buf_capacity"]. */
    Py_ssize_t cap = 16384;
    PyObject *py_cap = PyDict_GetItemString(spec, "text_buf_capacity");
    if (py_cap != NULL && PyLong_Check(py_cap)) {
        long v = PyLong_AsLong(py_cap);
        if (v > 0 && v < (long)(64 * 1024 * 1024)) {
            cap = (Py_ssize_t)v;
        }
    }
    char *buf = (char *)PyMem_Malloc((size_t)cap);
    if (buf == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    buf[0] = '\0';
    outputs.text_buf = buf;
    outputs.text_buf_capacity = (size_t)cap;

    nrl_v1_status rc = nrl_v1_ladder_resolve(&inputs, &outputs);

    PyObject *result = NULL;
    if (rc == NRL_OK) {
        result = Py_BuildValue(
            "{s:i,s:s,s:i,s:s#,s:i,s:n,s:d,s:i}",
            "served_rung",     (int)outputs.served_rung,
            "served_rung_name", nrl_v1_ladder_rung_name(outputs.served_rung),
            "tokens_emitted",  (int)outputs.tokens_emitted,
            "text",             outputs.text_buf,
                               (Py_ssize_t)((outputs.text_buf_capacity > 0)
                                  ? strnlen(outputs.text_buf, outputs.text_buf_capacity)
                                  : 0),
            "text_truncated",  (int)outputs.text_truncated,
            "text_byte_len",   (Py_ssize_t)outputs.text_byte_len,
            "wall_seconds",    (double)outputs.wall_seconds,
            "served_class",    (int)outputs.served_class);
    } else {
        PyErr_Format(NrlPyError, "ladder_resolve failed: rc=%d", (int)rc);
    }
    PyMem_Free(buf);
    return result;
}

/* --------------------------------------------------------------------------
 * Phase 8-EG: full-native hot path bindings.
 *
 * Exposes:
 *   * ``mm_lookup``     - native muscle-memory probe (R0).
 *   * ``zpm_lookup``    - native ZPM nullspace probe (R1).
 *   * ``r2_set_callback`` / ``r2_has_callback`` - R2 bridge callback.
 *   * ``ladder_run_turn_full`` - full-native turn orchestrator that
 *     drives R0 -> R1 -> R2 (callback) -> R5 without entering Python
 *     on the hot path.
 *
 * Like the Phase 7-EG bindings, dicts are used for request/response so
 * the ABI can grow without breaking the binding signature.
 * --------------------------------------------------------------------------*/

static PyObject *g_r2_callback_pyobj = NULL;  /* owned reference */

static nrl_v1_status nrlpy_r2_callback_thunk(
    void *user_data,
    const nrl_r2_probe_request *request,
    nrl_r2_probe_response *response) {
    (void)user_data;
    if (g_r2_callback_pyobj == NULL || request == NULL || response == NULL) {
        return NRL_ERR_ARGS;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    nrl_v1_status status = NRL_ERR_INTERNAL;
    PyObject *intent_bytes = NULL;
    if (request->intent_anchor_bytes != NULL && request->intent_anchor_len > 0) {
        intent_bytes = PyBytes_FromStringAndSize(
            (const char *)request->intent_anchor_bytes,
            (Py_ssize_t)request->intent_anchor_len);
    } else {
        intent_bytes = PyBytes_FromStringAndSize("", 0);
    }
    if (intent_bytes == NULL) {
        PyGILState_Release(gstate);
        return NRL_ERR_ALLOC;
    }
    PyObject *py_request = Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:d,s:s,s:O}",
        "coherence_lane",       (int)request->coherence_lane,
        "r2_shadow_enabled",    (int)request->r2_shadow_enabled,
        "zpm_threshold_bits",   (int)request->zpm_threshold_bits,
        "omega_iterations",     (int)request->omega_iterations,
        "omega_candidates",     (int)request->omega_candidates,
        "omega_budget_ms",      (double)request->omega_budget_ms,
        "model_sha256",
            request->model_sha256_utf8 ? request->model_sha256_utf8 : "",
        "intent_anchor_bytes",  intent_bytes);
    Py_DECREF(intent_bytes);
    if (py_request == NULL) {
        PyGILState_Release(gstate);
        return NRL_ERR_ALLOC;
    }
    PyObject *result = PyObject_CallOneArg(g_r2_callback_pyobj, py_request);
    Py_DECREF(py_request);
    if (result == NULL) {
        PyErr_Clear();
        PyGILState_Release(gstate);
        return NRL_ERR_INTERNAL;
    }
    if (PyDict_Check(result)) {
        PyObject *py_avail  = PyDict_GetItemString(result, "available");
        PyObject *py_tokens = PyDict_GetItemString(result, "tokens");
        PyObject *py_text   = PyDict_GetItemString(result, "text");
        PyObject *py_idx    = PyDict_GetItemString(result, "stored_entry_index");
        PyObject *py_dist   = PyDict_GetItemString(result, "distance_bits");
        PyObject *py_wall   = PyDict_GetItemString(result, "wall_seconds");

        const char *text = "";
        Py_ssize_t text_len = 0;
        if (py_text != NULL && PyUnicode_Check(py_text)) {
            text = PyUnicode_AsUTF8AndSize(py_text, &text_len);
            if (text == NULL) text = "";
        }
        response->available = (py_avail != NULL && PyObject_IsTrue(py_avail))
            ? 1 : 0;
        response->tokens_emitted = (py_tokens != NULL && PyLong_Check(py_tokens))
            ? (int32_t)PyLong_AsLong(py_tokens) : 0;
        response->stored_entry_index = (py_idx != NULL && PyLong_Check(py_idx))
            ? (int32_t)PyLong_AsLong(py_idx) : -1;
        response->distance_bits = (py_dist != NULL && PyLong_Check(py_dist))
            ? (int32_t)PyLong_AsLong(py_dist) : 256;
        response->wall_seconds = (py_wall != NULL && PyFloat_Check(py_wall))
            ? PyFloat_AsDouble(py_wall) : 0.0;
        if (response->text_buf != NULL && response->text_buf_capacity > 0) {
            const size_t cap = response->text_buf_capacity;
            const size_t writable = cap - 1;
            const size_t to_copy = ((size_t)text_len <= writable)
                ? (size_t)text_len : writable;
            if (to_copy > 0) memcpy(response->text_buf, text, to_copy);
            response->text_buf[to_copy] = '\0';
            response->text_byte_len = (size_t)text_len;
            response->text_truncated = ((size_t)text_len > writable) ? 1 : 0;
        } else {
            response->text_byte_len = (size_t)text_len;
            response->text_truncated = 1;
        }
        status = NRL_OK;
    } else {
        status = NRL_ERR_INTERNAL;
    }
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return status;
}

static PyObject *py_r2_set_callback(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cb = NULL;
    if (!PyArg_ParseTuple(args, "O:r2_set_callback", &cb)) return NULL;
    if (cb == Py_None) {
        Py_XDECREF(g_r2_callback_pyobj);
        g_r2_callback_pyobj = NULL;
        nrl_v1_r2_set_callback(NULL, NULL);
        Py_RETURN_NONE;
    }
    if (!PyCallable_Check(cb)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }
    Py_INCREF(cb);
    Py_XDECREF(g_r2_callback_pyobj);
    g_r2_callback_pyobj = cb;
    nrl_v1_r2_set_callback(nrlpy_r2_callback_thunk, NULL);
    Py_RETURN_NONE;
}

static PyObject *py_r2_has_callback(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyBool_FromLong(nrl_v1_r2_has_callback() ? 1 : 0);
}

/* Helper: pull a required string value out of a dict, falling back to
 * an empty string when missing. Returned pointer is borrowed from the
 * dict value's internal UTF-8 buffer; valid for the lifetime of that
 * value (caller must not outlive the dict). */
static const char *dict_get_utf8(PyObject *d, const char *key) {
    if (d == NULL) return "";
    PyObject *v = PyDict_GetItemString(d, key);
    if (v == NULL || !PyUnicode_Check(v)) return "";
    const char *s = PyUnicode_AsUTF8(v);
    return s ? s : "";
}

static int dict_get_int(PyObject *d, const char *key, int fallback) {
    if (d == NULL) return fallback;
    PyObject *v = PyDict_GetItemString(d, key);
    if (v == NULL || !PyLong_Check(v)) return fallback;
    return (int)PyLong_AsLong(v);
}

static double dict_get_double(PyObject *d, const char *key, double fallback) {
    if (d == NULL) return fallback;
    PyObject *v = PyDict_GetItemString(d, key);
    if (v == NULL) return fallback;
    if (PyFloat_Check(v)) return PyFloat_AsDouble(v);
    if (PyLong_Check(v))  return (double)PyLong_AsLong(v);
    return fallback;
}

/* mm_lookup({root_dir, model_sha256, prompt, sampler_fingerprint, seed,
 *            max_tokens, muscle_memory_on, text_buf_capacity?}) -> dict */
static PyObject *py_mm_lookup(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *spec = NULL;
    if (!PyArg_ParseTuple(args, "O!:mm_lookup", &PyDict_Type, &spec)) return NULL;

    nrl_mm_lookup_request req;
    memset(&req, 0, sizeof(req));
    req.root_dir_utf8            = dict_get_utf8(spec, "root_dir");
    req.model_sha256_utf8        = dict_get_utf8(spec, "model_sha256");
    req.prompt_utf8              = dict_get_utf8(spec, "prompt");
    req.sampler_fingerprint_utf8 = dict_get_utf8(spec, "sampler_fingerprint");
    req.seed       = (int32_t)dict_get_int(spec, "seed", 0);
    req.max_tokens = (int32_t)dict_get_int(spec, "max_tokens", 0);
    req.muscle_memory_on = (int32_t)dict_get_int(spec, "muscle_memory_on", 0);

    Py_ssize_t cap = 65536;
    PyObject *py_cap = PyDict_GetItemString(spec, "text_buf_capacity");
    if (py_cap != NULL && PyLong_Check(py_cap)) {
        long v = PyLong_AsLong(py_cap);
        if (v > 0 && v < (long)(256 * 1024 * 1024)) cap = (Py_ssize_t)v;
    }
    char *buf = (char *)PyMem_Malloc((size_t)cap);
    if (buf == NULL) { PyErr_NoMemory(); return NULL; }
    buf[0] = '\0';

    nrl_mm_lookup_result res;
    memset(&res, 0, sizeof(res));
    res.text_buf = buf;
    res.text_buf_capacity = (size_t)cap;
    nrl_v1_status rc = nrl_v1_mm_lookup(&req, &res);
    if (rc != NRL_OK) {
        PyMem_Free(buf);
        PyErr_Format(NrlPyError, "mm_lookup failed: rc=%d", (int)rc);
        return NULL;
    }
    PyObject *out = Py_BuildValue(
        "{s:i,s:i,s:K,s:d,s:s#,s:i,s:n}",
        "hit",             (int)res.hit,
        "tokens_emitted",  (int)res.tokens_emitted,
        "key_fnv1a64",     (unsigned long long)res.key_fnv1a64,
        "wall_seconds",    (double)res.wall_seconds,
        "text",            buf,
                           (Py_ssize_t)((res.text_buf_capacity > 0)
                                ? strnlen(buf, res.text_buf_capacity) : 0),
        "text_truncated",  (int)res.text_truncated,
        "text_byte_len",   (Py_ssize_t)res.text_byte_len);
    PyMem_Free(buf);
    return out;
}

/* zpm_lookup({index_path, model_sha256, prompt, sampler_fingerprint,
 *             seed, max_tokens, threshold_bits, enabled,
 *             text_buf_capacity?}) -> dict */
static PyObject *py_zpm_lookup(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *spec = NULL;
    if (!PyArg_ParseTuple(args, "O!:zpm_lookup", &PyDict_Type, &spec)) return NULL;

    nrl_zpm_lookup_request req;
    memset(&req, 0, sizeof(req));
    req.index_path_utf8          = dict_get_utf8(spec, "index_path");
    req.model_sha256_utf8        = dict_get_utf8(spec, "model_sha256");
    req.prompt_utf8              = dict_get_utf8(spec, "prompt");
    req.sampler_fingerprint_utf8 = dict_get_utf8(spec, "sampler_fingerprint");
    req.seed            = (int32_t)dict_get_int(spec, "seed", 0);
    req.max_tokens      = (int32_t)dict_get_int(spec, "max_tokens", 0);
    req.threshold_bits  = (int32_t)dict_get_int(spec, "threshold_bits", 0);
    req.enabled         = (int32_t)dict_get_int(spec, "enabled", 0);

    Py_ssize_t cap = 65536;
    PyObject *py_cap = PyDict_GetItemString(spec, "text_buf_capacity");
    if (py_cap != NULL && PyLong_Check(py_cap)) {
        long v = PyLong_AsLong(py_cap);
        if (v > 0 && v < (long)(256 * 1024 * 1024)) cap = (Py_ssize_t)v;
    }
    char *buf = (char *)PyMem_Malloc((size_t)cap);
    if (buf == NULL) { PyErr_NoMemory(); return NULL; }
    buf[0] = '\0';

    nrl_zpm_lookup_result res;
    memset(&res, 0, sizeof(res));
    res.text_buf = buf;
    res.text_buf_capacity = (size_t)cap;
    nrl_v1_status rc = nrl_v1_zpm_lookup(&req, &res);
    if (rc != NRL_OK) {
        PyMem_Free(buf);
        PyErr_Format(NrlPyError, "zpm_lookup failed: rc=%d", (int)rc);
        return NULL;
    }
    PyObject *state_tup = Py_BuildValue(
        "(KKKK)",
        (unsigned long long)res.state[0],
        (unsigned long long)res.state[1],
        (unsigned long long)res.state[2],
        (unsigned long long)res.state[3]);
    PyObject *out = Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:O,s:d,s:s#,s:i,s:n}",
        "hit",             (int)res.hit,
        "tokens_emitted",  (int)res.tokens_emitted,
        "exact",           (int)res.exact,
        "within_threshold",(int)res.within_threshold,
        "distance_bits",   (int)res.distance_bits,
        "stored_entry_index",(int)res.stored_entry_index,
        "state",           state_tup,
        "wall_seconds",    (double)res.wall_seconds,
        "text",            buf,
                           (Py_ssize_t)((res.text_buf_capacity > 0)
                                ? strnlen(buf, res.text_buf_capacity) : 0),
        "text_truncated",  (int)res.text_truncated,
        "text_byte_len",   (Py_ssize_t)res.text_byte_len);
    Py_XDECREF(state_tup);
    PyMem_Free(buf);
    return out;
}

/* ladder_run_turn_full(spec) -> dict
 *
 * spec keys:
 *   mm:   {root_dir, model_sha256, prompt, sampler_fingerprint, seed,
 *          max_tokens, muscle_memory_on}
 *   zpm:  {index_path, model_sha256, prompt, sampler_fingerprint, seed,
 *          max_tokens, threshold_bits, enabled}
 *   coherence_lane, r2_shadow_enabled, zpm_threshold_bits,
 *   omega_iterations, omega_candidates, omega_budget_ms
 *   intent_anchor_bytes: bytes
 *   r5_request: {model, prompt, max_tokens, seed, n_ctx, n_threads,
 *                n_batch, temperature, top_p, top_k, repeat_penalty}
 *   text_buf_capacity (optional)
 */
static PyObject *py_ladder_run_turn_full(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *spec = NULL;
    if (!PyArg_ParseTuple(args, "O!:ladder_run_turn_full", &PyDict_Type, &spec))
        return NULL;

    nrl_full_turn_request req;
    memset(&req, 0, sizeof(req));

    PyObject *mm_d  = PyDict_GetItemString(spec, "mm");
    PyObject *zpm_d = PyDict_GetItemString(spec, "zpm");

    req.mm.root_dir_utf8            = dict_get_utf8(mm_d, "root_dir");
    req.mm.model_sha256_utf8        = dict_get_utf8(mm_d, "model_sha256");
    req.mm.prompt_utf8              = dict_get_utf8(mm_d, "prompt");
    req.mm.sampler_fingerprint_utf8 = dict_get_utf8(mm_d, "sampler_fingerprint");
    req.mm.seed       = (int32_t)dict_get_int(mm_d, "seed", 0);
    req.mm.max_tokens = (int32_t)dict_get_int(mm_d, "max_tokens", 0);
    req.mm.muscle_memory_on = (int32_t)dict_get_int(mm_d, "muscle_memory_on", 0);

    req.zpm.index_path_utf8           = dict_get_utf8(zpm_d, "index_path");
    req.zpm.model_sha256_utf8         = dict_get_utf8(zpm_d, "model_sha256");
    req.zpm.prompt_utf8               = dict_get_utf8(zpm_d, "prompt");
    req.zpm.sampler_fingerprint_utf8  = dict_get_utf8(zpm_d, "sampler_fingerprint");
    req.zpm.seed            = (int32_t)dict_get_int(zpm_d, "seed", 0);
    req.zpm.max_tokens      = (int32_t)dict_get_int(zpm_d, "max_tokens", 0);
    req.zpm.threshold_bits  = (int32_t)dict_get_int(zpm_d, "threshold_bits", 0);
    req.zpm.enabled         = (int32_t)dict_get_int(zpm_d, "enabled", 0);

    const char *lane_name = dict_get_utf8(spec, "coherence_lane");
    if (!lane_name || !lane_name[0]) lane_name = "fast-stable";
    req.coherence_lane = nrl_v1_coherence_lane_from_str(lane_name);
    if (req.coherence_lane < 0) req.coherence_lane = NRL_COHERENCE_LANE_FAST_STABLE;
    req.r2_shadow_enabled  = (int32_t)dict_get_int(spec, "r2_shadow_enabled", 0);
    req.zpm_threshold_bits = (int32_t)dict_get_int(spec, "zpm_threshold_bits", 0);
    req.omega_iterations   = (int32_t)dict_get_int(spec, "omega_iterations", 0);
    req.omega_candidates   = (int32_t)dict_get_int(spec, "omega_candidates", 0);
    req.omega_budget_ms    = dict_get_double(spec, "omega_budget_ms", 0.0);

    /* intent_anchor_bytes is a bytes object. Borrow its buffer for the
     * call duration (we don't call into Python reentrantly until the
     * R2 thunk, which itself makes a new bytes copy). */
    PyObject *py_intent = PyDict_GetItemString(spec, "intent_anchor_bytes");
    if (py_intent != NULL && PyBytes_Check(py_intent)) {
        Py_ssize_t n = 0;
        char *data = NULL;
        PyBytes_AsStringAndSize(py_intent, &data, &n);
        req.intent_anchor_bytes = (const uint8_t *)data;
        req.intent_anchor_len = (size_t)n;
    }

    PyObject *r5_d = PyDict_GetItemString(spec, "r5_request");
    req.r5_request.model_utf8  = dict_get_utf8(r5_d, "model");
    req.r5_request.prompt_utf8 = dict_get_utf8(r5_d, "prompt");
    req.r5_request.max_tokens     = (int32_t)dict_get_int(r5_d, "max_tokens", 0);
    req.r5_request.seed           = (int32_t)dict_get_int(r5_d, "seed", 0);
    req.r5_request.n_ctx          = (int32_t)dict_get_int(r5_d, "n_ctx", 0);
    req.r5_request.n_threads      = (int32_t)dict_get_int(r5_d, "n_threads", 0);
    req.r5_request.n_batch        = (int32_t)dict_get_int(r5_d, "n_batch", 0);
    req.r5_request.temperature    = (float)dict_get_double(r5_d, "temperature", 0.0);
    req.r5_request.top_p          = (float)dict_get_double(r5_d, "top_p", 0.0);
    req.r5_request.top_k          = (int32_t)dict_get_int(r5_d, "top_k", 0);
    req.r5_request.repeat_penalty = (float)dict_get_double(r5_d, "repeat_penalty", 0.0);

    Py_ssize_t cap = 65536;
    PyObject *py_cap = PyDict_GetItemString(spec, "text_buf_capacity");
    if (py_cap != NULL && PyLong_Check(py_cap)) {
        long v = PyLong_AsLong(py_cap);
        if (v > 0 && v < (long)(256 * 1024 * 1024)) cap = (Py_ssize_t)v;
    }
    char *buf = (char *)PyMem_Malloc((size_t)cap);
    if (buf == NULL) { PyErr_NoMemory(); return NULL; }
    buf[0] = '\0';

    nrl_full_turn_result res;
    memset(&res, 0, sizeof(res));
    res.text_buf = buf;
    res.text_buf_capacity = (size_t)cap;

    nrl_v1_status rc = nrl_v1_ladder_run_turn(&req, &res);
    if (rc != NRL_OK) {
        PyMem_Free(buf);
        PyErr_Format(NrlPyError, "ladder_run_turn_full failed: rc=%d", (int)rc);
        return NULL;
    }

    PyObject *state_tup = Py_BuildValue(
        "(KKKK)",
        (unsigned long long)res.zpm_report.state[0],
        (unsigned long long)res.zpm_report.state[1],
        (unsigned long long)res.zpm_report.state[2],
        (unsigned long long)res.zpm_report.state[3]);

    PyObject *mm_dict = Py_BuildValue(
        "{s:i,s:i,s:K,s:d}",
        "hit",             (int)res.mm_report.hit,
        "tokens_emitted",  (int)res.mm_report.tokens_emitted,
        "key_fnv1a64",     (unsigned long long)res.mm_report.key_fnv1a64,
        "wall_seconds",    (double)res.mm_report.wall_seconds);

    PyObject *zpm_dict = Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:O,s:d}",
        "hit",             (int)res.zpm_report.hit,
        "tokens_emitted",  (int)res.zpm_report.tokens_emitted,
        "exact",           (int)res.zpm_report.exact,
        "within_threshold",(int)res.zpm_report.within_threshold,
        "distance_bits",   (int)res.zpm_report.distance_bits,
        "stored_entry_index",(int)res.zpm_report.stored_entry_index,
        "state",           state_tup,
        "wall_seconds",    (double)res.zpm_report.wall_seconds);
    Py_XDECREF(state_tup);

    PyObject *r2_dict = Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:d}",
        "available",          (int)res.r2_available,
        "tokens_emitted",     (int)res.r2_tokens_emitted,
        "stored_entry_index", (int)res.r2_stored_entry_index,
        "distance_bits",      (int)res.r2_distance_bits,
        "wall_seconds",       (double)res.r2_wall_seconds);

    PyObject *r5_dict = Py_BuildValue(
        "{s:i,s:i,s:d}",
        "invoked",        (int)res.r5_invoked,
        "tokens_emitted", (int)res.r5_tokens_emitted,
        "wall_seconds",   (double)res.r5_wall_seconds);

    PyObject *out = Py_BuildValue(
        "{s:i,s:s,s:i,s:s#,s:i,s:n,s:d,s:O,s:O,s:O,s:O}",
        "served_rung",      (int)res.served_rung,
        "served_rung_name", nrl_v1_ladder_rung_name(res.served_rung),
        "tokens_emitted",   (int)res.tokens_emitted,
        "text",              buf,
                            (Py_ssize_t)((res.text_buf_capacity > 0)
                                ? strnlen(buf, res.text_buf_capacity) : 0),
        "text_truncated",   (int)res.text_truncated,
        "text_byte_len",    (Py_ssize_t)res.text_byte_len,
        "wall_seconds",     (double)res.wall_seconds,
        "mm_report",        mm_dict,
        "zpm_report",       zpm_dict,
        "r2_report",        r2_dict,
        "r5_report",        r5_dict);
    Py_XDECREF(mm_dict);
    Py_XDECREF(zpm_dict);
    Py_XDECREF(r2_dict);
    Py_XDECREF(r5_dict);
    PyMem_Free(buf);
    return out;
}

#define NRLPY_KW_METHOD(fn) ((PyCFunction)(void (*)(void))(fn))

/* --- Phase 11: native FNV-1a64 for hot-path absorption/hash loops ---
 *
 * Matches `checksum_u64` in engine/src/main.c byte-for-byte so anything
 * written by the pure-Python `nrlpy.runtime.fnv1a64_packed` continues to
 * round-trip. Accepts anything implementing the buffer protocol (bytes,
 * bytearray, memoryview, mmap) and returns the 64-bit digest as a
 * Python int.
 */
static PyObject *py_fnv1a64_bytes(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer buf = {0};
    if (!PyArg_ParseTuple(args, "y*:fnv1a64_bytes", &buf)) {
        return NULL;
    }
    const unsigned char *p = (const unsigned char *)buf.buf;
    Py_ssize_t n = buf.len;
    uint64_t x = 1469598103934665603ull;
    const uint64_t prime = 1099511628211ull;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        x ^= (uint64_t)p[i];
        x *= prime;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&buf);
    return PyLong_FromUnsignedLongLong((unsigned long long)x);
}

/* --- Phase 11: native pack_int4_from_bytes ---
 *
 * Folds `raw` into `units` deterministic INT4 nibbles by hashing each of
 * `units` evenly-spaced slices with FNV-1a64 and taking the low nibble.
 * Preserves byte-identical output with the pure-Python reference in
 * `nrlpy.lmo.pack_int4_from_bytes`, so any existing LMO or Stage A-VI
 * attestation continues to verify. This function is the hot path of
 * LMO absorption for large GGUFs; the pure-Python implementation is
 * O(units) Python-level loop overhead per call and is pathologically
 * slow on multi-GB models.
 */
static PyObject *py_pack_int4_from_bytes(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer raw = {0};
    Py_ssize_t units = 0;
    if (!PyArg_ParseTuple(args, "y*n:pack_int4_from_bytes", &raw, &units)) {
        return NULL;
    }
    if (units <= 0 || raw.len <= 0) {
        PyBuffer_Release(&raw);
        return PyBytes_FromStringAndSize(NULL, 0);
    }
    Py_ssize_t out_len = (units + 1) / 2;
    PyObject *out_obj = PyBytes_FromStringAndSize(NULL, out_len);
    if (!out_obj) {
        PyBuffer_Release(&raw);
        return NULL;
    }
    unsigned char *out = (unsigned char *)PyBytes_AS_STRING(out_obj);
    const unsigned char *p = (const unsigned char *)raw.buf;
    const Py_ssize_t n = raw.len;
    const uint64_t prime = 1099511628211ull;
    const uint64_t seed = 1469598103934665603ull;
    Py_BEGIN_ALLOW_THREADS
    /* Zero the output (bytes may or may not start zeroed in CPython). */
    for (Py_ssize_t k = 0; k < out_len; k++) { out[k] = 0; }
    Py_ssize_t prev = 0;
    for (Py_ssize_t i = 0; i < units; i++) {
        Py_ssize_t end = ((i + 1) * n) / units;
        if (end <= prev) {
            end = (n < prev + 1) ? n : (prev + 1);
        }
        uint64_t x = seed;
        for (Py_ssize_t j = prev; j < end; j++) {
            x ^= (uint64_t)p[j];
            x *= prime;
        }
        unsigned int nibble = (unsigned int)(x & 0x0Full);
        if ((i & 1) == 0) {
            out[i / 2] = (unsigned char)(nibble & 0x0Fu);
        } else {
            out[i / 2] |= (unsigned char)((nibble & 0x0Fu) << 4);
        }
        prev = end;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&raw);
    return out_obj;
}

static PyMethodDef NrlPyMethods[] = {
    {"init", py_init, METH_NOARGS, "Initialize NRL runtime."},
    {"version", py_version, METH_NOARGS, "NRL version string."},
    {"features", py_features, METH_NOARGS, "CPU feature map."},
    {"active_variant", py_active_variant, METH_VARARGS, "Active variant by kernel name."},
    {"braincore_packed_bytes", py_braincore_packed_bytes, METH_VARARGS,
     "Packed byte count for even neuron_count (0 if invalid)."},
    {"braincore_int4_inplace", NRLPY_KW_METHOD(py_braincore_int4_inplace),
     METH_VARARGS | METH_KEYWORDS,
     "INT4 lattice update writing into caller-owned packed buffers (binary assimilation)."},
    {"braincore_int4", NRLPY_KW_METHOD(py_braincore_int4), METH_VARARGS | METH_KEYWORDS,
     "Run INT4 braincore kernel."},
    {"llama_set_backend", py_llama_set_backend, METH_VARARGS,
     "Phase 7-EG: select libllama bridge backend ('stub' | 'callback')."},
    {"llama_get_backend", py_llama_get_backend, METH_NOARGS,
     "Phase 7-EG: name of the active libllama bridge backend."},
    {"llama_set_callback", py_llama_set_callback, METH_VARARGS,
     "Phase 7-EG: register a Python callable as the libllama bridge backend."},
    {"lane_allows_r2_active", py_lane_allows_r2_active, METH_VARARGS,
     "Phase 7-EG: native lane gate for Rung R2 active mode."},
    {"ladder_rung_name", py_ladder_rung_name, METH_VARARGS,
     "Phase 7-EG: name for an integer rung id (0..5)."},
    {"ladder_resolve", py_ladder_resolve, METH_VARARGS,
     "Phase 7-EG: drive a single Resolution Ladder turn through the C dispatcher."},
    {"mm_lookup", py_mm_lookup, METH_VARARGS,
     "Phase 8-EG: native muscle-memory probe (R0)."},
    {"zpm_lookup", py_zpm_lookup, METH_VARARGS,
     "Phase 8-EG: native ZPM nullspace probe (R1)."},
    {"r2_set_callback", py_r2_set_callback, METH_VARARGS,
     "Phase 8-EG: register a Python callable as the R2 probe backend."},
    {"r2_has_callback", py_r2_has_callback, METH_NOARGS,
     "Phase 8-EG: True iff an R2 probe callback is currently installed."},
    {"ladder_run_turn_full", py_ladder_run_turn_full, METH_VARARGS,
     "Phase 8-EG: drive a full native turn (R0 -> R1 -> R2 -> R5) in C."},
    {"fnv1a64_bytes", py_fnv1a64_bytes, METH_VARARGS,
     "Phase 11: native FNV-1a64 over a bytes-like (matches checksum_u64)."},
    {"pack_int4_from_bytes", py_pack_int4_from_bytes, METH_VARARGS,
     "Phase 11: native pack_int4_from_bytes (hot path for LMO absorption)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef nrlpy_module = {
    PyModuleDef_HEAD_INIT,
    "nrlpy._core",
    "NRL engine bindings (machine-code kernels).",
    -1,
    NrlPyMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject *m = PyModule_Create(&nrlpy_module);
    if (m == NULL) {
        return NULL;
    }
    NrlPyError = PyErr_NewException("nrlpy._core.NrlPyError", NULL, NULL);
    Py_XINCREF(NrlPyError);
    if (PyModule_AddObject(m, "NrlPyError", NrlPyError) < 0) {
        Py_DECREF(NrlPyError);
        Py_DECREF(m);
        return NULL;
    }
    (void)nrl_v1_init();
    PyModule_AddStringConstant(m, "__version__", nrl_v1_version());
    return m;
}
