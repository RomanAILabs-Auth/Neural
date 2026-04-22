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

#define NRLPY_KW_METHOD(fn) ((PyCFunction)(void (*)(void))(fn))

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
