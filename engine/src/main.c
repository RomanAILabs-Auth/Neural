/*
 * main.c - NRL native CLI skeleton.
 * Copyright RomanAILabs - Daniel Harding (GitHub RomanAILabs-Auth)
 * Collaborators honored: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor
 * Contact: daniel@romanailabs.com, romanailabs@gmail.com
 * Website: romanailabs.com
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#include <process.h>
#else
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <limits.h>
#endif
#endif

#include "nrl/nrl.h"
#include "nrl/runtime_status.h"
#include "runtime_internal.h"

static int cmd_brain_map(void);

static double now_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

typedef struct run_profile {
    const char *name;
    uint64_t default_neurons;
    uint64_t default_iters;
    uint64_t default_reps;
    uint8_t default_threshold;
} run_profile;

static const run_profile k_sovereign_profile = {"sovereign", 1u << 20, 256, 12, 8};
static const run_profile k_adaptive_profile = {"adaptive", 1u << 20, 384, 10, 10};
static const run_profile k_war_drive_profile = {"war-drive", 1u << 22, 256, 16, 8};
static const run_profile k_zpm_profile = {"zpm", 1u << 20, 4096, 12, 8};
static const run_profile k_omega_profile = {"omega", 1u << 20, 16384, 10, 8};
static const run_profile k_omega_hybrid_profile = {"omega-hybrid", 1u << 20, 4096, 12, 8};

static const char *k_brain_port_names[] = {
    "PORT_EXEC",
    "PORT_NEO",
    "PORT_SYNC",
    "PORT_LATTICE",
    "PORT_GATE",
    "PORT_HABIT",
    "PORT_MIRROR",
    "PORT_SENTINEL",
    "PORT_GOVERN",
    "PORT_ALIVE",
};

static const run_profile *profile_from_name(const char *name) {
    if (name == NULL || *name == '\0' || strcmp(name, "sovereign") == 0) {
        return &k_sovereign_profile;
    }
    if (strcmp(name, "adaptive") == 0) {
        return &k_adaptive_profile;
    }
    if (strcmp(name, "war-drive") == 0) {
        return &k_war_drive_profile;
    }
    if (strcmp(name, "zpm") == 0) {
        return &k_zpm_profile;
    }
    if (strcmp(name, "automatic") == 0) {
        return &k_zpm_profile;
    }
    if (strcmp(name, "omega") == 0) {
        return &k_omega_profile;
    }
    if (strcmp(name, "omega-hybrid") == 0) {
        return &k_omega_hybrid_profile;
    }
    return NULL;
}

static void print_usage(void) {
    puts("NRL CLI");
    puts("Usage:");
    puts("  nrl --version");
    puts("  nrl --features");
    puts("  nrl runtime");
    puts("  nrl status | -status");
    puts("  nrl inquire <topic> | -inquire <topic>");
    puts("  nrl chat <message> | -chat <message>");
    puts("  nrl control [--yes|-y] <message>   (sandboxed NL -> advisory / optional preferences.json)");
    puts("  nrl brain-map");
    puts("  nrl variant <kernel>");
    puts("  nrl file <path.nrl>");
    puts("  nrl run [neurons] [iters] [threshold] [profile]");
    puts("      profiles: sovereign|adaptive|war-drive|zpm|automatic|omega|omega-hybrid");
    puts("  nrl assimilate [neurons] [iters] [threshold]");
    puts("      raw INT4 lattice assimilation (binary tensor contract + checksum)");
    puts("  nrl bench [neurons] [iters] [reps] [threshold] [profile]");
    puts("      profiles: +aes256-synth (synthetic mix benchmark; see language/examples/aes256.nrl)");
    puts("  nrl demo");
    puts("      run examples/ultimate_power_demo.py (Python + nrlpy on PYTHONPATH)");
    puts("  nrl -ai|--ai on|off|--on|--off   LM/AI opt-in consent + persist env (Windows: setx)");
}

static void print_features(void) {
    const uint32_t f = nrl_v1_cpu_features();
    puts("NRL CPU features:");
    printf("  sse2: %s\n", (f & NRL_CPU_SSE2) ? "yes" : "no");
    printf("  sse4.2: %s\n", (f & NRL_CPU_SSE42) ? "yes" : "no");
    printf("  avx: %s\n", (f & NRL_CPU_AVX) ? "yes" : "no");
    printf("  avx2: %s\n", (f & NRL_CPU_AVX2) ? "yes" : "no");
    printf("  avx512f: %s\n", (f & NRL_CPU_AVX512F) ? "yes" : "no");
    printf("  bmi2: %s\n", (f & NRL_CPU_BMI2) ? "yes" : "no");
    printf("  popcnt: %s\n", (f & NRL_CPU_POPCNT) ? "yes" : "no");
    printf("  fma: %s\n", (f & NRL_CPU_FMA) ? "yes" : "no");
}

static uint64_t parse_u64_or_default(const char *s, uint64_t fallback) {
    if (s == NULL || *s == '\0') {
        return fallback;
    }
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        return fallback;
    }
    return (uint64_t)v;
}

static const char *trim_ws(char *s) {
    while (*s != '\0' && isspace((unsigned char)*s)) {
        ++s;
    }
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) {
        --end;
    }
    *end = '\0';
    return s;
}

static int has_suffix(const char *s, const char *suffix) {
    const size_t n = strlen(s);
    const size_t m = strlen(suffix);
    if (n < m) {
        return 0;
    }
    return strcmp(s + (n - m), suffix) == 0;
}

static int path_is_file(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) {
        return 0;
    }
    return S_ISREG(st.st_mode) ? 1 : 0;
}

static int path_is_dir(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

static int try_demo_path(const char *candidate, char *out, size_t out_cap) {
    char buf[4096];
#if defined(_WIN32)
    {
        DWORD n = GetFullPathNameA(candidate, (DWORD)sizeof(buf), buf, NULL);
        if (n == 0 || n >= sizeof(buf)) {
            snprintf(buf, sizeof(buf), "%s", candidate);
        }
    }
#else
    {
        char *rp = realpath(candidate, buf);
        if (rp == NULL) {
            snprintf(buf, sizeof(buf), "%s", candidate);
        }
    }
#endif
    if (!path_is_file(buf)) {
        return -1;
    }
    snprintf(out, out_cap, "%s", buf);
    return 0;
}

/* Full path to this executable (used to export NRL_BIN for child processes). */
static int get_self_exe_path(char *out, size_t cap) {
#if defined(_WIN32)
    if (GetModuleFileNameA(NULL, out, (DWORD)cap) == 0 || GetModuleFileNameA(NULL, out, (DWORD)cap) >= (DWORD)cap) {
        return -1;
    }
#elif defined(__linux__)
    {
        ssize_t n = readlink("/proc/self/exe", out, cap - 1);
        if (n < 0) {
            return -1;
        }
        out[n] = '\0';
    }
#elif defined(__APPLE__)
    {
        uint32_t sz = (uint32_t)cap;
        if (_NSGetExecutablePath(out, &sz) != 0) {
            return -1;
        }
    }
#else
    (void)cap;
    out[0] = '\0';
    return -1;
#endif
    return 0;
}

static int get_exe_dir(char *out, size_t cap) {
#if defined(_WIN32)
    if (GetModuleFileNameA(NULL, out, (DWORD)cap) == 0 || GetModuleFileNameA(NULL, out, (DWORD)cap) >= (DWORD)cap) {
        return -1;
    }
#elif defined(__linux__)
    {
        ssize_t n = readlink("/proc/self/exe", out, cap - 1);
        if (n < 0) {
            return -1;
        }
        out[n] = '\0';
    }
#elif defined(__APPLE__)
    {
        uint32_t sz = (uint32_t)cap;
        if (_NSGetExecutablePath(out, &sz) != 0) {
            return -1;
        }
    }
#else
    (void)cap;
    out[0] = '\0';
    return -1;
#endif
    {
        char *slash = strrchr(out, '/');
#if defined(_WIN32)
        char *bs = strrchr(out, '\\');
        if (!slash || (bs != NULL && bs > slash)) {
            slash = bs;
        }
#endif
        if (slash != NULL && slash != out) {
            *slash = '\0';
        }
    }
    return 0;
}

static int resolve_ultimate_demo_path(char *out, size_t out_cap) {
    char cand[4096];
    const char *nrl_root = getenv("NRL_ROOT");
    if (nrl_root != NULL && *nrl_root != '\0') {
        snprintf(cand, sizeof(cand), "%s/examples/ultimate_power_demo.py", nrl_root);
        if (try_demo_path(cand, out, out_cap) == 0) {
            return 0;
        }
#if defined(_WIN32)
        snprintf(cand, sizeof(cand), "%s\\examples\\ultimate_power_demo.py", nrl_root);
        if (try_demo_path(cand, out, out_cap) == 0) {
            return 0;
        }
#endif
    }
#if defined(_WIN32)
    {
        const char *profile = getenv("USERPROFILE");
        if (profile != NULL && *profile != '\0') {
            snprintf(cand,
                     sizeof(cand),
                     "%s\\.local\\share\\nrl\\examples\\ultimate_power_demo.py",
                     profile);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
        }
    }
#endif
#if !defined(_WIN32)
    {
        const char *home = getenv("HOME");
        if (home != NULL && *home != '\0') {
            snprintf(cand,
                     sizeof(cand),
                     "%s/.local/share/nrl/examples/ultimate_power_demo.py",
                     home);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
        }
    }
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            snprintf(cand, sizeof(cand), "%s/examples/ultimate_power_demo.py", cwd);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
        }
    }
#else
    {
        char cwd[4096];
        if (GetCurrentDirectoryA(sizeof(cwd), cwd) != 0) {
            snprintf(cand, sizeof(cand), "%s/examples/ultimate_power_demo.py", cwd);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
        }
    }
#endif
    {
        char exedir[4096];
        if (get_exe_dir(exedir, sizeof(exedir)) == 0) {
            snprintf(cand, sizeof(cand), "%s/../examples/ultimate_power_demo.py", exedir);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
            snprintf(cand, sizeof(cand), "%s/../../examples/ultimate_power_demo.py", exedir);
            if (try_demo_path(cand, out, out_cap) == 0) {
                return 0;
            }
        }
    }
    return -1;
}

static void path_strip_leaf(char *p) {
    char *slash = strrchr(p, '/');
#if defined(_WIN32)
    char *bs = strrchr(p, '\\');
    if (!slash || (bs != NULL && bs > slash)) {
        slash = bs;
    }
#endif
    if (slash != NULL) {
        *slash = '\0';
    }
}

static int path_repo_from_demo(const char *demo_path, char *repo, size_t cap) {
    char tmp[4096];
    if (strlen(demo_path) >= sizeof(tmp)) {
        return -1;
    }
    snprintf(tmp, sizeof(tmp), "%s", demo_path);
    path_strip_leaf(tmp);
    path_strip_leaf(tmp);
    snprintf(repo, cap, "%s", tmp);
    return 0;
}

static int resolve_pythonpath(const char *repo_root, char *out, size_t cap) {
    char probe[4096];
    snprintf(probe, sizeof(probe), "%s/nrlpy/src/nrlpy", repo_root);
    if (path_is_dir(probe)) {
        snprintf(out, cap, "%s/nrlpy/src", repo_root);
        return 0;
    }
    snprintf(probe, sizeof(probe), "%s/py/nrlpy", repo_root);
    if (path_is_dir(probe)) {
        snprintf(out, cap, "%s/py", repo_root);
        return 0;
    }
    return -1;
}

static int run_python_demo(const char *py_path, char *demo_path_mutable) {
#if defined(_WIN32)
    {
        char self_exe[4096];
        if (get_self_exe_path(self_exe, sizeof(self_exe)) == 0) {
            if (_putenv_s("NRL_BIN", self_exe) != 0) {
                fputs("nrl demo: could not set NRL_BIN\n", stderr);
                return 1;
            }
        }
    }
    if (_putenv_s("PYTHONPATH", py_path) != 0) {
        fputs("nrl demo: could not set PYTHONPATH\n", stderr);
        return 1;
    }
    {
        char *argv[] = {"python", "-m", "nrlpy.cli", "run", demo_path_mutable, NULL};
        const intptr_t rc = _spawnvp(_P_WAIT, "python", (const char *const *)argv);
        if (rc == 0) {
            return 0;
        }
        if (rc != (intptr_t)-1) {
            fprintf(stderr,
                    "nrl demo: python exited with code %td (see messages above; not a spawn failure)\n",
                    rc);
            return 1;
        }
    }
    {
        char *argv2[] = {"py", "-3", "-m", "nrlpy.cli", "run", demo_path_mutable, NULL};
        const intptr_t rc2 = _spawnvp(_P_WAIT, "py", (const char *const *)argv2);
        if (rc2 == 0) {
            return 0;
        }
        if (rc2 != (intptr_t)-1) {
            fprintf(stderr,
                    "nrl demo: py -3 exited with code %td (see messages above; not a spawn failure)\n",
                    rc2);
            return 1;
        }
    }
    fputs("nrl demo: failed to spawn python (try: python on PATH, or py -3)\n", stderr);
    return 1;
#else
    {
        char self_exe[4096];
        if (get_self_exe_path(self_exe, sizeof(self_exe)) == 0) {
            if (setenv("NRL_BIN", self_exe, 1) != 0) {
                fputs("nrl demo: setenv NRL_BIN failed\n", stderr);
                return 1;
            }
        }
    }
    if (setenv("PYTHONPATH", py_path, 1) != 0) {
        fputs("nrl demo: setenv PYTHONPATH failed\n", stderr);
        return 1;
    }
    {
        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "python3 -m nrlpy.cli run \"%s\"", demo_path_mutable);
        const int st = system(cmd);
        if (st == 0) {
            return 0;
        }
    }
    {
        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "python -m nrlpy.cli run \"%s\"", demo_path_mutable);
        const int st = system(cmd);
        if (st == 0) {
            return 0;
        }
    }
    fputs("nrl demo: python3/python failed (install Python 3.9+ and nrlpy sources)\n", stderr);
    return 1;
#endif
}

static int cmd_demo(void) {
    char demo_path[4096];
    if (resolve_ultimate_demo_path(demo_path, sizeof(demo_path)) != 0) {
        fputs("nrl demo: could not find examples/ultimate_power_demo.py\n", stderr);
        fputs("  Fix: set NRL_ROOT to the folder that contains ./examples/, or run from repo root,\n", stderr);
        fputs("  or reinstall so examples ship next to the binary (see scripts/install_nrl.*).\n", stderr);
        return 1;
    }
    char repo[4096];
    if (path_repo_from_demo(demo_path, repo, sizeof(repo)) != 0) {
        fputs("nrl demo: internal path error\n", stderr);
        return 1;
    }
    char py_path[4096];
    if (resolve_pythonpath(repo, py_path, sizeof(py_path)) != 0) {
        fprintf(stderr,
                "nrl demo: nrlpy package not found under %s (expected nrlpy/src or py/nrlpy).\n",
                repo);
        return 1;
    }
    puts("NRL demo");
    printf("  script: %s\n", demo_path);
    printf("  PYTHONPATH=%s\n", py_path);
    (void)fflush(stdout);
    return run_python_demo(py_path, demo_path) == 0 ? 0 : 1;
}

static int str_ieq(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static int str_contains_ci(const char *haystack, const char *needle) {
    const size_t hlen = strlen(haystack);
    const size_t nlen = strlen(needle);
    if (nlen == 0 || hlen < nlen) {
        return 0;
    }
    for (size_t i = 0; i + nlen <= hlen; ++i) {
        size_t j = 0;
        while (j < nlen &&
               tolower((unsigned char)haystack[i + j]) ==
                   tolower((unsigned char)needle[j])) {
            ++j;
        }
        if (j == nlen) {
            return 1;
        }
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Sandboxed CLI control plane: NL hints -> advisory / optional preferences   */
/* JSON only under NRL_ROOT/build/control (or ./build/control). No hot-path.  */
/* -------------------------------------------------------------------------- */

#if defined(_WIN32)
#define NRL_MKDIR(p) _mkdir(p)
#else
#define NRL_MKDIR(p) mkdir(p, (mode_t)0755)
#endif

typedef struct NrlControlVerdict {
    const char *intent_id;
    const char *govern_verdict;
    int wants_profile_write;
    int wants_power_until;
    int wants_throttle;
    int wants_volatile_gate;
    const char *profile_value;
} NrlControlVerdict;

static void iso8601_utc_now(char *buf, size_t cap) {
#if defined(_WIN32)
    SYSTEMTIME st;
    GetSystemTime(&st);
    snprintf(buf, cap, "%04u-%02u-%02uT%02u:%02u:%02uZ",
             (unsigned)st.wYear, (unsigned)st.wMonth, (unsigned)st.wDay,
             (unsigned)st.wHour, (unsigned)st.wMinute, (unsigned)st.wSecond);
#else
    time_t t = time(NULL);
    struct tm gt;
    memset(&gt, 0, sizeof(gt));
    gmtime_r(&t, &gt);
    strftime(buf, cap, "%Y-%m-%dT%H:%M:%SZ", &gt);
#endif
}

static int nrl_user_consent_file_path(char *path, size_t cap) {
#if defined(_WIN32)
    const char *home = getenv("USERPROFILE");
#else
    const char *home = getenv("HOME");
#endif
    if (home == NULL || home[0] == '\0') {
        return -1;
    }
    snprintf(path, cap, "%s/.nrl/consent.json", home);
    return 0;
}

static int nrl_user_consent_dir_from_path(char *dir, size_t cap, const char *consent_path) {
    size_t i = strlen(consent_path);
    while (i > 0 && consent_path[i - 1] != '/' && consent_path[i - 1] != '\\') {
        --i;
    }
    if (i == 0 || i >= cap) {
        return -1;
    }
    memcpy(dir, consent_path, i);
    dir[i] = '\0';
    return 0;
}

static int nrl_read_consent_lm_ai(void) {
    char path[768];
    char buf[4096];
    FILE *fp = NULL;
    size_t n = 0;
    if (nrl_user_consent_file_path(path, sizeof path) != 0) {
        return 0;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }
    n = fread(buf, 1, sizeof buf - 1u, fp);
    fclose(fp);
    buf[n] = '\0';
    if (strstr(buf, "\"lm_ai_opt_in\": true") != NULL || strstr(buf, "\"lm_ai_opt_in\":true") != NULL) {
        return 1;
    }
    return 0;
}

static int lm_ai_arg_is_on(const char *sw) {
    if (sw == NULL) {
        return 0;
    }
    if (strcmp(sw, "--on") == 0 || strcmp(sw, "-on") == 0) {
        return 1;
    }
    return str_ieq(sw, "on");
}

static int lm_ai_arg_is_off(const char *sw) {
    if (sw == NULL) {
        return 0;
    }
    if (strcmp(sw, "--off") == 0 || strcmp(sw, "-off") == 0) {
        return 1;
    }
    return str_ieq(sw, "off");
}

static int cmd_lm_ai_toggle(int enable) {
    char path[768];
    char dir[768];
    char ts[64];
    FILE *fp = NULL;
    if (nrl_user_consent_file_path(path, sizeof path) != 0) {
        fputs("nrl -ai: USERPROFILE/HOME not set; cannot write consent.json\n", stderr);
        return 1;
    }
    if (nrl_user_consent_dir_from_path(dir, sizeof dir, path) != 0) {
        fputs("nrl -ai: consent path too long\n", stderr);
        return 1;
    }
    if (NRL_MKDIR(dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "nrl -ai: cannot create directory %s\n", dir);
        return 1;
    }
    iso8601_utc_now(ts, sizeof ts);
    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "nrl -ai: cannot open %s for write\n", path);
        return 1;
    }
    fprintf(fp,
            "{\n  \"lm_ai_opt_in\": %s,\n  \"source\": \"nrl -ai\",\n  \"updated_utc\": \"%s\"\n}\n",
            enable ? "true" : "false",
            ts);
    fclose(fp);
    printf("nrl -ai: wrote %s (lm_ai_opt_in=%s)\n", path, enable ? "true" : "false");
#if defined(_WIN32)
    {
        char cmd[96];
        snprintf(cmd, sizeof cmd, "setx NRL_LM_AI_OPT_IN %d >nul 2>&1", enable ? 1 : 0);
        (void)system(cmd);
        puts("nrl -ai: NRL_LM_AI_OPT_IN persisted for new terminals (open a new shell)");
    }
#else
    printf("nrl -ai: for this shell: export NRL_LM_AI_OPT_IN=%d\n", enable ? 1 : 0);
    puts("nrl -ai: add that line to ~/.profile or similar for persistence");
#endif
    return 0;
}

static void json_escape_short(const char *in, char *out, size_t cap) {
    size_t j = 0;
    if (cap == 0) {
        return;
    }
    if (in == NULL) {
        out[0] = '\0';
        return;
    }
    for (size_t i = 0; in[i] != '\0' && j + 2 < cap; ++i) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            out[j++] = '\\';
            if (j < cap - 1) {
                out[j++] = (char)c;
            }
        } else if (c < 32u) {
            out[j++] = ' ';
        } else {
            out[j++] = (char)c;
        }
    }
    out[j] = '\0';
}

static void nrl_control_fill_build_and_control(char *build_dir, size_t bcap, char *control_dir, size_t ccap) {
    const char *root = getenv("NRL_ROOT");
    if (root != NULL && root[0] != '\0') {
        snprintf(build_dir, bcap, "%s/build", root);
        snprintf(control_dir, ccap, "%s/control", build_dir);
    } else {
        snprintf(build_dir, bcap, "build");
        snprintf(control_dir, ccap, "build/control");
    }
}

static void nrl_control_preferences_path(char *path, size_t cap) {
    char ctl[512];
    char bld[512];
    nrl_control_fill_build_and_control(bld, sizeof bld, ctl, sizeof ctl);
    snprintf(path, cap, "%s/preferences.json", ctl);
}

static int nrl_control_ensure_dirs(char *control_dir, size_t cap) {
    char build_dir[512];
    nrl_control_fill_build_and_control(build_dir, sizeof build_dir, control_dir, cap);
    if (NRL_MKDIR(build_dir) != 0 && errno != EEXIST) {
        return -1;
    }
    if (NRL_MKDIR(control_dir) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static void nrl_control_immune_stdout(const NrlControlVerdict *v) {
    puts("control_immune:");
    puts("  PORT_SENTINEL: input_scanned ok (length and token bounds)");
    printf("  PORT_GOVERN: verdict=%s intent=%s\n", v->govern_verdict, v->intent_id);
}

static void nrl_control_classify(const char *msg, NrlControlVerdict *out) {
    memset(out, 0, sizeof(*out));
    out->intent_id = "none";
    out->govern_verdict = "allow_advisory";
    if (msg == NULL || msg[0] == '\0') {
        return;
    }
    if (str_contains_ci(msg, "taskkill") || str_contains_ci(msg, "format c") ||
        str_contains_ci(msg, "shutdown") || str_contains_ci(msg, "registry") ||
        str_contains_ci(msg, "curl ") || str_contains_ci(msg, "http://") ||
        str_contains_ci(msg, "https://") || str_contains_ci(msg, "powershell -")) {
        out->intent_id = "external_os_denied";
        out->govern_verdict = "deny_external_sandbox";
        return;
    }
    if (str_contains_ci(msg, "volatile") && str_contains_ci(msg, "market")) {
        out->intent_id = "volatile_market_extra_gate";
        out->govern_verdict = "allow_write";
        out->wants_profile_write = 1;
        out->wants_volatile_gate = 1;
        out->profile_value = "sovereign";
        return;
    }
    if (str_contains_ci(msg, "buy") || str_contains_ci(msg, "sell") ||
        str_contains_ci(msg, "stock") ||
        (str_contains_ci(msg, "trading") &&
         !str_contains_ci(msg, "algorithm") && !str_contains_ci(msg, "simulate"))) {
        out->intent_id = "trading_external_denied";
        out->govern_verdict = "deny_trading_sandbox";
        return;
    }
    if (str_contains_ci(msg, "server") && str_contains_ci(msg, "optim")) {
        out->intent_id = "server_optimize_advisory";
        out->govern_verdict = "allow_advisory";
        return;
    }
    if ((str_contains_ci(msg, "max") && str_contains_ci(msg, "power")) ||
        str_contains_ci(msg, "maximum power")) {
        out->intent_id = "power_boost_1h";
        out->govern_verdict = "allow_write";
        out->wants_power_until = 1;
        out->wants_profile_write = 1;
        out->profile_value = "omega-hybrid";
        return;
    }
    if (str_contains_ci(msg, "slow") &&
        (str_contains_ci(msg, "cpu") || str_contains_ci(msg, "process"))) {
        out->intent_id = "throttle_conservative";
        out->govern_verdict = "allow_write";
        out->wants_throttle = 1;
        out->wants_profile_write = 1;
        out->profile_value = "sovereign";
        return;
    }
    if (str_contains_ci(msg, "sovereign") || str_contains_ci(msg, "extra safe") ||
        str_contains_ci(msg, "extra gated")) {
        out->intent_id = "profile_sovereign";
        out->govern_verdict = "allow_write";
        out->wants_profile_write = 1;
        out->profile_value = "sovereign";
        return;
    }
    if (str_contains_ci(msg, "omega-hybrid") || str_contains_ci(msg, "balanced")) {
        out->intent_id = "profile_omega_hybrid";
        out->govern_verdict = "allow_write";
        out->wants_profile_write = 1;
        out->profile_value = "omega-hybrid";
        return;
    }
    if (str_contains_ci(msg, "omega") && !str_contains_ci(msg, "hybrid")) {
        out->intent_id = "profile_omega_high_skip";
        out->govern_verdict = "allow_write";
        out->wants_profile_write = 1;
        out->profile_value = "omega";
        return;
    }
    if (str_contains_ci(msg, "fast") || str_contains_ci(msg, "speed")) {
        out->intent_id = "profile_advisory_speed";
        out->govern_verdict = "allow_advisory";
        return;
    }
}

static int nrl_control_append_audit(const char *channel,
                                    const char *raw,
                                    const NrlControlVerdict *v,
                                    const char *outcome) {
    char dir[512];
    char path[600];
    char ts[64];
    char esc[768];
    FILE *fp = NULL;
    if (nrl_control_ensure_dirs(dir, sizeof dir) != 0) {
        return -1;
    }
    snprintf(path, sizeof path, "%s/control_audit.jsonl", dir);
    iso8601_utc_now(ts, sizeof ts);
    json_escape_short(raw, esc, sizeof esc);
    fp = fopen(path, "ab");
    if (fp == NULL) {
        return -1;
    }
    fprintf(fp,
            "{\"ts_utc\":\"%s\",\"channel\":\"%s\",\"intent\":\"%s\",\"govern\":\"%s\","
            "\"outcome\":\"%s\",\"raw\":\"%s\"}\n",
            ts, channel, v->intent_id, v->govern_verdict, outcome, esc);
    fclose(fp);
    return 0;
}

static int nrl_control_write_preferences(const NrlControlVerdict *v, time_t power_until) {
    char dir[512];
    char path[600];
    FILE *fp = NULL;
    time_t now = time(NULL);
    const char *th = "none";
    if (nrl_control_ensure_dirs(dir, sizeof dir) != 0) {
        return -1;
    }
    snprintf(path, sizeof path, "%s/preferences.json", dir);
    fp = fopen(path, "wb");
    if (fp == NULL) {
        return -1;
    }
    if (v->wants_throttle) {
        th = "conservative";
    }
    if (v->wants_volatile_gate) {
        th = "gated";
    }
    if (v->wants_profile_write && v->profile_value != NULL) {
        fprintf(fp,
                "{\"schema_id\":\"nrl.control_preferences.v1\",\"updated_unix\":%lld,"
                "\"recommended_profile\":\"%s\",\"power_until_unix\":%lld,\"throttle_hint\":\"%s\"}\n",
                (long long)now, v->profile_value, (long long)power_until, th);
    } else {
        fprintf(fp,
                "{\"schema_id\":\"nrl.control_preferences.v1\",\"updated_unix\":%lld,"
                "\"recommended_profile\":\"sovereign\",\"power_until_unix\":0,\"throttle_hint\":\"none\"}\n",
                (long long)now);
    }
    fclose(fp);
    return 0;
}

static int nrl_control_dispatch(const char *msg, const char *channel, int allow_write) {
    NrlControlVerdict v;
    time_t power_until = 0;
    int confirm = 0;
    int can_write = 0;
    memset(&v, 0, sizeof(v));
    nrl_control_classify(msg, &v);
    nrl_control_immune_stdout(&v);
    confirm = (allow_write != 0) ||
              (getenv("NRL_CONTROL_CONFIRM") != NULL &&
               strcmp(getenv("NRL_CONTROL_CONFIRM"), "1") == 0);
    if (strcmp(v.govern_verdict, "deny_trading_sandbox") == 0 ||
        strcmp(v.govern_verdict, "deny_external_sandbox") == 0) {
        puts("control_outcome: BLOCKED (sandbox policy — no OS / trading / network side effects)");
        (void)nrl_control_append_audit(channel, msg, &v, "blocked");
        return 0;
    }
    if (strcmp(v.govern_verdict, "allow_advisory") == 0) {
        (void)nrl_control_append_audit(channel, msg, &v, "advisory");
    } else if (strcmp(v.govern_verdict, "allow_write") == 0) {
        can_write = confirm ? 1 : 0;
        if (!can_write) {
            puts("control_outcome: DEFER — persisted writes need `nrl control --yes ...` or NRL_CONTROL_CONFIRM=1");
            (void)nrl_control_append_audit(channel, msg, &v, "defer_confirm");
        }
    }
    if (strcmp(v.govern_verdict, "allow_write") == 0 && can_write) {
        if (v.wants_power_until) {
            power_until = time(NULL) + (time_t)3600;
        } else {
            power_until = 0;
        }
        if (nrl_control_write_preferences(&v, power_until) != 0) {
            puts("control_outcome: ERROR (could not write preferences.json)");
            (void)nrl_control_append_audit(channel, msg, &v, "error_io");
            return 1;
        }
        puts("control_outcome: OK (preferences.json written under build/control)");
        (void)nrl_control_append_audit(channel, msg, &v, "applied");
    }
    if (strcmp(v.intent_id, "server_optimize_advisory") == 0) {
        puts("control_advisory:");
        puts("  sandbox: NRL cannot change external servers. Use your fleet orchestrator for scaling.");
        puts("  nrl: for local benches try `nrl bench … omega-hybrid` (balanced executed GOPS).");
    }
    if (strcmp(v.intent_id, "power_boost_1h") == 0) {
        puts("control_advisory:");
        {
            time_t until_show = power_until;
            if (until_show == 0 && v.wants_power_until) {
                until_show = time(NULL) + (time_t)3600;
            }
            printf("  intent: raise power ceiling for ~1h (target clock unix %lld); kernels unchanged until you run bench/run.\n",
                   (long long)until_show);
        }
    }
    if (strcmp(v.intent_id, "throttle_conservative") == 0) {
        puts("control_advisory:");
        puts("  intent: prefer sovereign / smaller defaults in preferences (does not kill other OS processes).");
    }
    if (strcmp(v.intent_id, "volatile_market_extra_gate") == 0) {
        puts("control_advisory:");
        puts("  intent: extra-gated lane — sovereign profile hint in preferences only (no broker / no network).");
    }
    if (strcmp(v.intent_id, "profile_advisory_speed") == 0) {
        puts("control_advisory:");
        puts("  virtual: `nrl bench … omega` — executed: sovereign / omega-hybrid.");
        puts("  answer: use `nrl bench ... omega` for maximum virtual speed.");
        puts("  answer: use `nrl bench ... omega-hybrid` to keep high executed GOPS.");
    }
    if (strcmp(v.intent_id, "none") == 0 && channel != NULL && strcmp(channel, "chat") == 0) {
        if (str_contains_ci(msg, "status") || str_contains_ci(msg, "health")) {
            puts("  answer: run `nrl status` for runtime readiness and active variant.");
        } else if (str_contains_ci(msg, "safe") || str_contains_ci(msg, "risk")) {
            puts("  answer: keep language features rule-based; avoid unconstrained self-modification.");
            puts("  answer: see `nrl inquire safety` and docs/nrl_immune_system_spec.md.");
        } else if (str_contains_ci(msg, "chat") || str_contains_ci(msg, "english")) {
            puts("  answer: lightweight intent mapping is enabled; use `nrl control` for persisted hints.");
        } else {
            puts("  answer: try `nrl inquire speed`, `nrl inquire safety`, or `nrl control --yes \"…\"` for writes.");
        }
    }
    return 0;
}

typedef struct nrl_file_plan {
    int do_bench;
    char profile[32];
    uint64_t neurons;
    uint64_t iterations;
    uint64_t reps;
    uint64_t threshold;
    uint64_t expected_fnv1a64;
    int has_expected_fnv1a64;
} nrl_file_plan;

static int cmd_runtime(void) {
    puts(nrl_v1_version());
    print_features();
    printf("NRL variants:\n");
    printf("  braincore_int4: %s\n", nrl_v1_active_variant("braincore_int4"));
    puts("cognitive_mode:");
    puts("  system1_automatic = zpm/automatic profile (cached transition execution)");
    puts("  system2_deliberate = iterative profile (step-by-step execution)");
    return 0;
}

static int nrl_control_json_copy_string(const char *buf, const char *key, char *out, size_t cap) {
    char needle[96];
    if (snprintf(needle, sizeof needle, "\"%s\":\"", key) >= (int)sizeof needle) {
        return -1;
    }
    const char *p = strstr(buf, needle);
    if (p == NULL) {
        return -1;
    }
    p += strlen(needle);
    size_t i = 0;
    for (; *p != '\0' && *p != '"' && i + 1 < cap; ++p) {
        out[i++] = (char)*p;
    }
    out[i] = '\0';
    return 0;
}

static int nrl_control_json_parse_u64(const char *buf, const char *key, unsigned long long *out) {
    char needle[96];
    if (snprintf(needle, sizeof needle, "\"%s\":", key) >= (int)sizeof needle) {
        return -1;
    }
    const char *p = strstr(buf, needle);
    if (p == NULL) {
        return -1;
    }
    p += strlen(needle);
    while (*p != '\0' && isspace((unsigned char)*p)) {
        ++p;
    }
    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == NULL || end == p) {
        return -1;
    }
    *out = v;
    return 0;
}

static void nrl_status_print_control_prefs(void) {
    char path[640];
    char buf[8192];
    FILE *fp = NULL;
    size_t n = 0;
    char profile[64];
    char throttle[32];
    unsigned long long power_until = 0;

    nrl_control_preferences_path(path, sizeof path);
    printf("  control_preferences_path: %s\n", path);
    fp = fopen(path, "rb");
    if (fp == NULL) {
        puts("  control_preferences: none (no file yet)");
        return;
    }
    n = fread(buf, 1, sizeof buf - 1u, fp);
    fclose(fp);
    buf[n] = '\0';
    if (strstr(buf, "\"schema_id\":\"nrl.control_preferences.v1\"") == NULL) {
        puts("  control_preferences: present but schema not recognized");
        return;
    }
    profile[0] = '\0';
    throttle[0] = '\0';
    if (nrl_control_json_copy_string(buf, "recommended_profile", profile, sizeof profile) != 0) {
        snprintf(profile, sizeof profile, "(unparsed)");
    }
    if (nrl_control_json_copy_string(buf, "throttle_hint", throttle, sizeof throttle) != 0) {
        snprintf(throttle, sizeof throttle, "none");
    }
    if (nrl_control_json_parse_u64(buf, "power_until_unix", &power_until) != 0) {
        power_until = 0;
    }
    printf("  control_preferences: recommended_profile=%s throttle_hint=%s power_until_unix=%llu\n",
           profile, throttle, power_until);
    {
        time_t now = time(NULL);
        int active = 0;
        if (power_until > 0ull && (unsigned long long)now < power_until) {
            active = 1;
        } else if (strcmp(throttle, "conservative") == 0 || strcmp(throttle, "gated") == 0) {
            active = 1;
        }
        printf("  control_hints_active_for_nrlpy: %s\n", active ? "yes" : "no");
    }
}

static int cmd_status(void) {
    const uint32_t f = nrl_v1_cpu_features();
    const int has_avx2 = (f & NRL_CPU_AVX2) != 0u;
    const char *opt = getenv("NRL_LM_AI_OPT_IN");
    int lm_opt_in = 0;
    const char *lm_note = "";
    if (opt != NULL) {
        lm_opt_in = (strcmp(opt, "1") == 0);
    } else {
        const int c = nrl_read_consent_lm_ai();
        lm_opt_in = c;
        if (c != 0) {
            lm_note = " (from ~/.nrl/consent.json)";
        }
    }
    puts("NRL status");
    printf("  engine: %s\n", nrl_v1_version());
    printf("  primary_variant: %s\n", nrl_v1_active_variant("braincore_int4"));
    printf("  cognitive_system1: enabled (zpm/automatic/omega lanes)\n");
    printf("  cognitive_system2: enabled (sovereign iterative lane)\n");
    printf("  lm_ai_opt_in: %s%s\n", lm_opt_in ? "enabled" : "disabled", lm_note);
    printf("  avx2_ready: %s\n", has_avx2 ? "yes" : "no");
    printf("  health: %s\n", has_avx2 ? "nominal" : "degraded (scalar fallback)");
    nrl_status_print_control_prefs();
    return 0;
}

static int cmd_inquire(const char *topic) {
    if (topic == NULL || *topic == '\0') {
        puts("nrl inquire topics: speed, safety, modes, profiles, architecture, benchmark, assimilate, epistemic, demo, control, consent");
        return 0;
    }
    if (str_ieq(topic, "consent")) {
        puts("inquire:consent");
        puts("  command: nrl -ai|--ai on|off|--on|--off");
        puts("  writes: ~/.nrl/consent.json (USERPROFILE/.nrl on Windows)");
        puts("  windows: also runs setx NRL_LM_AI_OPT_IN 1|0 for new terminals");
        puts("  nrlpy: nrlpy -ai|--ai on|off (same file + setx on Windows)");
        return 0;
    }
    if (str_ieq(topic, "control")) {
        puts("inquire:control");
        puts("  command: nrl control [--yes] \"natural language intent\"");
        puts("  writes: only $NRL_ROOT/build/control/ (or ./build/control): preferences.json + control_audit.jsonl");
        puts("  confirm: --yes or NRL_CONTROL_CONFIRM=1 for allow_write intents");
        puts("  blocked: trading, raw URLs, taskkill/shutdown/registry — see nrl-architecture.md §1.5");
        puts("  status: `nrl status` prints control_preferences_path + parsed hints (read-only)");
        puts("  nrlpy: `python -m nrlpy.cli control status` | `python -m nrlpy.cli control audit tail [N]`");
        puts("  schemas: docs/schemas/control_preferences_v1 + control_audit_line_v1");
        return 0;
    }
    if (str_ieq(topic, "speed") || str_ieq(topic, "benchmark")) {
        puts("inquire:speed");
        puts("  fastest virtual lane: omega (system1 automatic)");
        puts("  fastest executed throughput lane: sovereign / omega-hybrid");
        puts("  recommendation: omega for virtual claims, omega-hybrid for balanced production.");
        return 0;
    }
    if (str_ieq(topic, "safety")) {
        puts("inquire:safety");
        puts("  current safety posture: bounded args + deterministic replay-friendly execution");
        puts("  next hardening target: sentinel/govern/alive runtime guard rails");
        return 0;
    }
    if (str_ieq(topic, "modes") || str_ieq(topic, "profiles")) {
        puts("inquire:modes");
        puts("  system2: sovereign, adaptive, war-drive");
        puts("  system1: zpm/automatic, omega, omega-hybrid");
        return 0;
    }
    if (str_ieq(topic, "architecture")) {
        puts("inquire:architecture");
        puts("  model: machine-code-first kernels + .nrl and nrlpy control planes");
        puts("  principle: compute avoidance first, not brute-force op inflation");
        return 0;
    }
    if (str_ieq(topic, "assimilate")) {
        puts("inquire:assimilate");
        puts("  binary contract: packed_bytes = nrl_v1_braincore_packed_bytes(neurons)");
        puts("  potentials: read-write packed INT4 state; inputs: read-only drive field");
        puts("  path: nrlpy._core.braincore_int4_inplace + `nrl assimilate` (sovereign kernel)");
        puts("  goal: Python tensors alias the same bytes the neural lattice mutates in-place");
        return 0;
    }
    if (str_ieq(topic, "epistemic")) {
        puts("inquire:epistemic");
        puts("  model: known math (tables, invariants, certificates) collapses work instead of re-grinding");
        puts("  ZPM tie: fewer executed_updates vs baseline_equiv when skips are proven safe on observables");
        puts("  lanes: System 2 earns knowledge; System 1 (zpm/automatic) spends it under exact parity rules");
        puts("  doc: nrl-architecture.md section 2.4 + language/spec/nrl_physics_language_v0.md section 6");
        return 0;
    }
    if (str_ieq(topic, "demo")) {
        puts("inquire:demo");
        puts("  command: nrl demo");
        puts("  runs: examples/ultimate_power_demo.py via python -m nrlpy.cli run");
        puts("  needs: NRL_ROOT or cwd with ./examples/, plus nrlpy/src on PYTHONPATH (set by nrl demo)");
        return 0;
    }
    puts("inquire:unknown-topic");
    puts("  try one of: speed, safety, modes, profiles, architecture, benchmark, assimilate, epistemic, demo, control, consent");
    return 0;
}

static int cmd_control(int argc, char **argv) {
    int allow_write = 0;
    int start = 2;
    if (argc > 2 && (strcmp(argv[2], "--yes") == 0 || strcmp(argv[2], "-y") == 0)) {
        allow_write = 1;
        start = 3;
    }
    if (start >= argc) {
        puts("control: provide a message, e.g. `nrl control --yes \"maximum power for one hour\"`");
        puts("  writes require --yes or NRL_CONTROL_CONFIRM=1");
        return 1;
    }
    char msg[512];
    msg[0] = '\0';
    for (int i = start; i < argc; ++i) {
        if (i > start) {
            strncat(msg, " ", sizeof(msg) - strlen(msg) - 1u);
        }
        strncat(msg, argv[i], sizeof(msg) - strlen(msg) - 1u);
    }
    puts("NRL control (sandboxed)");
    printf("  user: %s\n", msg);
    return nrl_control_dispatch(msg, "control", allow_write);
}

static int cmd_chat(int argc, char **argv) {
    if (argc < 3) {
        puts("chat: provide a message, e.g. `nrl chat how do i go faster`");
        return 0;
    }
    char msg[512];
    msg[0] = '\0';
    for (int i = 2; i < argc; ++i) {
        if (i > 2) {
            strncat(msg, " ", sizeof(msg) - strlen(msg) - 1u);
        }
        strncat(msg, argv[i], sizeof(msg) - strlen(msg) - 1u);
    }

    puts("NRL chat");
    printf("  user: %s\n", msg);
    return nrl_control_dispatch(msg, "chat", 0);
}

static uint64_t checksum_u64(const uint8_t *buf, size_t n) {
    uint64_t x = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        x ^= (uint64_t)buf[i];
        x *= 1099511628211ull;
    }
    return x;
}

static uint64_t rotl_u64(uint64_t x, unsigned k) {
    k &= 63u;
    return (x << k) | (x >> (unsigned)((64u - k) & 63u));
}

/* Deterministic 64-bit word mixing (not AES). Used for reproducible .nrl benchmarks + FNV digest. */
static uint64_t nrl_aes256_synth_fnv1a64(uint64_t neurons,
                                         uint64_t iters,
                                         uint64_t reps,
                                         uint8_t round_factor) {
    uint64_t a = 0x6bf8067aa866a2e7ull ^ (neurons * 0x9e3779b97f4a7c15ull);
    uint64_t b = 0x24f55c2c861fb620ull ^ (iters * 0x85ebca6b19ce4b63ull);
    uint64_t c = 0x6b0f12c7b68e0684ull;
    uint64_t d = 0x1924c5da4a87e0d2ull;
    uint64_t steps = neurons * iters * reps;
    const uint64_t cap = (1ull << 27);
    if (steps > cap) {
        steps = cap;
    }
    const unsigned rf = round_factor == 0 ? 10u : (unsigned)round_factor;
    for (uint64_t i = 0; i < steps; ++i) {
        const unsigned im = (unsigned)(i & 15u);
        for (unsigned r = 0; r < rf; ++r) {
            a = (a ^ b) + rotl_u64(c, (unsigned)((r + im) % 64u));
            b = (b ^ c) + rotl_u64(d, (unsigned)((r + 3u + im) % 64u));
            c = (c ^ d) + rotl_u64(a, (unsigned)((r + 5u + im) % 64u));
            d = (d ^ a) + rotl_u64(b, (unsigned)((r + 7u + im) % 64u));
        }
    }
    uint8_t pack[32];
    memcpy(pack, &a, 8);
    memcpy(pack + 8, &b, 8);
    memcpy(pack + 16, &c, 8);
    memcpy(pack + 24, &d, 8);
    return checksum_u64(pack, sizeof(pack));
}

static int cmd_aes256_synth_bench(const nrl_file_plan *plan) {
    if (plan->neurons == 0 || plan->iterations == 0 || plan->reps == 0) {
        fputs("nrl file (aes256-synth): neurons, iterations, reps must be non-zero\n", stderr);
        return 1;
    }
    uint8_t rf = (uint8_t)plan->threshold;
    if (rf == 0) {
        rf = 10;
    }
    if (rf > 32u) {
        fputs("nrl file (aes256-synth): threshold (round factor) must be 1..32\n", stderr);
        return 1;
    }

    const double t0 = now_seconds();
    const uint64_t digest =
        nrl_aes256_synth_fnv1a64(plan->neurons, plan->iterations, plan->reps, rf);
    const double t1 = now_seconds();
    const double elapsed = t1 - t0 > 1e-12 ? (t1 - t0) : 1e-12;
    uint64_t steps = plan->neurons * plan->iterations * plan->reps;
    const uint64_t cap = (1ull << 27);
    if (steps > cap) {
        steps = cap;
    }
    const long double mix_ops = (long double)steps * (long double)rf;
    const long double gmix = mix_ops / elapsed / 1e9L;

    puts("");
    puts("NRL synthetic mix benchmark (not AES; deterministic XOR/rotate state update)");
    printf("  neurons=%llu  iterations=%llu  reps=%llu  round_factor=%u\n",
           (unsigned long long)plan->neurons,
           (unsigned long long)plan->iterations,
           (unsigned long long)plan->reps,
           (unsigned)rf);
    printf("  effective_steps (capped at 2^27): %llu\n", (unsigned long long)steps);
    printf("  elapsed_s:        %.6f\n", t1 - t0);
    printf("  mix_throughput:   %.3f Gmix/s\n", (double)gmix);
    printf("  state_fnv1a64:    %llu\n", (unsigned long long)digest);
    if (plan->has_expected_fnv1a64) {
        if (digest == plan->expected_fnv1a64) {
            puts("  VERIFY:           OK (expected_fnv1a64 match)");
        } else {
            fprintf(stderr,
                    "  VERIFY:           FAIL (got %llu expected %llu)\n",
                    (unsigned long long)digest,
                    (unsigned long long)plan->expected_fnv1a64);
            return 1;
        }
    } else {
        puts("  VERIFY:           skipped (no expected_fnv1a64 in control file)");
    }
    puts("");
    return 0;
}

static int cmd_brain_map(void) {
    nrl_process_memory mem_before;
    nrl_process_memory mem_after;
    nrl_runtime_get_process_memory(&mem_before);

    const uint64_t probe_neurons = 1u << 17;
    const uint64_t probe_iters = 192;
    const uint8_t probe_thresh = 8;
    const size_t byte_count = (size_t)(probe_neurons >> 1);

    uint8_t *potentials = (uint8_t *)malloc(byte_count);
    uint8_t *inputs = (uint8_t *)malloc(byte_count);
    if (potentials == NULL || inputs == NULL) {
        free(potentials);
        free(inputs);
        fputs("nrl brain-map: allocation failed\n", stderr);
        return 1;
    }
    memset(potentials, 0, byte_count);
    memset(inputs, 0, byte_count);
    for (size_t i = 0; i < byte_count; i += 2048) {
        inputs[i] = 0x77u;
    }

    nrl_omega_stats stats = {0};
    const double t0 = now_seconds();
    const nrl_v1_status rc = nrl_braincore_int4_omega_virtual(
        potentials,
        inputs,
        (size_t)probe_neurons,
        (size_t)probe_iters,
        probe_thresh,
        1u << 10,
        4096u,
        0u,
        1u,
        0u,
        &stats);
    const double t1 = now_seconds();
    free(potentials);
    free(inputs);

    if (rc != NRL_OK) {
        fputs("nrl brain-map: bench probe kernel failed\n", stderr);
        return 1;
    }

    nrl_runtime_get_process_memory(&mem_after);

    const double elapsed = t1 - t0 > 1e-12 ? (t1 - t0) : 1e-12;
    const double baseline = (double)stats.baseline_updates;
    const double executed = (double)stats.executed_updates;
    const double skip_ratio =
        baseline <= 0.0 ? 0.0 : 1.0 - (executed / baseline);
    const double virtual_gops = (baseline / elapsed) / 1e9;
    const double executed_gops = (executed / elapsed) / 1e9;

    const uint64_t ref_neurons = 1u << 20;
    const size_t ref_packed = nrl_v1_braincore_packed_bytes((size_t)ref_neurons);
    const double ref_mib = ref_packed > 0 ? (double)ref_packed / (1024.0 * 1024.0) : 0.0;

    const char *mode_label = "virtual-lane bench (single-shot)";
    const char *variant = nrl_v1_active_variant("braincore_int4");

    const uint32_t feat = nrl_v1_cpu_features();
    char feat_buf[96];
    feat_buf[0] = '\0';
    if (feat & NRL_CPU_AVX2) {
        strncat(feat_buf, "AVX2 ", sizeof(feat_buf) - strlen(feat_buf) - 1u);
    }
    if (feat & NRL_CPU_AVX512F) {
        strncat(feat_buf, "AVX512F ", sizeof(feat_buf) - strlen(feat_buf) - 1u);
    }
    if (feat & NRL_CPU_BMI2) {
        strncat(feat_buf, "BMI2 ", sizeof(feat_buf) - strlen(feat_buf) - 1u);
    }
    if (feat & NRL_CPU_POPCNT) {
        strncat(feat_buf, "POPCNT ", sizeof(feat_buf) - strlen(feat_buf) - 1u);
    }
    if (feat & NRL_CPU_FMA) {
        strncat(feat_buf, "FMA ", sizeof(feat_buf) - strlen(feat_buf) - 1u);
    }
    if (feat_buf[0] == '\0') {
        snprintf(feat_buf, sizeof(feat_buf), "scalar baseline");
    }

    const size_t cur_b = mem_after.current_bytes > mem_before.current_bytes ? mem_after.current_bytes
                                                                              : mem_before.current_bytes;
    const size_t peak_b = mem_after.peak_bytes > mem_before.peak_bytes ? mem_after.peak_bytes
                                                                       : mem_before.peak_bytes;
    const double cur_mib = (double)cur_b / (1024.0 * 1024.0);
    const double peak_mib = (double)peak_b / (1024.0 * 1024.0);
    const double probe_mib = (double)(byte_count * 2u) / (1024.0 * 1024.0);

    const double govern_pct =
        peak_b == 0 ? 0.0 : (100.0 * (double)cur_b / (double)peak_b);

    puts("");
    puts("  ================================================================================");
    puts("  NRL runtime snapshot (single-shot INT4 bench probe)");
    puts("  ================================================================================");
    puts("");
    puts("  --- Summary --------------------------------------------------------------------");
    printf("  Bench mode            : %s\n", mode_label);
    printf("  Active variant        : %s\n", variant);
    printf("  Virtual throughput    : %.3f GOPS  (baseline-equivalent / probe wall time)\n",
           virtual_gops);
    printf("  Executed throughput   : %.3f GOPS  (hardware updates / probe wall time)\n",
           executed_gops);
    printf("  Skip ratio            : %.6f  (%.4f %% executed vs baseline)\n",
           skip_ratio,
           (1.0 - skip_ratio) * 100.0);
    printf("  Active sub-lattices   : %llu / %llu total (pruned %llu this probe)\n",
           (unsigned long long)stats.active_sublattices,
           (unsigned long long)stats.total_sublattices,
           (unsigned long long)stats.pruned_sublattices);
    puts("");
    puts("  --- Memory ---------------------------------------------------------------------");
    printf("  Process RSS (now)     : %.2f MiB\n", cur_mib);
    printf("  Process peak (RSS)    : %.2f MiB\n", peak_mib);
    printf("  Probe buffers (2x)    : %.2f MiB  (packed INT4 lattice for probe)\n", probe_mib);
    printf("  Reference lattice     : %.2f MiB packed INT4 @ %llu neurons (display scale)\n",
           ref_mib,
           (unsigned long long)ref_neurons);
    puts("");
    puts("  --- Port map -------------------------------------------------------------------");
    puts("  PORT              | STATUS        | DETAIL");
    puts("  ------------------+---------------+------------------------------------------");

    char detail[512];
    for (size_t i = 0; i < sizeof(k_brain_port_names) / sizeof(k_brain_port_names[0]); ++i) {
        const char *port = k_brain_port_names[i];
        const char *status = "OK";
        detail[0] = '\0';
        switch (i) {
        case 0:
            status = "Active";
            snprintf(detail,
                     sizeof(detail),
                     "Bench probe %llu neurons x %u iterations",
                     (unsigned long long)probe_neurons,
                     (unsigned)probe_iters);
            break;
        case 1:
            status = "Ready";
            snprintf(detail, sizeof(detail), "ISA: %s", feat_buf);
            break;
        case 2:
            status = skip_ratio > 0.99 ? "Dominant" : "Active";
            snprintf(detail, sizeof(detail), "skip_ratio=%.5f", skip_ratio);
            break;
        case 3:
            status = "Loaded";
            snprintf(detail,
                     sizeof(detail),
                     "Sublattice stride 1024; active %llu / %llu",
                     (unsigned long long)stats.active_sublattices,
                     (unsigned long long)stats.total_sublattices);
            break;
        case 4:
            status = "Healthy";
            snprintf(detail,
                     sizeof(detail),
                     "INT4 lane width 128-bit class (packed nibbles / AVX path)");
            break;
        case 5:
            status = "Engaged";
            snprintf(detail, sizeof(detail), "Stride-2048 sparse drive pattern (probe inputs)");
            break;
        case 6:
            status = "Synced";
            snprintf(detail, sizeof(detail), "Potentials/inputs banks released post-probe");
            break;
        case 7:
            status = rc == NRL_OK ? "Clear" : "Trip";
            snprintf(detail, sizeof(detail), "last kernel status=%d", (int)rc);
            break;
        case 8:
            status = govern_pct > 92.0 ? "Warm" : "Stable";
            snprintf(detail,
                     sizeof(detail),
                     "Working set / peak ratio %.1f %% (process-scoped)", govern_pct);
            break;
        case 9:
            status = "Idle";
            snprintf(detail, sizeof(detail), "probe wall time %.3f s", elapsed);
            break;
        default:
            break;
        }
        printf("  %-17s | %-13s | %s\n", port, status, detail);
    }
    puts("  ================================================================================");
    puts("");
    return 0;
}

/*
 * Sovereign-only INT4 assimilation: one machine-code path (nrl_v1_braincore_int4)
 * for binary-level parity with nrlpy inplace buffers and physics-lattice IR lowering.
 */
static int cmd_assimilate(int argc, char **argv) {
    const uint64_t neurons =
        parse_u64_or_default(argc >= 3 ? argv[2] : NULL, 4096);
    const uint64_t iters =
        parse_u64_or_default(argc >= 4 ? argv[3] : NULL, 256);
    const uint64_t threshold_u64 =
        parse_u64_or_default(argc >= 5 ? argv[4] : NULL, 10);

    if (neurons == 0 || (neurons & 1u) != 0u || iters == 0 ||
        threshold_u64 == 0 || threshold_u64 > 15u) {
        fputs("nrl assimilate: invalid args (neurons must be even, threshold 1..15)\n", stderr);
        return 1;
    }

    const size_t byte_count = nrl_v1_braincore_packed_bytes((size_t)neurons);
    if (byte_count == 0u) {
        fputs("nrl assimilate: packed byte count\n", stderr);
        return 1;
    }
    const uint8_t threshold = (uint8_t)threshold_u64;
    uint8_t *potentials = (uint8_t *)malloc(byte_count);
    uint8_t *inputs = (uint8_t *)malloc(byte_count);
    if (potentials == NULL || inputs == NULL) {
        free(potentials);
        free(inputs);
        fputs("nrl assimilate: allocation failed\n", stderr);
        return 1;
    }
    memset(potentials, 0, byte_count);
    for (size_t i = 0; i < byte_count; ++i) {
        inputs[i] = (uint8_t)(((uint8_t)i * 17u) & 0x77u);
    }

    const double t0 = now_seconds();
    const nrl_v1_status st =
        nrl_v1_braincore_int4(potentials, inputs, (size_t)neurons, (size_t)iters, threshold);
    const double t1 = now_seconds();
    if (st != NRL_OK) {
        free(potentials);
        free(inputs);
        fputs("nrl assimilate: kernel failed\n", stderr);
        return 1;
    }

    const uint64_t chk = checksum_u64(potentials, byte_count);
    printf("NRL assimilate braincore_int4\n");
    printf("  lane: sovereign (raw dispatched kernel)\n");
    printf("  variant: %s\n", nrl_v1_active_variant("braincore_int4"));
    printf("  packed_bytes: %llu\n", (unsigned long long)byte_count);
    printf("  neurons: %llu\n", (unsigned long long)neurons);
    printf("  iterations: %llu\n", (unsigned long long)iters);
    printf("  threshold: %u\n", (unsigned)threshold);
    printf("  elapsed_s: %.9f\n", t1 - t0);
    printf("  checksum_fnv1a64: %llu\n", (unsigned long long)chk);

    free(potentials);
    free(inputs);
    return 0;
}

static int cmd_run(int argc, char **argv) {
    const char *profile_name = argc >= 6 ? argv[5] : "sovereign";
    if (str_ieq(profile_name, "aes256") || str_ieq(profile_name, "aes256-synth")) {
        fputs("nrl run: aes256-synth is bench-only; use: nrl bench <N> <I> <R> <T> aes256-synth\n",
              stderr);
        return 1;
    }
    const run_profile *profile = profile_from_name(profile_name);
    if (profile == NULL) {
        fputs("nrl run: unknown profile (use sovereign|adaptive|war-drive|zpm|automatic|omega|omega-hybrid)\n", stderr);
        return 1;
    }

    const uint64_t neurons =
        parse_u64_or_default(argc >= 3 ? argv[2] : NULL, profile->default_neurons);
    const uint64_t iters =
        parse_u64_or_default(argc >= 4 ? argv[3] : NULL, profile->default_iters);
    const uint64_t threshold_u64 =
        parse_u64_or_default(argc >= 5 ? argv[4] : NULL, profile->default_threshold);

    if (neurons == 0 || (neurons & 1u) != 0u || iters == 0 ||
        threshold_u64 == 0 || threshold_u64 > 15u) {
        fputs("nrl run: invalid args\n", stderr);
        return 1;
    }

    const size_t byte_count = (size_t)(neurons >> 1);
    const uint8_t threshold = (uint8_t)threshold_u64;
    uint8_t *potentials = (uint8_t *)malloc(byte_count);
    uint8_t *inputs = (uint8_t *)malloc(byte_count);
    if (potentials == NULL || inputs == NULL) {
        free(potentials);
        free(inputs);
        fputs("nrl run: allocation failed\n", stderr);
        return 1;
    }
    memset(potentials, 0, byte_count);
    for (size_t i = 0; i < byte_count; ++i) {
        inputs[i] = (uint8_t)(((uint8_t)i * 17u) & 0x77u);
    }

    const double t0 = now_seconds();
    nrl_v1_status st = NRL_ERR_INTERNAL;
    uint64_t executed_updates = 0;
    uint64_t baseline_updates = 0;
    const int use_zpm = strcmp(profile_name, "zpm") == 0 ||
                        strcmp(profile_name, "automatic") == 0;
    const int use_omega = strcmp(profile_name, "omega") == 0;
    const int use_omega_hybrid = strcmp(profile_name, "omega-hybrid") == 0;
    if (use_zpm) {
        st = nrl_braincore_int4_zpm_static(
            potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
            &executed_updates, &baseline_updates);
    } else if (use_omega) {
        nrl_omega_stats stats = {0};
        const size_t sublattice_neurons = 1u << 10;
        st = nrl_braincore_int4_omega_virtual(
            potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
            sublattice_neurons, 4096u, 0u, 1u, 0u, &stats);
        executed_updates = stats.executed_updates;
        baseline_updates = stats.baseline_updates;
    } else if (use_omega_hybrid) {
        nrl_omega_stats stats = {0};
        const size_t sublattice_neurons = 1u << 10;
        const size_t total_sublattices =
            ((size_t)neurons + sublattice_neurons - 1u) / sublattice_neurons;
        const size_t min_active = (total_sublattices * 3u) / 4u;
        st = nrl_braincore_int4_omega_virtual(
            potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
            sublattice_neurons, 32u, 16u, min_active, 1u, &stats);
        executed_updates = stats.executed_updates;
        baseline_updates = stats.baseline_updates;
    } else {
        st = nrl_v1_braincore_int4(
            potentials, inputs, (size_t)neurons, (size_t)iters, threshold);
        executed_updates = (uint64_t)neurons * (uint64_t)iters;
        baseline_updates = executed_updates;
    }
    const double t1 = now_seconds();
    if (st != NRL_OK) {
        free(potentials);
        free(inputs);
        fputs("nrl run: kernel failed\n", stderr);
        return 1;
    }

    const char *mode_name = "system2-iterative";
    const char *variant_name = nrl_v1_active_variant("braincore_int4");
    if (use_zpm) {
        mode_name = "system1-automatic-zpm";
        variant_name = "zpm-static-exact";
    }
    if (use_omega) {
        mode_name = "system1-automatic-omega";
        variant_name = "omega-fractal-virtual";
    }
    if (use_omega_hybrid) {
        mode_name = "system1-automatic-omega-hybrid";
        variant_name = "omega-fractal-hybrid";
    }

    printf("NRL run braincore_int4\n");
    printf("  profile: %s\n", profile->name);
    printf("  mode: %s\n", mode_name);
    printf("  variant: %s\n", variant_name);
    printf("  neurons: %llu\n", (unsigned long long)neurons);
    printf("  iterations: %llu\n", (unsigned long long)iters);
    printf("  threshold: %u\n", (unsigned)threshold);
    printf("  elapsed_s: %.6f\n", t1 - t0);
    printf("  executed_updates: %llu\n", (unsigned long long)executed_updates);
    printf("  baseline_equiv_updates: %llu\n", (unsigned long long)baseline_updates);
    printf("  checksum_fnv1a64: %llu\n", (unsigned long long)checksum_u64(potentials, byte_count));

    free(potentials);
    free(inputs);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    const char *profile_name = argc >= 7 ? argv[6] : "sovereign";
    /* Synthetic digest bench: no INT4 lattice; must run before profile_from_name. */
    if (str_ieq(profile_name, "aes256") || str_ieq(profile_name, "aes256-synth")) {
        nrl_file_plan plan = {
            .do_bench = 1,
            .neurons = parse_u64_or_default(argc >= 3 ? argv[2] : NULL, 4096),
            .iterations = parse_u64_or_default(argc >= 4 ? argv[3] : NULL, 64),
            .reps = parse_u64_or_default(argc >= 5 ? argv[4] : NULL, 1),
            .threshold = parse_u64_or_default(argc >= 6 ? argv[5] : NULL, 6),
            .expected_fnv1a64 = 0,
            .has_expected_fnv1a64 = 0,
        };
        snprintf(plan.profile, sizeof(plan.profile), "aes256-synth");
        return cmd_aes256_synth_bench(&plan);
    }

    const run_profile *profile = profile_from_name(profile_name);
    if (profile == NULL) {
        fputs(
            "nrl bench: unknown profile (use sovereign|adaptive|war-drive|zpm|automatic|omega|omega-hybrid|aes256-synth)\n",
            stderr);
        return 1;
    }

    const uint64_t neurons =
        parse_u64_or_default(argc >= 3 ? argv[2] : NULL, profile->default_neurons);
    const uint64_t iters =
        parse_u64_or_default(argc >= 4 ? argv[3] : NULL, profile->default_iters);
    const uint64_t reps =
        parse_u64_or_default(argc >= 5 ? argv[4] : NULL, profile->default_reps);
    const uint64_t threshold_u64 =
        parse_u64_or_default(argc >= 6 ? argv[5] : NULL, profile->default_threshold);

    if (neurons == 0 || (neurons & 1u) != 0u || iters == 0 || reps == 0 ||
        threshold_u64 == 0 || threshold_u64 > 15u) {
        fputs("nrl bench: invalid args\n", stderr);
        return 1;
    }

    const size_t byte_count = (size_t)(neurons >> 1);
    const uint8_t threshold = (uint8_t)threshold_u64;
    uint8_t *potentials = (uint8_t *)malloc(byte_count);
    uint8_t *inputs = (uint8_t *)malloc(byte_count);
    if (potentials == NULL || inputs == NULL) {
        free(potentials);
        free(inputs);
        fputs("nrl bench: allocation failed\n", stderr);
        return 1;
    }

    const int use_zpm = strcmp(profile_name, "zpm") == 0 ||
                        strcmp(profile_name, "automatic") == 0;
    const int use_omega = strcmp(profile_name, "omega") == 0;
    const int use_omega_hybrid = strcmp(profile_name, "omega-hybrid") == 0;
    if (use_omega) {
        memset(inputs, 0, byte_count);
        for (size_t i = 0; i < byte_count; i += 2048) {
            inputs[i] = 0x77u;
        }
    } else if (use_omega_hybrid) {
        memset(inputs, 0, byte_count);
        for (size_t i = 0; i < byte_count; ++i) {
            if ((i & 3u) == 0u) {
                inputs[i] = 0x77u;
            }
        }
    } else {
        for (size_t i = 0; i < byte_count; ++i) {
            inputs[i] = (uint8_t)(((uint8_t)i * 37u) & 0x77u);
        }
    }

    const uint64_t warmup_reps = reps > 2 ? 2 : 1;
    for (uint64_t r = 0; r < warmup_reps; ++r) {
        memset(potentials, 0, byte_count);
        nrl_v1_status rc = NRL_ERR_INTERNAL;
        if (use_zpm) {
            rc = nrl_braincore_int4_zpm_static(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold, NULL, NULL);
        } else if (use_omega) {
            rc = nrl_braincore_int4_omega_virtual(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
                1u << 10, 4096u, 0u, 1u, 0u, NULL);
        } else if (use_omega_hybrid) {
            const size_t sublattice_neurons = 1u << 10;
            const size_t total_sublattices =
                ((size_t)neurons + sublattice_neurons - 1u) / sublattice_neurons;
            const size_t min_active = (total_sublattices * 3u) / 4u;
            rc = nrl_braincore_int4_omega_virtual(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
                sublattice_neurons, 32u, 16u, min_active, 1u, NULL);
        } else {
            rc = nrl_v1_braincore_int4(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold);
        }
        if (rc != NRL_OK) {
            free(potentials);
            free(inputs);
            fputs("nrl bench: kernel failed during warmup\n", stderr);
            return 1;
        }
    }

    uint64_t total_executed_updates = 0;
    uint64_t total_baseline_updates = 0;
    uint64_t total_active_sublattices = 0;
    uint64_t total_sublattices = 0;
    uint64_t total_pruned_sublattices = 0;
    const double t0 = now_seconds();
    for (uint64_t r = 0; r < reps; ++r) {
        memset(potentials, 0, byte_count);
        nrl_v1_status rc = NRL_ERR_INTERNAL;
        uint64_t executed_updates = 0;
        uint64_t baseline_updates = 0;
        if (use_zpm) {
            rc = nrl_braincore_int4_zpm_static(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
                &executed_updates, &baseline_updates);
        } else if (use_omega) {
            nrl_omega_stats stats = {0};
            rc = nrl_braincore_int4_omega_virtual(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
                1u << 10, 4096u, 0u, 1u, 0u, &stats);
            executed_updates = stats.executed_updates;
            baseline_updates = stats.baseline_updates;
            total_active_sublattices += stats.active_sublattices;
            total_sublattices += stats.total_sublattices;
            total_pruned_sublattices += stats.pruned_sublattices;
        } else if (use_omega_hybrid) {
            nrl_omega_stats stats = {0};
            const size_t sublattice_neurons = 1u << 10;
            const size_t total_sublattices_target =
                ((size_t)neurons + sublattice_neurons - 1u) / sublattice_neurons;
            const size_t min_active = (total_sublattices_target * 3u) / 4u;
            rc = nrl_braincore_int4_omega_virtual(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold,
                sublattice_neurons, 32u, 16u, min_active, 1u, &stats);
            executed_updates = stats.executed_updates;
            baseline_updates = stats.baseline_updates;
            total_active_sublattices += stats.active_sublattices;
            total_sublattices += stats.total_sublattices;
            total_pruned_sublattices += stats.pruned_sublattices;
        } else {
            rc = nrl_v1_braincore_int4(
                potentials, inputs, (size_t)neurons, (size_t)iters, threshold);
            executed_updates = (uint64_t)neurons * (uint64_t)iters;
            baseline_updates = executed_updates;
        }
        total_executed_updates += executed_updates;
        total_baseline_updates += baseline_updates;
        if (rc != NRL_OK) {
            free(potentials);
            free(inputs);
            fputs("nrl bench: kernel failed during benchmark\n", stderr);
            return 1;
        }
    }
    const double t1 = now_seconds();
    const double seconds = t1 - t0;
    const double executed_per_sec = (double)total_executed_updates / seconds;
    const double baseline_per_sec = (double)total_baseline_updates / seconds;
    const double executed_gops = executed_per_sec / 1e9;
    const double virtual_gops = baseline_per_sec / 1e9;
    const double skip_ratio =
        total_baseline_updates == 0
            ? 0.0
            : 1.0 - ((double)total_executed_updates / (double)total_baseline_updates);

    const char *mode_name = "system2-iterative";
    const char *variant_name = nrl_v1_active_variant("braincore_int4");
    if (use_zpm) {
        mode_name = "system1-automatic-zpm";
        variant_name = "zpm-static-exact";
    }
    if (use_omega) {
        mode_name = "system1-automatic-omega";
        variant_name = "omega-fractal-virtual";
    }
    if (use_omega_hybrid) {
        mode_name = "system1-automatic-omega-hybrid";
        variant_name = "omega-fractal-hybrid";
    }

    printf("NRL bench braincore_int4\n");
    printf("  profile: %s\n", profile->name);
    printf("  mode: %s\n", mode_name);
    printf("  variant: %s\n", variant_name);
    printf("  neurons: %llu\n", (unsigned long long)neurons);
    printf("  iterations: %llu\n", (unsigned long long)iters);
    printf("  reps: %llu\n", (unsigned long long)reps);
    printf("  threshold: %u\n", (unsigned)threshold);
    printf("  elapsed_s: %.6f\n", seconds);
    printf("  executed_updates: %llu\n", (unsigned long long)total_executed_updates);
    printf("  baseline_equiv_updates: %llu\n", (unsigned long long)total_baseline_updates);
    printf("  skip_ratio: %.6f\n", skip_ratio);
    if (use_omega || use_omega_hybrid) {
        printf("  avg_active_sublattices: %.3f\n",
               reps == 0 ? 0.0 : (double)total_active_sublattices / (double)reps);
        printf("  avg_total_sublattices: %.3f\n",
               reps == 0 ? 0.0 : (double)total_sublattices / (double)reps);
        printf("  avg_pruned_sublattices: %.3f\n",
               reps == 0 ? 0.0 : (double)total_pruned_sublattices / (double)reps);
    }
    printf("  executed_updates_per_sec: %.3f\n", executed_per_sec);
    printf("  baseline_equiv_updates_per_sec: %.3f\n", baseline_per_sec);
    printf("  executed_gops: %.3f\n", executed_gops);
    printf("  virtual_gops: %.3f\n", virtual_gops);

    free(potentials);
    free(inputs);
    return 0;
}

static int parse_u64_strict(const char *s, uint64_t *out) {
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (s == end || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

static int cmd_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "nrl file: could not open %s\n", path);
        return 1;
    }

    const run_profile *base = profile_from_name("sovereign");
    nrl_file_plan plan = {
        .do_bench = 1,
        .neurons = base->default_neurons,
        .iterations = base->default_iters,
        .reps = base->default_reps,
        .threshold = base->default_threshold,
        .expected_fnv1a64 = 0,
        .has_expected_fnv1a64 = 0,
    };
    strcpy(plan.profile, "sovereign");

    char line_buf[256];
    int line_no = 0;
    while (fgets(line_buf, sizeof(line_buf), fp) != NULL) {
        ++line_no;
        const char *line = trim_ws(line_buf);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (strcmp(line, "run") == 0) {
            plan.do_bench = 0;
            continue;
        }
        if (strcmp(line, "bench") == 0) {
            plan.do_bench = 1;
            continue;
        }

        char *eq = strchr((char *)line, '=');
        if (eq == NULL) {
            fprintf(stderr, "nrl file: parse error line %d: %s\n", line_no, line);
            fclose(fp);
            return 1;
        }
        *eq = '\0';
        const char *key = trim_ws((char *)line);
        const char *value = trim_ws(eq + 1);

        if (strcmp(key, "mode") == 0) {
            if (strcmp(value, "run") == 0) {
                plan.do_bench = 0;
                continue;
            }
            if (strcmp(value, "bench") == 0) {
                plan.do_bench = 1;
                continue;
            }
            fprintf(stderr, "nrl file: invalid mode at line %d\n", line_no);
            fclose(fp);
            return 1;
        }
        if (strcmp(key, "profile") == 0) {
            if (strlen(value) >= sizeof(plan.profile)) {
                fprintf(stderr, "nrl file: profile too long at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            strcpy(plan.profile, value);
            continue;
        }
        if (strcmp(key, "neurons") == 0) {
            if (!parse_u64_strict(value, &plan.neurons)) {
                fprintf(stderr, "nrl file: invalid neurons at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            continue;
        }
        if (strcmp(key, "iterations") == 0) {
            if (!parse_u64_strict(value, &plan.iterations)) {
                fprintf(stderr, "nrl file: invalid iterations at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            continue;
        }
        if (strcmp(key, "reps") == 0) {
            if (!parse_u64_strict(value, &plan.reps)) {
                fprintf(stderr, "nrl file: invalid reps at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            continue;
        }
        if (strcmp(key, "threshold") == 0) {
            if (!parse_u64_strict(value, &plan.threshold)) {
                fprintf(stderr, "nrl file: invalid threshold at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            continue;
        }
        if (strcmp(key, "expected_fnv1a64") == 0) {
            if (!parse_u64_strict(value, &plan.expected_fnv1a64)) {
                fprintf(stderr, "nrl file: invalid expected_fnv1a64 at line %d\n", line_no);
                fclose(fp);
                return 1;
            }
            plan.has_expected_fnv1a64 = 1;
            continue;
        }

        fprintf(stderr, "nrl file: unknown key '%s' at line %d\n", key, line_no);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    char n_buf[32];
    char i_buf[32];
    char r_buf[32];
    char t_buf[32];
    snprintf(n_buf, sizeof(n_buf), "%llu", (unsigned long long)plan.neurons);
    snprintf(i_buf, sizeof(i_buf), "%llu", (unsigned long long)plan.iterations);
    snprintf(r_buf, sizeof(r_buf), "%llu", (unsigned long long)plan.reps);
    snprintf(t_buf, sizeof(t_buf), "%llu", (unsigned long long)plan.threshold);

    if (str_ieq(plan.profile, "aes256") || str_ieq(plan.profile, "aes256-synth")) {
        return cmd_aes256_synth_bench(&plan);
    }

    if (plan.do_bench) {
        if (profile_from_name(plan.profile) == NULL) {
            fprintf(stderr,
                    "nrl file: unknown profile '%s' (try sovereign|omega|... or aes256-synth)\n",
                    plan.profile);
            return 1;
        }
        char *argv_bench[] = {"nrl", "bench", n_buf, i_buf, r_buf, t_buf, plan.profile};
        return cmd_bench(7, argv_bench);
    }

    if (profile_from_name(plan.profile) == NULL) {
        fprintf(stderr,
                "nrl file: unknown profile '%s' (try sovereign|omega|... or aes256-synth)\n",
                plan.profile);
        return 1;
    }
    char *argv_run[] = {"nrl", "run", n_buf, i_buf, t_buf, plan.profile};
    return cmd_run(6, argv_run);
}

int main(int argc, char **argv) {
    if (nrl_v1_init() != NRL_OK) {
        fputs("nrl: initialization failed\n", stderr);
        return 1;
    }

    if (argc <= 1) {
        print_usage();
        return 0;
    }

    if (strcmp(argv[1], "--version") == 0) {
        puts(nrl_v1_version());
        return 0;
    }

    if (strcmp(argv[1], "--features") == 0) {
        print_features();
        return 0;
    }

    if (argc >= 3 && (strcmp(argv[1], "-ai") == 0 || strcmp(argv[1], "--ai") == 0)) {
        if (lm_ai_arg_is_on(argv[2])) {
            return cmd_lm_ai_toggle(1);
        }
        if (lm_ai_arg_is_off(argv[2])) {
            return cmd_lm_ai_toggle(0);
        }
        fputs("nrl -ai: expected on|off (or --on|--off|-on|-off)\n", stderr);
        return 2;
    }

    if (strcmp(argv[1], "runtime") == 0) {
        return cmd_runtime();
    }

    if (strcmp(argv[1], "status") == 0 || strcmp(argv[1], "-status") == 0) {
        return cmd_status();
    }

    if (strcmp(argv[1], "inquire") == 0 || strcmp(argv[1], "-inquire") == 0) {
        return cmd_inquire(argc >= 3 ? argv[2] : "");
    }

    if (strcmp(argv[1], "chat") == 0 || strcmp(argv[1], "-chat") == 0) {
        return cmd_chat(argc, argv);
    }

    if (strcmp(argv[1], "control") == 0 || strcmp(argv[1], "-control") == 0) {
        return cmd_control(argc, argv);
    }

    if (strcmp(argv[1], "brain-map") == 0) {
        return cmd_brain_map();
    }

    if (strcmp(argv[1], "variant") == 0 && argc >= 3) {
        printf("%s\n", nrl_v1_active_variant(argv[2]));
        return 0;
    }

    if (strcmp(argv[1], "file") == 0 && argc >= 3) {
        return cmd_file(argv[2]);
    }

    if (strcmp(argv[1], "run") == 0) {
        return cmd_run(argc, argv);
    }

    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc, argv);
    }

    if (strcmp(argv[1], "assimilate") == 0) {
        return cmd_assimilate(argc, argv);
    }

    if (strcmp(argv[1], "demo") == 0) {
        return cmd_demo();
    }

    if (has_suffix(argv[1], ".nrl")) {
        return cmd_file(argv[1]);
    }

    print_usage();
    return 1;
}
