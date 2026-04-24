# stress_test.py
# Ultimate NRL / NRLPY Stress Test
# Copyright Daniel Harding - RomanAILabs

import subprocess
import time
import re

NRL_BIN = r"C:\Users\Asus\AppData\Local\Programs\NRL\bin\nrl.exe"

NEURON_SIZES = [131072, 262144, 524288, 1048576, 2097152]
ITERATIONS = 256
THRESHOLD = 8

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def parse_value(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None

def run_assimilate(n):
    cmd = [NRL_BIN, "assimilate", str(n), str(ITERATIONS), str(THRESHOLD)]
    out = run_cmd(cmd)
    checksum = parse_value(r"checksum_fnv1a64:\s*(\d+)", out, int)
    elapsed = parse_value(r"elapsed_s:\s*([\d\.]+)", out)
    return checksum, elapsed

def run_bench(n, mode):
    cmd = [NRL_BIN, "bench", str(n), str(ITERATIONS), str(THRESHOLD), "2", mode]
    out = run_cmd(cmd)

    return {
        "skip": parse_value(r"skip_ratio:\s*([\d\.]+)", out),
        "exec_gops": parse_value(r"executed_gops:\s*([\d\.]+)", out),
        "virt_gops": parse_value(r"virtual_gops:\s*([\d\.]+)", out),
    }

print("\n=== NRL ULTIMATE STRESS TEST ===\n")

for n in NEURON_SIZES:
    print(f"\n--- SIZE: {n} ---")

    c1, _ = run_assimilate(n)
    c2, _ = run_assimilate(n)
    print(f"checksum: {c1} | verified: {c1 == c2}")

    t0 = time.time()
    sov = run_bench(n, "sovereign")
    t1 = time.time()

    omg = run_bench(n, "omega")
    t2 = time.time()

    sov_time = t1 - t0
    omg_time = t2 - t1
    speedup = sov_time / omg_time if omg_time > 0 else 0

    print(f"SOV: {sov_time:.4f}s | OMG: {omg_time:.4f}s | SPEEDUP: {speedup:.2f}x")
    print(f"SKIP: {omg['skip']:.6f} | REAL GOPS: {sov['exec_gops']:.2f} | VIRT GOPS: {omg['virt_gops']:.2f}")

print("\n=== DONE ===\n")
