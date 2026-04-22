# NRL Initial Native Benchmark Snapshot

Machine: local development host (Windows x86_64, AVX2 available)
Kernel: `braincore_int4`
Variant: `avx2`
Date: 2026-04-22

## Runs

### Sovereign profile (default)

Command:
`nrl bench`

Result:
- neurons: 1,048,576
- iterations: 256
- reps: 12
- threshold: 8
- elapsed_s: 0.159100
- updates_per_sec: 20,246,482,394.012
- gops: 20.246

### Adaptive profile sample

Command:
`nrl bench 1048576 384 10 10 adaptive`

Result:
- neurons: 1,048,576
- iterations: 384
- reps: 10
- threshold: 10
- elapsed_s: 0.205398
- updates_per_sec: 19,603,540,048.436
- gops: 19.604

### War-Drive profile sample

Command:
`nrl bench 4194304 256 16 8 war-drive`

Result:
- neurons: 4,194,304
- iterations: 256
- reps: 16
- threshold: 8
- elapsed_s: 1.991653
- updates_per_sec: 8,625,937,096.955
- gops: 8.626

## ZPM static accelerator (exact, no approximation)

These runs use the same kernel semantics as iterative execution, but collapse
repeated static-input steps through precomputed transition maps.

### Same workload as sovereign baseline

Commands:
- `nrl bench 1048576 256 12 8 sovereign`
- `nrl bench 1048576 256 12 8 zpm`

Results:
- iterative elapsed_s: 0.154222
- zpm elapsed_s: 0.007821
- speedup: 19.72x
- zpm skip_ratio: 0.996094
- iterative gops: 20.887
- zpm virtual_gops: 411.848

### High-iteration workload (ZPM muscle-memory lane)

Commands:
- `nrl bench 1048576 4096 12 8 sovereign`
- `nrl bench 1048576 4096 12 8 zpm`

Results:
- iterative elapsed_s: 2.360455
- zpm elapsed_s: 0.008400
- speedup: 281.01x
- zpm skip_ratio: 0.999756
- iterative gops: 21.835
- zpm virtual_gops: 6135.375

## Omega fractal-routed benchmark (System 1 virtual lane)

Command:
`nrl bench 1048576 16384 10 8 omega`

Result:
- mode: `system1-automatic-omega`
- variant: `omega-fractal-virtual`
- elapsed_s: 0.004944
- executed_updates: 327,680
- baseline_equiv_updates: 171,798,691,840
- skip_ratio: 0.999998
- avg_active_sublattices: 32.0 / 1024.0
- avg_pruned_sublattices: 768.0 / 1024.0
- executed_gops: 0.066
- virtual_gops: 34,749.629

Interpretation:
- Omega is intentionally compute-avoidance-first.
- It runs a tiny active fraction while preserving deterministic routing policy.

## Omega Hybrid (keep native throughput + skip dark regions)

Commands:
- `nrl bench 1048576 4096 12 8 sovereign`
- `nrl bench 1048576 4096 12 8 omega-hybrid`
- `nrl bench 1048576 4096 12 8 automatic`

Results:
- sovereign executed_gops: 19.646
- omega-hybrid executed_gops: 19.031
- omega-hybrid virtual_gops: 25.375
- omega-hybrid skip_ratio: 0.250000
- omega-hybrid elapsed_s: 2.031131
- sovereign elapsed_s: 2.623411
- automatic (zpm) elapsed_s: 0.008371

Takeaway:
- `omega-hybrid` keeps near-original raw GOPS while still skipping 25% of baseline work.
- `automatic` remains the extreme System 1 path for maximum time-to-answer acceleration.

## Notes

- These are first-pass native measurements, intended as baseline data.
- Cross-language and apples-to-apples methodology will be added in the benchmark governance phase.
