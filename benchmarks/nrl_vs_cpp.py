# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Locked NRL-vs-C++ benchmark harness for INT4 braincore.

Builds C++ reference baselines (-O0 and -O3), runs native NRL profiles, and
emits machine-readable + markdown artifacts for claim governance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


def _ensure_repo_nrlpy_on_path(root: Path) -> None:
    src = root / "nrlpy" / "src"
    if src.is_dir():
        sp = str(src)
        if sp not in sys.path:
            sys.path.insert(0, sp)


CPP_SRC = r"""
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static inline void braincore_int4_scalar(
    std::uint8_t* packed_potentials,
    const std::uint8_t* packed_inputs,
    std::size_t neuron_count,
    std::size_t iterations,
    std::uint8_t threshold) {
  const std::size_t byte_count = neuron_count >> 1;
  for (std::size_t it = 0; it < iterations; ++it) {
    for (std::size_t i = 0; i < byte_count; ++i) {
      const std::uint8_t p = packed_potentials[i];
      const std::uint8_t in = packed_inputs[i];
      std::uint8_t lo = (std::uint8_t)((p & 0x0fu) + (in & 0x0fu));
      std::uint8_t hi = (std::uint8_t)(((p >> 4) & 0x0fu) + ((in >> 4) & 0x0fu));
      if (lo > 15u) lo = 15u;
      if (hi > 15u) hi = 15u;
      if (lo >= threshold) lo = 0u;
      if (hi >= threshold) hi = 0u;
      packed_potentials[i] = (std::uint8_t)(lo | (std::uint8_t)(hi << 4));
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: cpp_baseline <neurons> <iters> <reps> <threshold>\n");
    return 2;
  }
  const std::size_t neurons = (std::size_t)std::strtoull(argv[1], nullptr, 10);
  const std::size_t iterations = (std::size_t)std::strtoull(argv[2], nullptr, 10);
  const std::size_t reps = (std::size_t)std::strtoull(argv[3], nullptr, 10);
  const std::uint8_t threshold = (std::uint8_t)std::strtoul(argv[4], nullptr, 10);

  if (neurons == 0 || (neurons & 1u) != 0u || iterations == 0 || reps == 0 || threshold == 0 || threshold > 15u) {
    std::fprintf(stderr, "invalid args\n");
    return 2;
  }

  const std::size_t byte_count = neurons >> 1;
  std::vector<std::uint8_t> pot(byte_count, 0);
  std::vector<std::uint8_t> inp(byte_count, 0);
  for (std::size_t i = 0; i < byte_count; ++i) {
    inp[i] = (std::uint8_t)(((std::uint8_t)i * 37u) & 0x77u);
  }

  const std::size_t warmup_reps = reps > 2 ? 2 : 1;
  for (std::size_t r = 0; r < warmup_reps; ++r) {
    std::fill(pot.begin(), pot.end(), 0);
    braincore_int4_scalar(pot.data(), inp.data(), neurons, iterations, threshold);
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (std::size_t r = 0; r < reps; ++r) {
    std::fill(pot.begin(), pot.end(), 0);
    braincore_int4_scalar(pot.data(), inp.data(), neurons, iterations, threshold);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  const double updates = (double)neurons * (double)iterations * (double)reps;
  const double gops = updates / seconds / 1e9;
  std::printf("elapsed_s=%.9f\n", seconds);
  std::printf("gops=%.6f\n", gops);
  return 0;
}
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nrl_bin(root: Path) -> Path:
    exe = "nrl.exe" if os.name == "nt" else "nrl"
    p = root / "build" / "bin" / exe
    if not p.exists():
        raise FileNotFoundError(f"NRL binary missing: {p}. Run build.ps1 first.")
    return p


def _zig(root: Path) -> str:
    candidates = [
        root / "tools" / "zig" / "zig.exe",
        root.parent / "trinary" / "tools" / "zig" / "zig.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    if shutil.which("zig"):
        return "zig"
    raise FileNotFoundError("zig not found (NRL/tools/zig or Trinary/tools/zig or PATH)")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _parse_simple_kv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _parse_nrl_bench(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line or line.startswith("NRL bench"):
            continue
        k, v = [x.strip() for x in line.split(":", 1)]
        k = k.replace(" ", "_")
        if k in {"profile", "mode", "variant"}:
            out[k] = v
            continue
        if k in {"neurons", "iterations", "reps", "threshold", "executed_updates", "baseline_equiv_updates"}:
            out[k] = int(float(v))
            continue
        out[k] = float(v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NRL vs C++ INT4 baseline benchmark")
    parser.add_argument("--neurons", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=16_384)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument(
        "--profiles",
        default="sovereign,omega-hybrid,omega,automatic",
        help="comma-separated NRL profiles",
    )
    args = parser.parse_args()

    root = _repo_root()
    _ensure_repo_nrlpy_on_path(root)
    from nrlpy.workload import build_workload_descriptor, workload_identity_block  # noqa: PLC0415

    try:
        from nrlpy import runtime  # noqa: PLC0415

        nrl_version = runtime.version()
    except Exception:
        nrl_version = None

    nrl = _nrl_bin(root)
    zig = _zig(root)
    out_dir = root / "build" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpp_src = out_dir / "cpp_int4_baseline.cpp"
    cpp_src.write_text(CPP_SRC, encoding="utf-8")
    cpp_o0 = out_dir / ("cpp_int4_o0.exe" if os.name == "nt" else "cpp_int4_o0")
    cpp_o3 = out_dir / ("cpp_int4_o3.exe" if os.name == "nt" else "cpp_int4_o3")

    compile_common = [zig, "c++", str(cpp_src), "-std=c++17", "-o"]
    for exe, opt in ((cpp_o0, "-O0"), (cpp_o3, "-O3")):
        cmd = compile_common + [str(exe), opt]
        cp = _run(cmd, cwd=root)
        if cp.returncode != 0:
            raise RuntimeError(f"C++ compile failed ({opt}): {cp.stderr.strip()}")

    run_args = [str(args.neurons), str(args.iterations), str(args.reps), str(args.threshold)]
    cp_o0 = _run([str(cpp_o0), *run_args], cwd=root)
    cp_o3 = _run([str(cpp_o3), *run_args], cwd=root)
    if cp_o0.returncode != 0 or cp_o3.returncode != 0:
        raise RuntimeError(
            f"C++ run failed\nO0:\n{cp_o0.stdout}\n{cp_o0.stderr}\nO3:\n{cp_o3.stdout}\n{cp_o3.stderr}"
        )
    base_o0 = _parse_simple_kv(cp_o0.stdout)
    base_o3 = _parse_simple_kv(cp_o3.stdout)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    nrl_results: list[dict[str, Any]] = []
    for profile in profiles:
        cp = _run(
            [
                str(nrl),
                "bench",
                str(args.neurons),
                str(args.iterations),
                str(args.reps),
                str(args.threshold),
                profile,
            ],
            cwd=root,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"NRL bench failed for profile={profile}: {cp.stderr.strip()}")
        parsed = _parse_nrl_bench(cp.stdout)
        parsed["speedup_vs_cpp_o0"] = base_o0["elapsed_s"] / parsed["elapsed_s"]
        parsed["speedup_vs_cpp_o3"] = base_o3["elapsed_s"] / parsed["elapsed_s"]
        nrl_results.append(parsed)

    artifact = {
        "workload": {
            "neurons": args.neurons,
            "iterations": args.iterations,
            "reps": args.reps,
            "threshold": args.threshold,
        },
        "cpp_baseline": {
            "o0": base_o0,
            "o3": base_o3,
        },
        "nrl_profiles": nrl_results,
    }
    desc = build_workload_descriptor(
        harness_id="nrl_vs_cpp",
        neurons=args.neurons,
        iterations=args.iterations,
        reps=args.reps,
        threshold=args.threshold,
        profiles=profiles,
        nrl_version=nrl_version,
    )
    artifact["workload_identity"] = workload_identity_block(desc)

    json_path = out_dir / "nrl_vs_cpp.json"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    md_lines = [
        "# NRL vs C++ INT4 Benchmark",
        "",
        f"Workload: neurons={args.neurons}, iterations={args.iterations}, reps={args.reps}, threshold={args.threshold}",
        "",
        f"C++ -O0: elapsed_s={base_o0['elapsed_s']:.6f}, gops={base_o0['gops']:.3f}",
        f"C++ -O3: elapsed_s={base_o3['elapsed_s']:.6f}, gops={base_o3['gops']:.3f}",
        "",
        "## NRL profiles",
    ]
    for row in nrl_results:
        md_lines.extend(
            [
                f"- profile={row['profile']} mode={row['mode']} elapsed_s={row['elapsed_s']:.6f}",
                f"  - executed_gops={row.get('executed_gops', 0.0):.3f} virtual_gops={row.get('virtual_gops', 0.0):.3f}",
                f"  - speedup_vs_cpp_o0={row['speedup_vs_cpp_o0']:.3f}x speedup_vs_cpp_o3={row['speedup_vs_cpp_o3']:.3f}x",
            ]
        )
    md_path = out_dir / "nrl_vs_cpp.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(textwrap.dedent(
        f"""
        wrote:
          {json_path}
          {md_path}
        """
    ).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
