"""nrlpy smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nrlpy import native, runtime
from nrlpy.cli import main as cli_main
from nrlpy.compat import llm_globals
from nrlpy.shell import run_path


def test_nrl_binary_path_respects_nrl_root_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "nrl.exe" if os.name == "nt" else "nrl"
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    exe = tmp_path / "bin" / name
    exe.write_bytes(b"")
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.delenv("NRL_BIN", raising=False)
    assert runtime.nrl_binary_path().resolve() == exe.resolve()


def test_version_has_nrl() -> None:
    assert "nrl" in runtime.version().lower()


def test_features_has_avx_key() -> None:
    features = runtime.features()
    assert "avx2" in features


def test_braincore_int4_runs() -> None:
    result = runtime.braincore_int4(neurons=2048, iterations=16, threshold=8)
    assert result["kernel"] == "braincore_int4"
    assert result["iterations"] == 16


def test_bench_cli_mode() -> None:
    result = runtime.bench_cli(
        neurons=2048,
        iterations=32,
        reps=2,
        threshold=8,
        profile="sovereign",
    )
    assert result["profile"] == "sovereign"
    assert result["executed_gops"] >= 0.0


def test_cli_implicit_py_invokes_run_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("# demo\n", encoding="utf-8")
    seen: list[tuple[str, list[str]]] = []

    def capture(path: str, extra: list[str]) -> None:
        seen.append((path, extra))

    monkeypatch.setattr("nrlpy.cli.run_path", capture)
    assert cli_main([str(script), "a", "b"]) == 0
    assert len(seen) == 1
    assert seen[0][0] == str(script.resolve())
    assert seen[0][1] == ["a", "b"]


def test_cli_demo_dispatches_to_run_path(monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []

    def capture(path: str, extra: list[str]) -> None:
        paths.append(path)
        assert extra == []

    monkeypatch.setattr("nrlpy.cli.run_path", capture)
    assert cli_main(["demo"]) == 0
    assert len(paths) == 1
    assert paths[0].endswith(os.path.join("examples", "ultimate_power_demo.py"))


def test_braincore_packed_bytes() -> None:
    assert runtime.braincore_packed_bytes(0) == 0
    assert runtime.braincore_packed_bytes(15) == 0
    assert runtime.braincore_packed_bytes(16) == 8


def test_assimilate_inplace_matches_cli() -> None:
    neurons, iters, thresh = 4096, 256, 10
    pot, inp = native.assimilation_tensors(neurons)
    out = runtime.braincore_int4_inplace(pot, inp, neurons, iters, thresh)
    cli = runtime.assimilate_cli(neurons, iters, thresh)
    assert out["checksum_fnv1a64"] == cli["checksum_fnv1a64"]


def test_llm_globals_has_nrl() -> None:
    g = llm_globals()
    assert "nrl" in g and "NRL" in g
    assert g["nrl"].packed_bytes(16) == 8


def test_run_path_injects_nrl(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text("assert nrl.packed_bytes(16) == 8\n", encoding="utf-8")
    run_path(str(script), [])


def test_run_nrl_file(tmp_path: Path) -> None:
    program = tmp_path / "sample.nrl"
    program.write_text(
        "mode=bench\nprofile=sovereign\nneurons=2048\niterations=32\nreps=2\nthreshold=8\n",
        encoding="utf-8",
    )
    out = runtime.run_nrl_file(str(program))
    assert "NRL bench braincore_int4" in out
