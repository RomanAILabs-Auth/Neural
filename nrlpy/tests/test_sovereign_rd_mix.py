# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 11 — Sovereign R&D workload + R2→R0 promotion regression tests.

These tests pin three invariants of the Phase 11 Final Production Hinge:

1. The ``_SOVEREIGN_RD_CORPUS`` and ``_SOVEREIGN_RD_NOVEL_PROMPTS``
   constants satisfy the n-gram rescue admission condition
   (rephrases score >= 0.30 Jaccard versus their seed; novels score
   < 0.30 versus every seed) so the scenario's turn-type labels are
   honest.
2. The scenario runner registers as ``sovereign_rd_mix`` in the
   benchmark report with a fixed 100-turn plan and produces all the
   Phase 11 segmentation fields the formatter expects.
3. ``_promote_rescue_to_r0`` writes a Muscle Memory file keyed by the
   rephrased prompt when called on a rescue-served turn — the
   production-beta contract that "the second ask of a rephrased
   question is instant".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nrlpy import final_wps, gguf
from nrlpy.final_wps import (
    _SOVEREIGN_RD_CORPUS,
    _SOVEREIGN_RD_NOVEL_PROMPTS,
    run_final_wps_benchmark,
)
from nrlpy.lmo import _char_ngrams, _ngram_jaccard, _NGRAM_OVERLAP_THRESHOLD


# --------------------------------------------------------------------------- #
# 1. Corpus invariants                                                        #
# --------------------------------------------------------------------------- #


class TestCorpusInvariants:
    def test_corpus_has_twenty_seeds(self) -> None:
        assert len(_SOVEREIGN_RD_CORPUS) == 20

    def test_every_seed_has_two_rephrases(self) -> None:
        for row in _SOVEREIGN_RD_CORPUS:
            seed_prompt, seed_reply, rephrases = row
            assert isinstance(seed_prompt, str) and seed_prompt
            assert isinstance(seed_reply, str) and seed_reply
            assert len(rephrases) == 2
            for r in rephrases:
                assert isinstance(r, str) and r

    def test_every_rephrase_clears_ngram_floor(self) -> None:
        """Every rephrase must score >= the release-gate floor against
        its seed, otherwise the R2 rescue path would reject it at
        admission and the scenario's turn-type label would be wrong.
        """
        floor = _NGRAM_OVERLAP_THRESHOLD
        for seed_prompt, _reply, (rephrase_a, rephrase_b) in _SOVEREIGN_RD_CORPUS:
            seed_grams = _char_ngrams(seed_prompt, 3)
            for rephrase in (rephrase_a, rephrase_b):
                overlap = _ngram_jaccard(
                    seed_grams, _char_ngrams(rephrase, 3)
                )
                assert overlap >= floor, (
                    f"rephrase {rephrase!r} vs seed {seed_prompt!r} "
                    f"scored {overlap:.3f} < floor {floor:.3f}"
                )

    def test_novel_prompts_stay_below_ngram_floor(self) -> None:
        """Novel prompts must not accidentally match any seed, or the R5
        turn count drifts and the segmentation block misreports.
        """
        floor = _NGRAM_OVERLAP_THRESHOLD
        seed_grams_by_seed = [
            (seed, _char_ngrams(seed, 3))
            for seed, _r, _rs in _SOVEREIGN_RD_CORPUS
        ]
        assert len(_SOVEREIGN_RD_NOVEL_PROMPTS) == 20
        for novel in _SOVEREIGN_RD_NOVEL_PROMPTS:
            novel_grams = _char_ngrams(novel, 3)
            for seed_prompt, seed_grams in seed_grams_by_seed:
                overlap = _ngram_jaccard(seed_grams, novel_grams)
                assert overlap < floor, (
                    f"novel prompt {novel!r} accidentally overlaps seed "
                    f"{seed_prompt!r} at {overlap:.3f} >= floor {floor:.3f}"
                )

    def test_all_prompts_are_unique(self) -> None:
        seen: set[str] = set()
        for seed_prompt, _reply, (rephrase_a, rephrase_b) in _SOVEREIGN_RD_CORPUS:
            for p in (seed_prompt, rephrase_a, rephrase_b):
                assert p not in seen, f"duplicate prompt: {p!r}"
                seen.add(p)
        for novel in _SOVEREIGN_RD_NOVEL_PROMPTS:
            assert novel not in seen, f"novel duplicates corpus: {novel!r}"
            seen.add(novel)
        # 20 seeds + 40 rephrases + 20 novels = 80 unique prompts.
        assert len(seen) == 80


# --------------------------------------------------------------------------- #
# 2. Scenario runner shape                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def bench_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    for k in (
        "NRL_ZPM",
        "NRL_ZPM_THRESHOLD",
        "NRL_GATE_SKIP_RATIO_OVERRIDE",
        "NRL_COHERENCE_LANE",
        "NRL_R2_SHADOW",
        "NRL_OMEGA_BUDGET_MS",
    ):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture()
def dummy_model(bench_env: Path) -> Path:
    p = bench_env / "model.gguf"
    p.write_bytes(b"GGUF\x00sovereign-rd-test")
    return p


class TestScenarioRunner:
    def test_sovereign_scenario_runs_exactly_one_hundred_turns(
        self, dummy_model: Path, bench_env: Path
    ) -> None:
        report = run_final_wps_benchmark(
            model_path=str(dummy_model),
            nrl_root=bench_env,
            runner_backend="native_full",
            seed=1,
            max_tokens=12,
            turns_per_scenario=4,
            realistic_chat_turns=20,
            benchmark_class="A",
        )
        sovereign = report.scenario("sovereign_rd_mix")
        assert sovereign is not None, "sovereign_rd_mix must appear in report"
        assert sovereign.turns == 100, (
            f"sovereign_rd_mix plan is fixed at 100 turns; got {sovereign.turns}"
        )

    def test_sovereign_report_exposes_segmentation_fields(
        self, dummy_model: Path, bench_env: Path
    ) -> None:
        report = run_final_wps_benchmark(
            model_path=str(dummy_model),
            nrl_root=bench_env,
            runner_backend="native_full",
            seed=1,
            max_tokens=12,
            turns_per_scenario=4,
            realistic_chat_turns=20,
            benchmark_class="A",
        )
        sovereign = report.scenario("sovereign_rd_mix")
        assert sovereign is not None
        for field in (
            "r2_active_hits",
            "r2_ngram_rescues",
            "r2_rescue_avg_overlap",
            "rung_histogram",
        ):
            assert hasattr(sovereign, field), field
        assert isinstance(sovereign.rung_histogram, dict)

    def test_sovereign_passes_into_release_gate_computation(
        self, dummy_model: Path, bench_env: Path
    ) -> None:
        """Either scenario clearing the bar must flip ``passes_gate``."""
        report = run_final_wps_benchmark(
            model_path=str(dummy_model),
            nrl_root=bench_env,
            runner_backend="native_full",
            seed=1,
            max_tokens=12,
            turns_per_scenario=4,
            realistic_chat_turns=20,
            benchmark_class="A",
        )
        realistic = report.scenario("realistic_chat")
        sovereign = report.scenario("sovereign_rd_mix")
        assert realistic is not None and sovereign is not None
        expected = (
            realistic.effective_wps >= report.effective_wps_gate
            or sovereign.effective_wps >= report.effective_wps_gate
        )
        assert report.passes_gate is expected


# --------------------------------------------------------------------------- #
# 3. R2→R0 promotion                                                          #
# --------------------------------------------------------------------------- #


def _make_manifest(
    model_path: Path, prompt: str, *, model_sha: str
) -> gguf.GgufManifest:
    m = gguf.manifest_from_args(
        str(model_path),
        prompt=prompt,
        max_tokens=16,
        seed=7,
        muscle_memory="on",
        runner_backend="native_full",
        coherence_lane="max-throughput",
        r2_shadow_enabled=True,
        benchmark_class="A",
    )
    m.model_sha256 = model_sha
    m.zpm_nullspace = True
    return m


class TestR2ToR0Promotion:
    def test_promotion_writes_mm_entry_under_rephrased_prompt_key(
        self, bench_env: Path, tmp_path: Path
    ) -> None:
        """A rescue-served turn must leave an MM file keyed by the
        NEW (rephrased) prompt so the next ask hits R0 directly.
        """
        model = tmp_path / "promotion.gguf"
        model.write_bytes(b"GGUF\x00promotion-fixture")
        model_sha = hashlib.sha256(b"promotion-fixture").hexdigest()
        rephrased_prompt = "Tell me about the Resolution Ladder in NRL."
        served_reply = "Seed-reply text served via Phase 11 n-gram rescue."
        mf = _make_manifest(model, rephrased_prompt, model_sha=model_sha)

        # Pre-condition: no MM entry yet for this prompt.
        pre_key = gguf._muscle_memory_key(mf)
        pre_path = bench_env / "cache" / "mm" / model_sha / f"{pre_key:016x}.mm"
        assert not pre_path.exists()

        # Call the promotion helper directly — exactly what the three
        # R2-served branches in ``run_gguf`` do on rescue_served turns.
        gguf._promote_rescue_to_r0(
            mf,
            prompt=rephrased_prompt,
            text=served_reply,
            tokens=max(1, len(served_reply.split())),
            model_sha=model_sha,
            wall_s=0.012,
        )

        # Post-condition: the MM file must exist, and its body must
        # round-trip back through ``muscle_memory_lookup`` to the served
        # reply — so the next turn of this prompt returns at R0 speed.
        assert pre_path.is_file(), (
            f"promotion did not write the MM file at {pre_path}"
        )
        hit = gguf.muscle_memory_lookup(mf)
        assert hit is not None, (
            "muscle_memory_lookup must find the promoted entry"
        )
        assert hit.text == served_reply

    def test_promotion_is_noop_when_muscle_memory_off(
        self, bench_env: Path, tmp_path: Path
    ) -> None:
        model = tmp_path / "mm-off.gguf"
        model.write_bytes(b"GGUF\x00mm-off-fixture")
        model_sha = hashlib.sha256(b"mm-off-fixture").hexdigest()
        mf = _make_manifest(model, "noop prompt", model_sha=model_sha)
        mf.muscle_memory = "off"

        gguf._promote_rescue_to_r0(
            mf,
            prompt="noop prompt",
            text="should never be written",
            tokens=4,
            model_sha=model_sha,
            wall_s=0.001,
        )

        mm_root = bench_env / "cache" / "mm" / model_sha
        # Directory may not exist at all, but if it does it must be empty.
        if mm_root.is_dir():
            assert not any(mm_root.iterdir()), (
                "promotion wrote to MM even though muscle_memory=off"
            )

    def test_promotion_is_noop_on_empty_text(
        self, bench_env: Path, tmp_path: Path
    ) -> None:
        model = tmp_path / "empty.gguf"
        model.write_bytes(b"GGUF\x00empty-fixture")
        model_sha = hashlib.sha256(b"empty-fixture").hexdigest()
        mf = _make_manifest(model, "empty-text prompt", model_sha=model_sha)

        gguf._promote_rescue_to_r0(
            mf,
            prompt="empty-text prompt",
            text="",
            tokens=0,
            model_sha=model_sha,
            wall_s=0.001,
        )

        key = gguf._muscle_memory_key(mf)
        path = bench_env / "cache" / "mm" / model_sha / f"{key:016x}.mm"
        assert not path.exists()

    def test_promotion_stamps_promoted_from_marker_on_zpm_entry(
        self, bench_env: Path, tmp_path: Path
    ) -> None:
        """R1 promotion must mark the source so audits can separate
        organic R5 writebacks from R2-rescue promotions.
        """
        from nrlpy import zpm as _zpm

        model = tmp_path / "zpm.gguf"
        model.write_bytes(b"GGUF\x00zpm-fixture")
        model_sha = hashlib.sha256(b"zpm-fixture").hexdigest()
        rephrased_prompt = "Describe the R2 rescue to R0 promotion."
        served_reply = "Rescue-served reply promoted to R1 via Phase 11."
        mf = _make_manifest(model, rephrased_prompt, model_sha=model_sha)

        gguf._promote_rescue_to_r0(
            mf,
            prompt=rephrased_prompt,
            text=served_reply,
            tokens=max(1, len(served_reply.split())),
            model_sha=model_sha,
            wall_s=0.009,
        )

        idx_path = gguf._zpm_index_path(model_sha)
        assert idx_path.is_file(), "ZPM index was not written"
        idx = _zpm.ZpmIndex.load(idx_path)
        stamps = [dict(entry.metadata) for entry in idx]
        promoted = [
            s for s in stamps if s.get("promoted_from") == "r2_ngram_rescue"
        ]
        assert len(promoted) == 1, (
            f"expected exactly one r2_ngram_rescue-stamped entry; got {len(promoted)}"
        )
        assert promoted[0].get("prompt_head") == rephrased_prompt[:256]
