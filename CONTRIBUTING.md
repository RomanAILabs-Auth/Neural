<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Contributing To NRL

Thanks for helping improve NRL and the Bio-Digital Brain MVP. This project is
local-first systems software, so correctness, reproducibility, safety, and
intellectual-property protection matter.

## License And Copyright Assignment

All contributions are made under the **RomanAILabs Proprietary
Source-Available Evaluation License 1.0** and the ownership terms below.

By submitting a pull request, patch, issue attachment, code snippet,
documentation change, design note, test, example, or any other contribution,
you agree to assign copyright in that contribution to RomanAILabs / Daniel
Harding, to the maximum extent permitted by law.

If your jurisdiction does not allow copyright assignment, you instead grant
RomanAILabs / Daniel Harding an irrevocable, perpetual, worldwide, exclusive,
royalty-free, sublicensable license to use, reproduce, modify, distribute,
commercialize, relicense, and enforce the contribution as part of NRL and
related RomanAILabs products.

Do not submit contributions unless you have the right to make this assignment
or grant.

## How To Submit A PR

1. Open an issue for non-trivial changes so scope can be discussed first.
2. Fork the repo and create a focused branch.
3. Keep PRs small: one feature, fix, or documentation improvement at a time.
4. Include tests for behavior changes.
5. Update docs when operator commands, environment variables, or output formats change.
6. Do not use a fork or branch to redistribute, host, commercialize, or build a
   competing implementation of NRL or the Bio-Digital Brain architecture.

## Code Style

- Python: type hints for new public functions, small modules, no hidden network calls.
- C: keep ABI boundaries stable; avoid allocations or feature checks in hot loops.
- Docs: keep claims measurable and label simulations clearly.
- File paths and commands should work on Windows first, with Linux/macOS where practical.

## Testing Requirements

Before opening a PR, run the narrow tests for your change. For the final MVP
lifecycle surface, run:

```bash
cd nrlpy
python -m pytest tests/test_bio_digital_brain_e2e.py -v
python -m nrlpy doctor
```

For persistence or pruning changes, also run:

```bash
python -m pytest tests/test_zpm_persist.py tests/test_lmo_disk_manager.py -v
```

## Release Tags

Release tags are created by maintainers after `RELEASE_CHECKLIST.md` is complete
and `AUDIT_REPORT.md` shows a passing P10 architecture audit.

- Use annotated tags in the form `vMAJOR.MINOR.PATCH`, for example `v3.0.0`.
- Tag only from the reviewed release branch after CI and local release gates pass.
- Include the release audit status in the tag message: `Bio-Digital Brain v3.0 - P10 audit passed`.
- Do not tag generated caches, model files, benchmark scratch data, or local LMO artifacts.
- GitHub release notes are generated from `.github/release.yml`; curate the final body before publishing.

## Safety And Claims

- Do not add cloud calls, telemetry, or remote execution without explicit review.
- Do not claim speedups unless the command, hardware, model, and output artifact are included.
- Keep `NRL_SAFE_MODE=1` behavior intact: it must disable background learning, WAL writes, and auto-prune hooks.
- Preserve attribution and licensing markers.
- Do not use repository contents for AI training, fine-tuning, retrieval,
  code-generation datasets, or competing implementation work without written
  permission.

## Review Checklist

- Tests pass locally.
- New files have clear names and narrow responsibility.
- User-facing commands are documented.
- Failure modes are safe by default.
- No generated binaries, caches, secrets, third-party proprietary code, or large model files are committed.
