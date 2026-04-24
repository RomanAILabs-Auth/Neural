<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Bio-Digital Brain v3.0 Release Checklist

Use this checklist immediately before publishing the GitHub release.

## Required Gates

- [ ] Run `cd nrlpy && python -m pytest tests/ -q --tb=no`.
- [ ] Run `cd nrlpy && python -m nrlpy doctor` and confirm `status: healthy`.
- [ ] Run `cd nrlpy && python -m nrlpy --version` and record the displayed version.
- [ ] Run all four Bio-Digital Brain examples without errors:
  - [ ] `python examples/bdb_chat.py`
  - [ ] `python examples/bdb_coverage_check.py`
  - [ ] `python examples/bdb_overnight_learning.py`
  - [ ] `python examples/bdb_prune_demo.py`
- [ ] Verify `nrlpy lmo info <model-or-sha>` works against an absorbed fixture or release model.
- [ ] Verify no legacy permissive-license text remains with the release audit scan.

## Documentation

- [ ] `AUDIT_REPORT.md` confirms P1-P9 are complete and integrated.
- [ ] `README.md` shows the MVP Complete badge and P1-P10 status.
- [ ] `README.md` includes the architecture summary and What's Next section.
- [ ] `Bio-Digital-Blueprint.MD` §11 includes the P10 completion status.
- [ ] `RELEASE_NOTES_v3.0.md` is reviewed for accuracy.
- [ ] `docs/README.md`, `docs/GETTING_STARTED.md`, and `docs/FUTURE_PHASES.md` are linked and current.

## License And Compliance

- [ ] `LICENSE` contains only the proprietary source-available evaluation terms.
- [ ] `LICENSES/NOTICE` matches the proprietary source-available release posture.
- [ ] `CONTRIBUTING.md` includes release tag guidance and contribution license terms.
- [ ] No generated binaries, model weights, LMO cache data, secrets, or benchmark scratch artifacts are staged.

## GitHub Release

- [ ] `.github/release.yml` is present for generated release-note categories.
- [ ] Open PR checks are green.
- [ ] Create an annotated release tag, for example:
  `git tag -a v3.0.0 -m "Bio-Digital Brain v3.0 - P10 audit passed"`.
- [ ] Push the tag after maintainer approval:
  `git push origin v3.0.0`.
- [ ] Create the GitHub release from the tag and review generated notes before publishing.

## Final Sign-Off

- [ ] P10 is complete.
- [ ] Bio-Digital Brain v3.0 is 100% complete.
- [ ] Architecture Audit passed.
- [ ] Project is GitHub Release Ready.
