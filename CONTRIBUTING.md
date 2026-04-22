# Contributing to NRL

NRL is architecture-driven. Read `nrl-architecture.md` first.

## Rules

- Keep hot-path work machine-code-first.
- Do not bypass benchmark governance.
- Keep adaptive behavior behind explicit, documented controls.
- Add tests for every functional change.
- Preserve attribution and licensing markers.

## Workflow

1. Align proposed changes with architecture sections/decisions.
2. Implement smallest correct increment.
3. Run formatting/lint/tests/bench checks.
4. Document behavior/perf impact clearly.
