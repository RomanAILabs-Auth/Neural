# .nrl Minimal Spec v0

This is a key-value control surface for deterministic runtime execution.

## Supported form

- UTF-8 text file
- one entry per line: `key=value`
- comments start with `#`
- blank lines are ignored

## Keys

- `mode`: `run` or `bench`
- `profile`: `sovereign|adaptive|war-drive|zpm|automatic|omega|omega-hybrid`
- `neurons`: unsigned integer (must be even for INT4 path)
- `iterations`: unsigned integer
- `reps`: unsigned integer (bench mode)
- `threshold`: integer in `[1, 15]`

## Execution

- Native: `nrl file path/to/program.nrl`
- Shortcut: `nrl path/to/program.nrl`
- Python: `nrlpy path/to/program.nrl`
