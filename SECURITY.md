<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Security Policy

NRL is released under the **RomanAILabs Proprietary Source-Available Evaluation
License 1.0**. Private vulnerability reproduction and private disclosure are
permitted only as part of local evaluation under that license. Commercial use,
competitive use, redistribution, AI training, hosted use, or derivative services
require a separate written license from RomanAILabs.

## Supported Versions

Security fixes target the current public release branch and active development
branch. Older experimental branches may not receive patches.

## Reporting A Vulnerability

Please report vulnerabilities privately by email:

- `daniel@romanailabs.com`
- `romanailabs@gmail.com`

Use the subject line:

```text
SECURITY: NRL vulnerability report
```

Include:

- Affected commit or release.
- Operating system and Python version.
- Minimal reproduction steps.
- Impact assessment.
- Whether the report involves local files, untrusted model input, command execution, model artifacts, or secrets.

## Responsible Disclosure

Please do not publish exploit details until RomanAILabs has acknowledged the
report and had a reasonable opportunity to investigate. We aim to acknowledge
reports within 7 days and provide a remediation plan or status update within
30 days when the issue is reproducible.

## Security Boundaries

NRL is local-first software. Treat GGUF files, `.nrl` manifests, Python scripts,
and shell commands as untrusted input unless you control their source.

`NRL_SAFE_MODE=1` is the emergency read-mostly mode: it disables background Learn
Mode, WAL writes, and auto-prune hooks. It is not a sandbox and does not make
untrusted code safe to execute.

See [`LICENSE`](./LICENSE) for the full proprietary source-available terms,
restrictions, patent reservation, attribution requirements, and commercial
licensing path.
