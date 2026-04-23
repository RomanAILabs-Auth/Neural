# NRL JSON Schemas (control plane)

| File | Purpose |
|------|---------|
| `workload_descriptor_v1.schema.json` | Identity fields for benchmark / run deduplication. |
| `immune_event_v1.schema.json` | Immune ladder events (JSONL). |
| `specialization_manifest_v1.schema.json` | Stub for promoted specialization blobs. |
| `learn_config_v1.schema.json` | Byte cap for bounded vocabulary store (`build/nrlpy_learn`). |
| `control_preferences_v1.schema.json` | Operator hints from `nrl control` (`build/control/preferences.json`). |
| `control_audit_line_v1.schema.json` | One JSON object per line in `build/control/control_audit.jsonl`. |

Normative prose: [`../nrl_alive_language_evolution_architecture.md`](../nrl_alive_language_evolution_architecture.md).
