# Prompt Registry

This directory holds versioned prompt templates for the clinical RAG system. Prompts are treated as **code**: versioned, documented, and evaluated. Each change is a new file with a rationale, so prompt evolution is auditable and every version can be compared against the others in the eval suite.

## Naming convention

`v{N}_{short_slug}.yaml`

- `{N}` - monotonically increasing version number (`v1`, `v2`, `v3`, …)
- `{short_slug}` - a few words describing the version's distinguishing change

Examples: `v1_baseline.yaml`, `v2_cited.yaml`, `v3_clinical_guard.yaml`.

## File structure

Each prompt file contains the system template plus metadata:

| Field | Purpose |
|---|---|
| `version` | Must match the filename version (e.g. `v1_baseline`) |
| `metadata.author` | Who created or last changed the prompt |
| `metadata.created` | ISO date (`YYYY-MM-DD`) |
| `metadata.rationale` | Why this version exists / what changed from the prior one |
| `system_template` | The prompt itself, with `{context}` and `{question}` placeholders |

## Which version is active?

The active prompt is selected in `config/generation.yaml`:

```yaml
prompt:
  active_version: "v1_baseline"   # filename stem, without .yaml
  directory: "prompts"
```

Changing `active_version` swaps the prompt with no code change.

## Version history

| Version | Date | Summary |
|---|---|---|
| `v1_baseline` | 2026-05-10 | Initial baseline: grounded answers with source citations |

_I'll add a row here whenever a new version is introduced._
