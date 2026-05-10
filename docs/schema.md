# mamabench Schema

The current in-use schema version is `0.2`. The code source of truth is
`SCHEMA_VERSION` in `src/mamabench/schema.py`. The machine-readable schema is
`schemas/mamabench_v0.2.schema.json`.

Each mamabench artifact is JSONL: one normalized benchmark item per line. Version
`0.2` is intentionally minimal and currently supports MCQ rows only. Labels such
as clinical domain, age group, task type, tags, contamination risk, and benchmark
split are not part of the canonical row yet. We can add them in a later schema
version when we have a concrete evaluator or labeling policy that needs them.

## Example

```json
{
  "id": "mamabench_v0.2_medmcqa_000dd38b-1d32-4390-9840-27452bd2e383",
  "schema_version": "0.2",
  "set_type": "mcq",
  "question": "Best diagnosis of ovulation is by :",
  "choices": [
    "Ultrasound",
    "Laproscopy",
    "Endometrial biopsy",
    "Chromotubation"
  ],
  "answer": "Ultrasound",
  "answer_index": 0,
  "source": {
    "dataset": "MedMCQA",
    "id": "000dd38b-1d32-4390-9840-27452bd2e383",
    "url": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
    "license": "Apache-2.0",
    "answer": "A"
  }
}
```

## Top-Level Fields

| Field | Required | Explanation and rationale |
| --- | --- | --- |
| `id` | yes | Stable mamabench row identifier. This lets validation reports, manifests, and error analyses refer to the same item without depending only on a source dataset's ID format. |
| `schema_version` | yes | Canonical schema version. Current value is `0.2`. This lets downstream code reject rows from an incompatible schema. |
| `set_type` | yes | Question format. Current supported value is `mcq`. |
| `question` | yes | The model-facing question. |
| `choices` | yes | The model-facing answer options. Must be a list of at least two non-empty strings. |
| `answer` | yes | Normalized full correct answer text used by scorers. For MCQs, this is the choice text, not the source letter key. |
| `answer_index` | yes | Zero-based index into `choices`. This removes ambiguity and gives evaluators a simple way to score by option index. |
| `source` | yes | Minimal source/audit object. It records where the item came from and the license needed for release decisions. |

For MCQs, the validation invariant is:

```text
answer == choices[answer_index]
```

## Source Fields

| Field | Required | Explanation and rationale |
| --- | --- | --- |
| `source.dataset` | yes | Original dataset name, such as `MedMCQA`. |
| `source.id` | yes | Original source row ID. May be `null` only when the source has no row identifier. |
| `source.url` | yes | URL for the source dataset or dataset card. |
| `source.license` | yes | Source dataset license. |
| `source.answer` | no | Original source answer key or value, such as `A` for MedMCQA. This is kept for audit; scorers should use top-level `answer` and `answer_index`. |

The v0.2 validator rejects unexpected top-level and source fields. This is
deliberate: adding new canonical fields should be an explicit schema decision.

## Removed From v0.1

Version `0.2` removes the earlier broad labels and placeholders:

```text
clinical_domain
age_group
task_type
safety_type
rubric
tags
icd10_codes
perturbation_of
perturbation_type
contamination_risk
split
provenance
source_dataset
source_id
source_answer
license
```

Those concepts may still be useful later, but they should be reintroduced only
when we can justify the field and document how it is generated.
