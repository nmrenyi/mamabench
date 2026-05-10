# mamabench Schema

The current in-use schema version is `0.3`. The code source of truth is
`SCHEMA_VERSION` in `src/mamabench/schema.py`. The machine-readable schema is
`schemas/mamabench_v0.3.schema.json`. That JSON Schema checks row structure
only; full semantic validation is implemented by `scripts/validate_mamabench.py`.

Each mamabench artifact is JSONL: one normalized benchmark item per line. Version
`0.3` is intentionally minimal and currently supports MCQ rows only. Labels such
as clinical domain, age group, task type, tags, contamination risk, and benchmark
split are not part of the canonical row yet. We can add them in a later schema
version when we have a concrete evaluator or labeling policy that needs them.

## Example

```json
{
  "id": "mamabench_v0.1_medmcqa_000dd38b-1d32-4390-9840-27452bd2e383",
  "schema_version": "0.3",
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
    "answer": "A"
  }
}
```

## Top-Level Fields

| Field | Required | Explanation and rationale |
| --- | --- | --- |
| `id` | yes | Stable mamabench row identifier. The version segment refers to the benchmark release, not the schema version. This lets validation reports, manifests, and error analyses refer to the same item without depending only on a source dataset's ID format. |
| `schema_version` | yes | Canonical schema version. Current value is `0.3`. This lets downstream code reject rows from an incompatible schema. |
| `set_type` | yes | Question format. Current supported value is `mcq`. |
| `question` | yes | The model-facing question. |
| `choices` | yes | The model-facing answer options. Must be a list of at least two non-empty strings. |
| `answer` | yes | Normalized full correct answer text used by scorers. For MCQs, this is the choice text, not the source letter key. |
| `answer_index` | yes | Zero-based index into `choices`. This removes ambiguity and gives evaluators a simple way to score by option index. |
| `source` | yes | Minimal source/audit object. Dataset-level metadata such as URL and license belongs in the manifest, not in each row. |

For MCQs, the validation invariant is:

```text
answer == choices[answer_index]
```

The Python validator also checks duplicate `id` values, duplicate
`source.dataset` + `source.id` pairs, `answer_index` bounds, and unexpected
fields. Use it for release validation rather than relying only on JSON Schema.

## Source Fields

| Field | Required | Explanation and rationale |
| --- | --- | --- |
| `source.dataset` | yes | Original dataset name, such as `MedMCQA`. |
| `source.id` | yes | Original source row ID. May be `null` only when the source has no row identifier. |
| `source.answer` | no | Original source answer key or value, such as `A` for MedMCQA. This is kept for audit; scorers should use top-level `answer` and `answer_index`. |

Dataset-level source metadata is stored once in the manifest:

```json
{
  "source_datasets": {
    "MedMCQA": {
      "url": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
      "license": "Apache-2.0"
    }
  }
}
```

The v0.3 validator rejects unexpected top-level and source fields. This is
deliberate: adding new canonical fields should be an explicit schema decision.

## Removed From Earlier Schemas

Version `0.3` keeps the v0.2 minimal MCQ shape and also removes row-level
dataset metadata:

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
source.url
source.license
```

Those concepts may still be useful later, but they should be reintroduced only
when we can justify the field and document how it is generated.
