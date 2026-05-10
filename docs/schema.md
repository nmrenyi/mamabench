# mamabench Schema

Each mamabench artifact is JSONL: one normalized benchmark item per line. The
schema is intentionally larger than the minimum needed to answer one question.
For a plain MCQ, the model-facing content is only `question`, `choices`, and the
correct answer. The rest of the fields make the row usable as a benchmark: they
support scoring, filtering, auditing, reproducibility, release management, and
future open-ended or safety tasks.

The examples below use the first MedMCQA row exported to
`data/processed/v0.1/medmcqa_first.json`.

## Field Guide

| Field | Example | Explanation and rationale |
| --- | --- | --- |
| `id` | `mamabench_v0.1_mcq_medmcqa_000dd38b-...` | Stable mamabench row identifier. This lets manifests, validation reports, error analyses, and future perturbation rows refer to the same item without depending on a source dataset's ID format. |
| `schema_version` | `0.1` | Version of the canonical mamabench schema. This lets downstream code know how to interpret the row if the schema evolves. |
| `set_type` | `mcq` | Question format. Current values are `mcq`, `open_ended`, and `safety`. This tells the evaluator which scoring path to use. |
| `source_dataset` | `MedMCQA` | Human-readable source dataset name. This supports source-level filtering and license/provenance review. |
| `source_id` | `000dd38b-1d32-4390-9840-27452bd2e383` | Original row ID from the source dataset, when available. This keeps the normalized row traceable back to the raw source. |
| `question` | `Best diagnosis of ovulation is by :` | The model-facing prompt. This is one of the essential fields for any benchmark item. |
| `clinical_domain` | `obgyn` | Normalized medical domain label, such as `obgyn`, `neonatal`, `infant`, `pediatric`, or `reproductive`. This lets us evaluate performance by MAMAI-relevant clinical area. |
| `age_group` | `adult` | Normalized patient age or population group, such as `maternal`, `neonate`, `infant`, `child`, or `adult`. This supports clinically meaningful slices that do not always match the broad domain. |
| `task_type` | `diagnosis` | Normalized task label, such as diagnosis, treatment, triage, dosage, prevention, factual lookup, or safety. This lets us report where the model succeeds or fails beyond overall accuracy. |
| `safety_type` | `null` | Safety subtype for safety-focused rows. It is `null` for ordinary MCQs like the MedMCQA example. |
| `choices` | `["Ultrasound", "Laproscopy", ...]` | Normalized answer choices for MCQ rows. This is one of the essential MCQ fields and is `null` for non-MCQ rows. |
| `answer` | `Ultrasound` | Normalized correct answer used by scorers. For MCQs, this must be the full choice text, not a source letter key. |
| `answer_index` | `0` | Zero-based index into `choices`. This removes ambiguity and gives evaluators a simple way to compare model outputs by option index. |
| `source_answer` | `A` | Raw source answer key or value. MedMCQA stores the answer as a letter, so we preserve `A` here while keeping `answer` normalized to `Ultrasound`. |
| `rubric` | `null` | Scoring rubric for open-ended or safety rows. It is `null` for ordinary MCQs. Keeping the field present makes all rows structurally consistent. |
| `tags` | `["medmcqa", "gynaecology_obstetrics", "choice_type:single"]` | Lightweight searchable labels. Tags are useful for quick filtering without changing the controlled schema. |
| `icd10_codes` | `[]` | Optional ICD-10 labels. Most current rows do not have them, but the field leaves room for clinically coded evaluation slices later. |
| `perturbation_of` | `null` | ID of the original row if this item is a perturbed variant, such as a rewritten safety challenge. It is `null` for source-original rows. |
| `perturbation_type` | `null` | Type of perturbation, if `perturbation_of` is set. This lets us evaluate robustness by perturbation class. |
| `contamination_risk` | `high` | Estimate of whether the row may have appeared in model pretraining or public benchmark exposure. Public datasets like MedMCQA are marked high. |
| `license` | `Apache-2.0` | Source data license. This is required for release decisions and downstream reuse. |
| `provenance` | object | Structured source metadata. It records where the row came from and source-specific details needed for audit and regeneration. |
| `split` | `test` | mamabench split label, such as `dev`, `test`, or `pilot`. This is separate from the source dataset's split, which is preserved in `provenance.source_split`. |

## MCQ Answer Fields

For MCQs, the essential scoring relationship is:

```text
answer == choices[answer_index]
```

In the MedMCQA example:

```json
{
  "choices": [
    "Ultrasound",
    "Laproscopy",
    "Endometrial biopsy",
    "Chromotubation"
  ],
  "answer": "Ultrasound",
  "answer_index": 0,
  "source_answer": "A"
}
```

`answer` is what the benchmark scorer should trust. `source_answer` is only an
audit field. Keeping both avoids overloading one field with two meanings.

## Provenance

Every row has a `provenance` object with these common fields:

| Field | MedMCQA example | Rationale |
| --- | --- | --- |
| `source_url` | `https://huggingface.co/datasets/openlifescienceai/medmcqa` | Points reviewers to the original dataset or dataset card. |
| `source_split` | `train` | Preserves the original source split even when mamabench assigns its own `split`. |
| `source_version` | `obgyn-qa-collection@71433e4` | Identifies the local source snapshot used to generate the row. |

Adapters may also preserve source-specific provenance fields when they are useful
for audit or error analysis. The MedMCQA adapter currently adds:

| Field | MedMCQA example | Rationale |
| --- | --- | --- |
| `source_subject` | `Gynaecology & Obstetrics` | Original subject label used by the adapter for domain classification. |
| `source_topic` | `null` | Original topic label, when present. Useful for diagnosing classification and coverage. |
| `source_choice_type` | `single` | Original choice type. Confirms whether the row is a single-answer MCQ. |
| `source_explanation` | `Ultrasound` | Original explanation or answer note. Useful for audit, but not used as the canonical answer. |

The common provenance fields are required by validation. Adapter-specific fields
are allowed so that source evidence is not lost during normalization.
