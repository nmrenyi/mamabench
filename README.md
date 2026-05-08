# mamabench

`mamabench` builds normalized QA benchmark artifacts for evaluating the end-to-end
MAMAI system: a Gemma 4 E4B + RAG medical-advice chatbot for nurses and midwives
in Zanzibar.

This repository is scoped to benchmark construction, validation, and lightweight
summaries. It does not implement production retrieval, retrieval-label generation,
model serving, or full model evaluation.

Generated benchmark artifacts under `data/processed/` are not tracked in Git.
They should be regenerated from adapter scripts during development and published
as versioned dataset releases on Hugging Face Datasets when stable.

## Current status

This is the Step 1 foundation:

- canonical JSONL schema definitions
- validation utilities
- manifest/summarization utilities
- CLI wrappers for validation and summarization
- a tiny synthetic sample file covering MCQ, open-ended, and safety items
- unit tests for valid and invalid rows

No external datasets are downloaded or adapted in this step.

## Schema

Each benchmark item is one JSON object per line. The schema version is `0.1`.
Common fields include:

- `id`
- `schema_version`
- `set_type`
- `source_dataset`
- `source_id`
- `question`
- `clinical_domain`
- `age_group`
- `task_type`
- `safety_type`
- `choices`
- `answer`
- `answer_index`
- `source_answer`
- `rubric`
- `tags`
- `icd10_codes`
- `perturbation_of`
- `perturbation_type`
- `contamination_risk`
- `license`
- `provenance`
- `split`

Supported `set_type` values are `mcq`, `open_ended`, and `safety`.

For MCQ-style rows, `answer` is the normalized full correct answer text and is
safe for downstream scorers. When `answer_index` is present, `answer` must equal
`choices[answer_index]`. If `answer_index` is omitted, `answer` must still match
one of the normalized choices. Use `source_answer` to preserve a source-native
answer key or raw answer value, such as `A`, `1`, or `null` when no meaningful
raw source answer exists.

Perturbation rows must include both `perturbation_of` and `perturbation_type`.
By default, validation allows `perturbation_of` to point outside the current
file because split artifacts such as `safety.jsonl` may reference originals in
`mcq.jsonl`. Use strict reference checking when validating a complete artifact
or when passing known target IDs.

## Validate sample data

```bash
python3 scripts/validate_mamabench.py data/samples/sample.jsonl
```

The command prints a JSON validation report and exits with status `0` when the
file is valid.

To require perturbation references to resolve against the current file:

```bash
python3 scripts/validate_mamabench.py --check-perturbation-refs data/samples/sample.jsonl
```

To validate a split file against IDs from another JSONL artifact:

```bash
python3 scripts/validate_mamabench.py \
  --known-ids-jsonl data/processed/v0.1/mcq.jsonl \
  data/processed/v0.1/safety.jsonl
```

## Summarize sample data

```bash
python3 scripts/summarize_mamabench.py data/samples/sample.jsonl
```

The command prints a manifest-style JSON summary with item counts by set type,
source dataset, clinical domain, age group, task type, safety type,
contamination risk, and split.

## Adapt MedMCQA

The MedMCQA adapter normalizes the already-filtered OBGYN/Pediatrics TSV from
`obgyn-qa-collection` into mamabench JSONL:

```bash
python3 scripts/adapt_medmcqa.py \
  /Users/renyi/Downloads/obgyn-qa-collection/medmcqa/data/obgyn_mcq.tsv \
  data/processed/v0.1/medmcqa.jsonl \
  --source-version obgyn-qa-collection@71433e4 \
  --manifest-output data/processed/v0.1/medmcqa_manifest.json \
  --validation-report-output data/processed/v0.1/medmcqa_validation_report.json
```

Adapter behavior:

- `answer` is the normalized full correct answer text.
- `answer_index` is derived from the source `correct_letter`.
- `source_answer` preserves the source letter key.
- `source_split`, `subject`, `topic`, `choice_type`, and explanations are
  preserved in provenance.
- all MedMCQA rows are tagged `contamination_risk: high`.
- broad Pediatrics rows are retained; rows that are not clearly neonatal or
  infant are labeled with `clinical_domain: unknown` and tagged `pediatrics`.

The command writes local generated files under `data/processed/`, which is
ignored by Git. Release-ready artifacts should be uploaded to Hugging Face
Datasets rather than committed to this repository.

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Next implementation step

After this foundation is stable, add exactly one MCQ source adapter and prove the
flow:

```text
raw source -> normalized JSONL -> validation -> manifest summary
```
