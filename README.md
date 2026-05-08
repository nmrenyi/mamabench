# mamabench

`mamabench` builds normalized QA benchmark artifacts for evaluating the end-to-end
MAMAI system: a Gemma 4 E4B + RAG medical-advice chatbot for nurses and midwives
in Zanzibar.

This repository is scoped to benchmark construction, validation, and lightweight
summaries. It does not implement production retrieval, retrieval-label generation,
model serving, or full model evaluation.

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

## Validate sample data

```bash
python3 scripts/validate_mamabench.py data/samples/sample.jsonl
```

The command prints a JSON validation report and exits with status `0` when the
file is valid.

## Summarize sample data

```bash
python3 scripts/summarize_mamabench.py data/samples/sample.jsonl
```

The command prints a manifest-style JSON summary with item counts by set type,
source dataset, clinical domain, age group, task type, safety type,
contamination risk, and split.

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

