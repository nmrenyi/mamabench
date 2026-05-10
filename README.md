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

The repository currently includes:

- canonical JSONL schema definitions
- validation utilities
- manifest/summarization utilities
- CLI wrappers for validation and summarization
- a tiny synthetic MCQ sample file
- a MedMCQA adapter for the filtered OBGYN/Pediatrics source file
- unit tests for valid and invalid rows

External datasets are not downloaded by this repository. Local processed outputs
are regenerated from adapter scripts and ignored by Git.

## Schema

The current in-use schema version is `0.2`. The code source of truth is
`SCHEMA_VERSION` in `src/mamabench/schema.py`. A machine-readable copy of the
current row shape is kept in `schemas/mamabench_v0.2.schema.json`.

The current benchmark artifact release is `v0.1`. Benchmark version and schema
version are separate: row IDs include the benchmark release, while each row's
`schema_version` describes the JSON row shape.

Each benchmark item is one JSON object per line.
See [docs/schema.md](docs/schema.md) for the explanation and rationale for each
field. Common fields include:

- `id`
- `schema_version`
- `set_type`
- `question`
- `choices`
- `answer`
- `answer_index`
- `source`

Supported `set_type` value is currently `mcq`.

For MCQ rows, `answer` is the normalized full correct answer text and must equal
`choices[answer_index]`. Source-native answer keys, such as `A` or `1`, are
preserved as `source.answer` when available.

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
source dataset, source license, and validation status.

## Adapt MedMCQA

The MedMCQA adapter normalizes the already-filtered OBGYN/Pediatrics TSV from
`obgyn-qa-collection` into mamabench JSONL:

```bash
python3 scripts/adapt_medmcqa.py \
  /Users/renyi/Downloads/obgyn-qa-collection/medmcqa/data/obgyn_mcq.tsv \
  data/processed/v0.1/medmcqa.jsonl \
  --manifest-output data/processed/v0.1/medmcqa_manifest.json \
  --validation-report-output data/processed/v0.1/medmcqa_validation_report.json
```

Adapter behavior:

- `answer` is the normalized full correct answer text.
- `answer_index` is derived from the source `correct_letter`.
- `source.answer` preserves the source letter key.
- `source` contains only minimal audit metadata: dataset, original row id, URL,
  license, and source answer.

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
