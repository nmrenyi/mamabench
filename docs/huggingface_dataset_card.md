---
language:
- en
license: other
license_name: per-source-see-card
pretty_name: mamabench
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- multiple-choice
tags:
- medical
- mcq
- obgyn
- obstetrics
- gynecology
- pediatrics
- reproductive-health
- maternal-health
- africa
- benchmark
- evaluation
source_datasets:
- extended|openlifescienceai/medmcqa
- extended|jind11/MedQA
- extended|intronhealth/afrimedqa_v2
configs:
- config_name: default
  default: true
  data_files:
  - split: test
    path: data/*.jsonl
- config_name: medmcqa
  data_files:
  - split: test
    path: data/medmcqa.jsonl
- config_name: medqa_usmle
  data_files:
  - split: test
    path: data/medqa_usmle.jsonl
- config_name: afrimedqa
  data_files:
  - split: test
    path: data/afrimedqa.jsonl
---

# mamabench

[GitHub](https://github.com/nmrenyi/mamabench) · 20,067 single-answer MCQ rows · schema `v0.3` · release `v0.1`

A normalized OBGYN / pediatrics / reproductive-health benchmark for evaluating end-to-end medical question-answering systems. Originally built to evaluate **MAMAI**, a Gemma 4 E4B + RAG medical-advice chatbot for nurses and midwives in Zanzibar.

## Release `v0.1`

This release covers the **multiple-choice track**: 20,067 single-answer MCQs sourced from three publicly-available medical QA benchmarks, filtered to OBGYN / pediatrics / reproductive-health, normalized to a minimal canonical schema, and end-to-end validated. Planned future releases will add open-ended and safety-focused tracks.

### At a glance

| Source | Rows | License | Notes |
|---|---|---|---|
| MedMCQA (OBGYN + Pediatrics subset) | 18,508 | Apache-2.0 | Indian AIIMS / NEET PG entrance exams |
| MedQA-USMLE (OBGYN subset) | 1,025 | MIT | USMLE-style board questions |
| AfriMed-QA (OBGYN single-answer) | 534 | **CC BY-NC-SA 4.0 (non-commercial)** | Pan-African expert exam questions |
| **Total** | **20,067** | mixed — see below | |

## License — read this before use

This dataset combines rows from three sources with different licenses. **Each row's `source.dataset` field identifies its license**; the top-level `license: other` reflects this mix rather than choosing one. There is no umbrella license overriding the per-source licenses.

- **`source.dataset == "MedMCQA"`** — Apache-2.0. Commercial use OK.
- **`source.dataset == "MedQA-USMLE"`** — MIT. Commercial use OK.
- **`source.dataset == "AfriMed-QA"`** — **CC BY-NC-SA 4.0**. Non-commercial use only; derivative works must share-alike.

If your use case requires commercial use, filter out AfriMed-QA rows or load only the permissive subsets:

```python
from datasets import load_dataset

# Option 1: filter at load time
ds = load_dataset("nmrenyi/mamabench")
permissive = ds.filter(lambda row: row["source"]["dataset"] != "AfriMed-QA")

# Option 2: load only the permissive configs
ds_medmcqa = load_dataset("nmrenyi/mamabench", "medmcqa")        # Apache-2.0
ds_usmle = load_dataset("nmrenyi/mamabench", "medqa_usmle")      # MIT
```

## Loading

```python
from datasets import load_dataset

# All 20,067 rows
ds = load_dataset("nmrenyi/mamabench")

# Single source
ds = load_dataset("nmrenyi/mamabench", "medmcqa")      # 18,508 rows
ds = load_dataset("nmrenyi/mamabench", "medqa_usmle")  # 1,025 rows
ds = load_dataset("nmrenyi/mamabench", "afrimedqa")    # 534 rows

# Pin to a specific release for reproducible evaluation
ds = load_dataset("nmrenyi/mamabench", revision="v0.1")
```

## Schema

Each row is a JSON object with the following fields (schema `v0.3`):

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable mamabench row id, e.g. `mamabench_v0.1_medmcqa_<source-id-or-hash>` |
| `schema_version` | string | Always `"0.3"` in this release |
| `set_type` | string | Always `"mcq"` in this release |
| `question` | string | The model-facing question |
| `choices` | list[string] | The model-facing answer options (≥ 2 entries) |
| `answer` | string | Normalized correct answer text. Always equals `choices[answer_index]` |
| `answer_index` | int | Zero-based index of the correct choice |
| `source` | object | `{dataset, id, answer}` for row-level provenance |

The full schema rationale lives in `schema/mamabench_v0.3.md`; the machine-readable JSON Schema in `schema/mamabench_v0.3.schema.json`.

## Per-source provenance and manifests

Each source has its own manifest under `manifests/`:

- `medmcqa_manifest.json`
- `medqa_usmle_manifest.json`
- `afrimedqa_manifest.json`
- `release_manifest.json` (aggregate across all three)

Manifests record the exact `obgyn-qa-collection` commit hash the data was extracted from (so the artifact is reproducible byte-for-byte), the full validation report (0 issues across all three artifacts), license and source URLs, and — for AfriMed-QA — filter counts (multi-answer rows skipped, ambiguous-position rows skipped, embedded-prefix rows cleaned).

## AfriMed-QA caveats

The AfriMed-QA subset went through more invasive normalization than the other two. Summary:

- **660 → 534 rows kept (81% retention).** 112 multi-answer rows were skipped (the v0.3 schema represents single-answer MCQs); 14 rows with ambiguous answer positions were dropped (the correct option's text appeared at multiple choice positions, making them unscorable as MCQ).
- **27 rows had embedded letter prefixes in their option text** from an upstream extraction quirk; those were stripped in-place so the rendered MCQ doesn't show doubled labels.
- A handful of remaining rows have cosmetic upstream artifacts that don't affect letter-based scoring. See the [GitHub README's "AfriMed-QA data quality notes"](https://github.com/nmrenyi/mamabench#afrimed-qa-data-quality-notes) for the full breakdown.

Upstream data-quality issues we surfaced during construction are tracked at the upstream repo: <https://github.com/nmrenyi/obgyn-qa-collection/issues>.

## Contamination caveat

MedMCQA and MedQA-USMLE are present in many model pretraining corpora and should be treated as **high contamination risk** for evaluating proprietary or web-trained models. AfriMed-QA is newer and less likely to be in pretraining corpora. The current release does not flag contamination risk per row; consumers should account for this in their evaluation methodology.

## Versioning

Releases are git tags on this dataset's HF repo. Pin a version for reproducible evaluation:

```python
load_dataset("nmrenyi/mamabench", revision="v0.1")
```

Schema version is tracked independently — every row carries `schema_version` so consumers can detect and adapt to schema changes safely across release versions.

## Building locally

The pipeline that builds this dataset is open source at <https://github.com/nmrenyi/mamabench>. Per-source adapter scripts, validators, manifest builders, and the release-manifest aggregator are all available. See the GitHub README for full instructions.

## Source datasets

- **MedMCQA** — Pal et al. 2022 ([paper](https://proceedings.mlr.press/v174/pal22a), [HF dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa))
- **MedQA** — Jin et al. 2020 ([paper](https://arxiv.org/abs/2009.13081), [repo](https://github.com/jind11/MedQA))
- **AfriMed-QA** — Olatunji et al. 2024 ([paper](https://arxiv.org/abs/2411.15640), [HF dataset](https://huggingface.co/datasets/intronhealth/afrimedqa_v2))

The OBGYN / pediatrics / reproductive-health filtering was done in [obgyn-qa-collection](https://github.com/nmrenyi/obgyn-qa-collection); mamabench normalizes those pre-filtered outputs into the canonical schema.

## Citation

If you use mamabench, please link to this dataset page (<https://huggingface.co/datasets/nmrenyi/mamabench>) and cite the upstream sources whose data is included:

```bibtex
@inproceedings{pal2022medmcqa,
  title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={Conference on Health, Inference, and Learning},
  pages={248--260},
  year={2022},
  organization={PMLR}
}

@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}

@article{olatunji2024afrimed,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Olatunji, Tobi and Nimo, Charles and others},
  journal={arXiv preprint arXiv:2411.15640},
  year={2024}
}
```
