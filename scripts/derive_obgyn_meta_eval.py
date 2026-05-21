#!/usr/bin/env python3
"""Derive an OBGYN-scoped slice of HealthBench's grader meta-evaluation set.

HealthBench ships a meta-evaluation file (``*_oss_meta_eval.jsonl``) used to
evaluate the *grader* rather than the model under test: each row is a
``(conversation, completion, single rubric criterion)`` triple carrying
independent physician ``binary_labels`` (criterion met / not met). It is
judge-calibration data, not benchmark questions — so it does NOT belong in
mamabench's loadable configs or row schema. It is shipped instead as a
standalone side-file under ``benchmark/v0.2/calibration/`` (mirrored to
``calibration/`` in the HF release).

This script filters the full meta_eval set down to the maternal / neonatal /
child / SRH scope of mamabench, so consumers can validate their rubric-track
LLM judge on the criteria mamabench actually contains, rather than on
general medicine. The filter reuses the existing OBGYN classifier verdicts:
a meta_eval row is kept iff its ``prompt_id`` has a non-NONE verdict in the
oss_eval classifier output (``prompt_id`` == classifier ``row_id``).

No new classification run is needed — same join pattern as
``derive_consensus_verdicts.py``.

Inputs:
  --meta-eval-source   HealthBench meta_eval JSONL (e.g.
                       2025-05-07-06-14-12_oss_meta_eval.jsonl).
  --oss-eval-verdicts  oss_eval classifier verdicts JSONL (carries the
                       category per prompt_id, including NONE).

Output:
  --output             JSONL of the OBGYN-scoped meta_eval rows. HealthBench's
                       native meta_eval fields are preserved verbatim; one
                       field, ``mamabench_obgyn_category``, is added so
                       consumers can slice by category without a second join.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-eval-source",
        type=Path,
        required=True,
        help="HealthBench meta_eval JSONL (e.g. 2025-05-07-06-14-12_oss_meta_eval.jsonl).",
    )
    parser.add_argument(
        "--oss-eval-verdicts",
        type=Path,
        default=Path("benchmark/v0.2/classification_verdicts/oss_eval.jsonl"),
        help="oss_eval classifier verdicts JSONL (default: the v0.2 path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v0.2/calibration/obgyn_meta_eval.jsonl"),
        help="Where to write the OBGYN-scoped meta_eval JSONL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.meta_eval_source.is_file():
        print(f"error: --meta-eval-source not found: {args.meta_eval_source}")
        return 2
    if not args.oss_eval_verdicts.is_file():
        print(f"error: --oss-eval-verdicts not found: {args.oss_eval_verdicts}")
        return 2

    # prompt_id -> category (including NONE) from the classifier verdicts.
    category_by_id: dict[str, str] = {}
    with args.oss_eval_verdicts.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            row_id = v.get("row_id")
            if isinstance(row_id, str):
                category_by_id[row_id] = v.get("category", "NONE")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    unmatched = 0
    kept_prompts: set[str] = set()
    cat_counts: Counter[str] = Counter()

    with args.meta_eval_source.open(encoding="utf-8") as f, args.output.open(
        "w", encoding="utf-8"
    ) as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            pid = row.get("prompt_id")
            category = category_by_id.get(pid)
            if category is None:
                # prompt_id not in the classified set (e.g. the documented
                # 7 oss_eval rows the classifier did not converge on).
                unmatched += 1
                continue
            if category == "NONE":
                continue
            row["mamabench_obgyn_category"] = category
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            kept_prompts.add(pid)
            cat_counts[category] += 1

    print(
        f"derived {kept} OBGYN-scoped meta_eval rows "
        f"({len(kept_prompts)} unique prompts) from {total} total meta_eval rows"
    )
    if unmatched:
        print(
            f"note: {unmatched} meta_eval rows had a prompt_id absent from the "
            f"classifier verdicts (unclassified prompts) — skipped"
        )
    for cat, n in cat_counts.most_common():
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
