#!/usr/bin/env python3
"""Derive HealthBench `consensus` classification verdicts from `oss_eval`.

HealthBench Consensus is a 3,671-prompt subset of HealthBench Eval — the
prompts overlap entirely with the 5,000 oss_eval prompts (the consensus
subset is the rows whose physician-written cluster rubrics applied; the
prompts themselves are identical). Re-classifying them would just re-run
the same prompts and waste compute, so this script reuses the existing
oss_eval verdicts and emits a consensus-labeled JSONL.

Inputs:
  --oss-eval-verdicts  JSONL produced by classify_obgyn.py for HealthBench
                       oss_eval (5,000 rows).
  --consensus-source   HealthBench Consensus source JSONL; we read its
                       prompt_ids to know which rows belong to consensus.

Output:
  --output             JSONL containing the subset of oss_eval verdicts
                       whose row_id matches a consensus prompt_id, with
                       the ``source`` field rewritten to ``"consensus"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oss-eval-verdicts",
        type=Path,
        default=Path("benchmark/v0.2/classification_verdicts/healthbench_oss_eval.jsonl"),
        help="oss_eval verdicts JSONL (default: the v0.2 path).",
    )
    parser.add_argument(
        "--consensus-source",
        type=Path,
        required=True,
        help="HealthBench Consensus source JSONL (e.g. consensus_2025-05-09-20-00-46.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v0.2/classification_verdicts/healthbench_consensus.jsonl"),
        help="Where to write the consensus verdicts JSONL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.oss_eval_verdicts.is_file():
        print(f"error: --oss-eval-verdicts not found: {args.oss_eval_verdicts}")
        return 2
    if not args.consensus_source.is_file():
        print(f"error: --consensus-source not found: {args.consensus_source}")
        return 2

    consensus_ids: set[str] = set()
    with args.consensus_source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            consensus_ids.add(json.loads(line)["prompt_id"])

    verdicts_by_id: dict[str, dict] = {}
    with args.oss_eval_verdicts.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if isinstance(v.get("row_id"), str):
                verdicts_by_id[v["row_id"]] = v

    overlap = consensus_ids & set(verdicts_by_id)
    missing = consensus_ids - set(verdicts_by_id)
    if missing:
        print(
            f"warning: {len(missing)} consensus prompt_ids have no oss_eval "
            f"verdict (e.g. {sorted(missing)[:3]}). They will be skipped — "
            f"re-run classify_obgyn for the missing rows separately."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for cid in consensus_ids:
            v = verdicts_by_id.get(cid)
            if v is None:
                continue
            rewritten = dict(v)
            rewritten["source"] = "consensus"
            out.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"derived {written} consensus verdicts from {len(verdicts_by_id)} "
        f"oss_eval verdicts (overlap: {len(overlap)}/{len(consensus_ids)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
