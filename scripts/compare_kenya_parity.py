#!/usr/bin/env python3
"""Compare mamabench OBGYN classifier verdicts on Kenya source vignettes
against Kenya's existing Gemini-generated labels.

Inputs:
  --verdicts  JSONL produced by classify_obgyn.py for SOURCE=kenya
              (one row per classified vignette; row_id == Kenya StudyID).
  --kenya-tsv The 4-category-labeled TSV produced by Kenya's own filter:
              `obgyn-qa-collection/kenya-clinical-vignettes/data/obgyn_vignettes.tsv`.
              Columns: study_id, scenario, clinician_response, category.

Kenya's TSV only contains the 284 vignettes their classifier kept (positive
categories). Any StudyID present in our verdicts but absent from the TSV is
treated as Kenya saying NONE — i.e. their classifier filtered it out.

Mamabench's `SEXUAL_AND_REPRODUCTIVE_HEALTH` is reported by Kenya as `SRH`;
the script normalises the two so they match.

Outputs (printed to stdout):
  - Overall agreement percentage.
  - Per-category precision and recall.
  - 5x5 confusion matrix (Kenya rows, mamabench columns).
  - A capped list of disagreements for spot-checking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ["MATERNAL", "NEONATAL", "CHILD_HEALTH", "SEXUAL_AND_REPRODUCTIVE_HEALTH", "NONE"]


def kenya_to_mamabench_category(kenya_category: str) -> str:
    """Normalise Kenya's category names to mamabench's spelling."""
    if kenya_category == "SRH":
        return "SEXUAL_AND_REPRODUCTIVE_HEALTH"
    return kenya_category


def load_verdicts(path: Path) -> dict[str, str]:
    """Read a verdicts JSONL → {study_id: category}.

    Malformed lines, rows without a string ``row_id``, and rows with a
    category outside the known set are silently skipped — they shouldn't
    appear in a healthy verdicts file but resilience here avoids tripping
    on a single bad line in a long run.
    """
    verdicts: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("row_id")
            category = row.get("category")
            if not isinstance(row_id, str) or category not in CATEGORIES:
                continue
            verdicts[row_id] = category
    return verdicts


def load_kenya_labels(tsv_path: Path) -> dict[str, str]:
    """Read Kenya's filtered TSV → {study_id: kenya_category_normalised}.

    Only positive categories are present. Callers should treat anything not
    in this dict as Kenya saying NONE.
    """
    labels: dict[str, str] = {}
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("study_id")
            cat = row.get("category")
            if not sid or not cat:
                continue
            labels[str(sid)] = kenya_to_mamabench_category(cat)
    return labels


def compute_metrics(
    verdicts: dict[str, str],
    kenya_labels: dict[str, str],
) -> dict:
    """Compare our verdicts against Kenya's labels.

    For each study_id we classified, the expected (Kenya) category is either
    its TSV-recorded category, or NONE if the TSV does not include it.
    Returns a dict with agreement, per-category precision/recall, and a
    confusion matrix indexed as ``cm[kenya_cat][our_cat]``.
    """
    n_total = 0
    n_agree = 0
    cm: dict[str, dict[str, int]] = {k: defaultdict(int) for k in CATEGORIES}
    our_counts: Counter[str] = Counter()
    kenya_counts: Counter[str] = Counter()
    disagreements: list[tuple[str, str, str]] = []

    for sid, our_cat in verdicts.items():
        kenya_cat = kenya_labels.get(sid, "NONE")
        n_total += 1
        our_counts[our_cat] += 1
        kenya_counts[kenya_cat] += 1
        cm[kenya_cat][our_cat] += 1
        if our_cat == kenya_cat:
            n_agree += 1
        else:
            disagreements.append((sid, kenya_cat, our_cat))

    per_cat: dict[str, dict[str, float | int]] = {}
    for cat in CATEGORIES:
        tp = cm[cat][cat]
        fn = sum(cm[cat][c] for c in CATEGORIES if c != cat)  # kenya said cat, we said other
        fp = sum(cm[other][cat] for other in CATEGORIES if other != cat)  # we said cat, kenya said other
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        per_cat[cat] = {
            "kenya_count": kenya_counts[cat],
            "ours_count": our_counts[cat],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }

    return {
        "n_total": n_total,
        "n_agree": n_agree,
        "agreement": n_agree / n_total if n_total else float("nan"),
        "per_category": per_cat,
        "confusion_matrix": {k: dict(v) for k, v in cm.items()},
        "disagreements": disagreements,
    }


def short_cat(cat: str) -> str:
    """Short alias for column/row headers in the confusion matrix."""
    return {
        "MATERNAL": "MAT",
        "NEONATAL": "NEO",
        "CHILD_HEALTH": "CHL",
        "SEXUAL_AND_REPRODUCTIVE_HEALTH": "SRH",
        "NONE": "NONE",
    }[cat]


def print_report(metrics: dict, *, max_disagreements: int) -> None:
    n_total = metrics["n_total"]
    if n_total == 0:
        print("no verdicts loaded — nothing to compare")
        return

    pct = metrics["agreement"] * 100
    print()
    print(f"Compared {n_total} StudyIDs; {metrics['n_agree']} agree ({pct:.1f}%)")
    print()

    print("Per-category metrics")
    print(f"  {'category':<35} {'kenya':>6} {'ours':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7}")
    for cat in CATEGORIES:
        m = metrics["per_category"][cat]
        p = f"{m['precision']:.3f}" if m["precision"] == m["precision"] else "  n/a"
        r = f"{m['recall']:.3f}" if m["recall"] == m["recall"] else "  n/a"
        print(
            f"  {cat:<35} {m['kenya_count']:>6} {m['ours_count']:>6} "
            f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {p:>7} {r:>7}"
        )
    print()

    print("Confusion matrix (rows = Kenya, cols = ours)")
    header_cells = [short_cat(c) for c in CATEGORIES]
    label = "kenya/ours"
    print(f"  {label:<10}" + "".join(f"{cell:>7}" for cell in header_cells))
    for kenya_cat in CATEGORIES:
        row_cells = [str(metrics["confusion_matrix"][kenya_cat].get(c, 0)) for c in CATEGORIES]
        print(f"  {short_cat(kenya_cat):<10}" + "".join(f"{cell:>7}" for cell in row_cells))
    print()

    disagreements = metrics["disagreements"]
    if not disagreements:
        print("no disagreements 🎉")
        return
    print(f"Disagreements ({len(disagreements)} total)" + (
        f" — showing first {max_disagreements}:" if len(disagreements) > max_disagreements else ":"
    ))
    for sid, kenya_cat, our_cat in disagreements[:max_disagreements]:
        print(f"  [{sid}] kenya={kenya_cat:<35} ours={our_cat}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdicts",
        type=Path,
        required=True,
        help="JSONL of mamabench verdicts for Kenya (from classify_obgyn.py).",
    )
    parser.add_argument(
        "--kenya-tsv",
        type=Path,
        default=Path(
            "/Users/renyi/Downloads/obgyn-qa-collection/kenya-clinical-vignettes/data/obgyn_vignettes.tsv"
        ),
        help="Kenya's existing 4-category-labeled TSV (default: the local "
        "obgyn-qa-collection path).",
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=30,
        help="Cap on the disagreement list printed at the bottom (default 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.verdicts.is_file():
        print(f"error: --verdicts not found: {args.verdicts}")
        return 2
    if not args.kenya_tsv.is_file():
        print(f"error: --kenya-tsv not found: {args.kenya_tsv}")
        return 2

    verdicts = load_verdicts(args.verdicts)
    kenya_labels = load_kenya_labels(args.kenya_tsv)
    metrics = compute_metrics(verdicts, kenya_labels)
    print_report(metrics, max_disagreements=args.max_disagreements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
