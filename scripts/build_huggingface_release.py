#!/usr/bin/env python3
"""Stage the v0.x release payload for upload to nmrenyi/mamabench on HF.

Writes a directory tree mirroring the layout the HF dataset will have:

    <staging_dir>/
        README.md
        data/
            <per-source JSONLs>
        side_tables/
            healthbench_criteria.jsonl                  (v0.2+ only; rubric side-table)
        calibration/                                    (v0.2.1+ only — judge-calibration side-file)
            obgyn_meta_eval.jsonl                       OBGYN-scoped HealthBench grader meta-eval
        manifests/
            <per-source manifest JSONs>
            release_manifest.json
        schema/
            mamabench_v<schema_version>.md
            mamabench_v<schema_version>.schema.json
        audit/                                          (v0.2+ only — LLM-pipeline provenance)
            prompts/
                obgyn_classifier.md                     classifier prompt docs
                obgyn_classifier/                       modular prompt sections
                keyfact_extractor.md                    single-file extractor prompt
            key_facts/
                <source>_keyfacts_reasoning.jsonl       per-row CoT for the keyfact extractor (joined by row_id)
            classification_verdicts/
                <source>_reasoning.jsonl                per-row CoT for the OBGYN classifier (joined by row_id)
                oss_eval.qwen3_397b_v8.jsonl            397B cross-classifier evidence (HealthBench oss_eval)
                oss_eval_reasoning.qwen3_397b_v8.jsonl  397B CoT for the same
                oss_eval_excluded.jsonl                 7 oss_eval rows the 27B classifier didn't converge on
                hard_excluded.jsonl                     2 hard rows (subset of the 7)

The prompts directory is included so HF dataset consumers can audit the
LLM-pipeline filters and annotations that produced each row. Each entry's
``source.metadata.obgyn_classification`` and ``source.metadata.key_fact_extraction``
references a ``prompt_version`` that resolves to the files staged here.

The script does NOT perform the upload. Inspect the staging directory, then
push it with `hf upload` and tag the release with `hf repos tag create`.

Default release shape is v0.2.1 (`--release v0.2.1`) — v0.2's benchmark rows
plus the judge-calibration side-file under calibration/. Pass `--release v0.2`
for the rows-only shape, or `--release v0.1` for the original three-MCQ-source
release. Each release locks in its file list to avoid surprises if the working
tree contains an in-progress next version.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# Per-release file lists. Keeping them explicit (rather than auto-discovering)
# means a partially-built next-version directory in the tree can't pollute the
# previous release's stage.
RELEASE_FILES: dict[str, dict[str, object]] = {
    "v0.1": {
        "benchmark_dir": Path("benchmark/v0.1"),
        "schema_version": "0.3",
        "schema_dir": Path("schemas"),
        "jsonls": [
            "medmcqa.jsonl",
            "medqa_usmle.jsonl",
            "afrimedqa.jsonl",
        ],
        "manifests": [
            "medmcqa_manifest.json",
            "medqa_usmle_manifest.json",
            "afrimedqa_manifest.json",
            "release_manifest.json",
        ],
        "side_tables": [],
    },
    "v0.2": {
        "benchmark_dir": Path("benchmark/v0.2"),
        "schema_version": "0.4",
        "schema_dir": Path("schemas"),
        "jsonls": [
            # MCQ track (set_type == "mcq")
            "medmcqa.jsonl",
            "medqa_usmle.jsonl",
            "afrimedqa.jsonl",
            # Open-ended track (set_type == "open_ended")
            "afrimedqa_saq.jsonl",
            "kenya.jsonl",
            "whb.jsonl",
            # Rubric track (set_type == "open_ended_rubric") — HealthBench
            "healthbench_oss_eval.jsonl",
            "healthbench_consensus.jsonl",
            "healthbench_hard.jsonl",
        ],
        "manifests": [
            "medmcqa_manifest.json",
            "medqa_usmle_manifest.json",
            "afrimedqa_manifest.json",
            "afrimedqa_saq_manifest.json",
            "kenya_manifest.json",
            "whb_manifest.json",
            "healthbench_manifest.json",
            "release_manifest.json",
        ],
        # Side-files live under data/ but are not benchmark rows. Documented
        # in the dataset card so consumers know to load them directly rather
        # than via `load_dataset()`.
        "side_tables": [
            "healthbench_criteria.jsonl",
        ],
        # Per-row reasoning side-files for the LLM-pipeline annotations.
        # source.metadata.key_fact_extraction and source.metadata.obgyn_classification
        # on each row carry the summary/keyfacts and category/rationale; the
        # matching `*_reasoning.jsonl` files here add the model's full
        # chain-of-thought for audit. Joined by row_id. Paths are relative
        # to benchmark_dir and preserved in the staging dir.
        "audit_files": [
            # Keyfact-extractor reasoning (open-ended sources with reference
            # answers — used to compute keyfact recall in evaluation).
            "key_facts/whb_keyfacts_reasoning.jsonl",
            "key_facts/afrimedqa_saq_keyfacts_reasoning.jsonl",
            "key_facts/kenya_keyfacts_reasoning.jsonl",
            # OBGYN-classifier reasoning (every source — captured for ALL
            # source rows including those filtered out as NONE, so consumers
            # can audit why any row was dropped vs kept).
            "classification_verdicts/oss_eval_reasoning.jsonl",
            "classification_verdicts/hard_reasoning.jsonl",
            "classification_verdicts/kenya_reasoning.jsonl",
            "classification_verdicts/medqa_usmle_reasoning.jsonl",
            # 397B cross-classifier evidence (Qwen3.5-397B-A17B-FP8 run with
            # identical prompt v8 on HealthBench oss_eval). 98.12% agreement
            # with the 27B; see the dataset card §"Cross-classifier
            # consistency check".
            "classification_verdicts/oss_eval.qwen3_397b_v8.jsonl",
            "classification_verdicts/oss_eval_reasoning.qwen3_397b_v8.jsonl",
            # Honest disclosure: 7 oss_eval prompts (2 also in hard, 3 in
            # consensus) on which the 27B classifier did not converge within
            # 64K reasoning tokens at temp=0; documented in the dataset card.
            "classification_verdicts/oss_eval_excluded.jsonl",
            "classification_verdicts/hard_excluded.jsonl",
        ],
        # LLM-pipeline prompts referenced by each row's prompt_version field.
        # Bundle them with the release so consumers can audit what the
        # classifier / extractor saw. Strings name files OR directories under
        # ./prompts/ — directories are copied recursively.
        "prompts": [
            "obgyn_classifier.md",          # docs (loader, change log, self-tests)
            "obgyn_classifier",             # modular markdown sections
            "keyfact_extractor.md",         # single-file extractor prompt
        ],
    },
}

# v0.2.1 — additive patch over v0.2. The benchmark rows, schema, configs and
# manifests are byte-identical to v0.2; the only addition is the judge-
# calibration side-file under calibration/. `benchmark_version` therefore
# stays "v0.2"; "v0.2.1" is the release tag, not a new benchmark version.
RELEASE_FILES["v0.2.1"] = {
    **RELEASE_FILES["v0.2"],
    # Calibration side-files: paths are relative to benchmark_dir/calibration/
    # and land under calibration/ in the staged release.
    "calibration_files": [
        "obgyn_meta_eval.jsonl",
    ],
}

DATASET_CARD = Path("docs/huggingface_dataset_card.md")


AUDIT_README = """# audit/ — LLM-pipeline provenance for mamabench v0.2

Everything here is **supporting material**, not benchmark rows. The benchmark
itself is in `data/` (joined by `id`); the rubric criteria side-table is in
`side_tables/`. Each file here joins back to benchmark rows by `row_id` so
you can audit any single decision.

## What's here

- `prompts/` — the exact prompts used by the LLM pipeline, versioned (`v8`).
  Each benchmark row's `source.metadata.{obgyn_classification,key_fact_extraction}.prompt_version`
  pins to these files.

- `key_facts/<source>_keyfacts_reasoning.jsonl` — the keyfact extractor's
  full chain-of-thought for each open-ended row with a reference answer
  (Kenya, AfriMed-SAQ, WHB). The keyfacts themselves are already inlined
  on each benchmark row under `source.metadata.key_fact_extraction`; the
  reasoning is the model's `<think>` block for that extraction.

- `classification_verdicts/<source>_reasoning.jsonl` — the OBGYN classifier's
  full chain-of-thought for **every** source row (4 sources × thousands of
  rows). Included rows have their verdict inlined on the benchmark row
  under `source.metadata.obgyn_classification`; reasoning here adds the
  `<think>` block. Excluded (`NONE`-classified) rows also appear here so
  you can audit why something was dropped.

- `classification_verdicts/oss_eval.qwen3_397b_v8.jsonl` + reasoning side-
  file — **397B cross-classifier evidence**. The default v0.2 classifier
  is Qwen3.6-27B-FP8; we also ran Qwen3.5-397B-A17B-FP8 on HealthBench
  `oss_eval` with the identical prompt v8 and saw 98.12% agreement. The
  larger-model verdicts and reasoning are preserved here for audit.

- `classification_verdicts/{oss_eval,hard}_excluded.jsonl` — 7 HealthBench
  `oss_eval` prompts (+ 2 also in `hard`) on which the 27B classifier did
  not converge within 64K reasoning tokens at temperature=0. Each entry
  records the full prompt text and exclusion reason. The 397B classified
  all 7 as `NONE` (non-OBGYN), so excluding them does not change the
  filtered row set; this file documents what would otherwise be silent.

## Join pattern

```python
import json
verdicts = {}
for line in open("audit/classification_verdicts/oss_eval_reasoning.jsonl"):
    r = json.loads(line)
    verdicts[r["row_id"]] = r  # {row_id, source, model, prompt, input, params, output: {content, reasoning}}

# For any benchmark row in data/healthbench_oss_eval.jsonl, look up by
# row.source.id (which equals row_id here).
```

See the dataset card (`/README.md`) for the full description.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        choices=sorted(RELEASE_FILES),
        default="v0.2.1",
        help="Which release version's file list to stage (default v0.2.1).",
    )
    parser.add_argument(
        "--staging-dir",
        required=True,
        help="Destination directory for the staged release payload.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the staging directory if it already exists.",
    )
    args = parser.parse_args(argv)

    spec = RELEASE_FILES[args.release]
    benchmark_dir: Path = spec["benchmark_dir"]  # type: ignore[assignment]
    schema_version: str = spec["schema_version"]  # type: ignore[assignment]
    schema_dir: Path = spec["schema_dir"]  # type: ignore[assignment]
    schema_json = ROOT / schema_dir / f"mamabench_v{schema_version}.schema.json"
    schema_md = ROOT / schema_dir / f"mamabench_v{schema_version}.md"

    staging_dir = Path(args.staging_dir).resolve()
    if staging_dir.exists():
        if not args.force:
            print(
                f"error: {staging_dir} already exists; pass --force to overwrite",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # File plan: (src, dst). All entries are individual files copied with copy2.
    plan: list[tuple[Path, Path]] = []
    plan.append((ROOT / DATASET_CARD, staging_dir / "README.md"))
    for name in spec["jsonls"]:  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / name, staging_dir / "data" / name))
    # Side-tables live under side_tables/ rather than data/ so HF's default
    # `data/*.jsonl` config doesn't try to merge their (different) schemas
    # with the benchmark rows.
    for name in spec["side_tables"]:  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / name, staging_dir / "side_tables" / name))
    # Audit-only side-files (per-row LLM-pipeline reasoning traces, prompts,
    # cross-classifier evidence, dropped-row disclosures). All consolidated
    # under `audit/` in the release so the top-level layout cleanly separates
    # benchmark rows (`data/`, `side_tables/`) from provenance.
    for name in spec.get("audit_files", []):  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / name, staging_dir / "audit" / name))
    # Calibration side-files (judge-calibration data — not benchmark rows, not
    # in the loadable configs, not counted in the release manifest). Shipped
    # under calibration/ alongside side_tables/ and audit/.
    for name in spec.get("calibration_files", []):  # type: ignore[union-attr]
        plan.append(
            (
                ROOT / benchmark_dir / "calibration" / name,
                staging_dir / "calibration" / name,
            )
        )
    for name in spec["manifests"]:  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / "manifests" / name, staging_dir / "manifests" / name))
    plan.append((schema_json, staging_dir / "schema" / schema_json.name))
    plan.append((schema_md, staging_dir / "schema" / schema_md.name))

    # Tree plan: (src_dir_or_file, dst_under_staging). Items in spec["prompts"]
    # name either files OR directories under ./prompts/; directories are
    # copied recursively so the modular obgyn_classifier/ folder lands intact.
    # Prompts go under `audit/prompts/` for the v0.2+ consolidated layout.
    tree_plan: list[tuple[Path, Path]] = []
    for name in spec.get("prompts", []):  # type: ignore[union-attr]
        src = ROOT / "prompts" / name
        dst = staging_dir / "audit" / "prompts" / name
        tree_plan.append((src, dst))

    missing = [src for src, _ in plan if not src.is_file()]
    missing.extend(src for src, _ in tree_plan if not src.exists())
    if missing:
        print(f"error: missing required files for {args.release}:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nRegenerate the missing artifacts before staging the release.",
            file=sys.stderr,
        )
        return 2

    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for src, dst in tree_plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Drop a short reader-facing README inside audit/ so consumers landing
    # there from the HF file browser understand the join pattern without
    # bouncing back to the dataset card.
    audit_dir = staging_dir / "audit"
    if audit_dir.is_dir():
        audit_readme = audit_dir / "README.md"
        audit_readme.write_text(AUDIT_README)

    _print_staged_layout(staging_dir)
    _print_publish_commands(staging_dir, benchmark_version=args.release)
    return 0


def _print_staged_layout(staging_dir: Path) -> None:
    print(f"staged release payload at: {staging_dir}\n")
    print("contents:")
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(staging_dir).as_posix()
            size_kb = path.stat().st_size / 1024
            print(f"  {rel:<50} {size_kb:>10.1f} KB")


def _print_publish_commands(staging_dir: Path, *, benchmark_version: str) -> None:
    tag = benchmark_version  # already normalized with leading 'v'
    print()
    print("to publish (review the README.md first):")
    print()
    print("  hf repos create nmrenyi/mamabench --type dataset --public --exist-ok")
    print(
        "  hf upload nmrenyi/mamabench"
        f" {staging_dir} ."
        ' --type dataset --commit-message "release ' + tag + '"'
    )
    print(
        f"  hf repos tag create nmrenyi/mamabench {tag} --type dataset"
    )
    print()
    print("verify after upload:")
    print(f"  hf download nmrenyi/mamabench --repo-type dataset --include README.md")


if __name__ == "__main__":
    raise SystemExit(main())
