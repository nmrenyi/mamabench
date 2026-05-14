#!/usr/bin/env python3
"""Stage the v0.x release payload for upload to nmrenyi/mamabench on HF.

Writes a directory tree mirroring the layout the HF dataset will have:

    <staging_dir>/
        README.md
        data/
            <per-source JSONLs>
        side_tables/
            healthbench_criteria.jsonl       (v0.2+ only; rubric side-table)
        key_facts/                            (v0.2+ only)
            <source>_keyfacts_reasoning.jsonl  per-row chain-of-thought (joined by row_id)
        manifests/
            <per-source manifest JSONs>
            release_manifest.json
        schema/
            mamabench_v<schema_version>.md
            mamabench_v<schema_version>.schema.json
        prompts/                              (v0.2+ only)
            obgyn_classifier.md               LLM-pipeline prompt docs
            obgyn_classifier/                 modular prompt sections
            keyfact_extractor.md              single-file prompt

The prompts directory is included so HF dataset consumers can audit the
LLM-pipeline filters and annotations that produced each row. Each entry's
``source.metadata.obgyn_classification`` and ``source.metadata.key_fact_extraction``
references a ``prompt_version`` that resolves to the files staged here.

The script does NOT perform the upload. Inspect the staging directory, then
push it with `hf upload` and tag the release with `hf repos tag create`.

Default release shape is v0.2 (`--release v0.2`); pass `--release v0.1` for
the original three-MCQ-source release. Each release locks in its file list
to avoid surprises if the working tree contains an in-progress next version.
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
        # source.metadata.key_fact_extraction on each row carries the
        # summary + key_facts; the matching `*_reasoning.jsonl` here adds
        # the full chain-of-thought (~1 KB per row) for audit. Joined by
        # row_id. Paths are relative to benchmark_dir.
        "audit_files": [
            "key_facts/whb_keyfacts_reasoning.jsonl",
            "key_facts/afrimedqa_saq_keyfacts_reasoning.jsonl",
            "key_facts/kenya_keyfacts_reasoning.jsonl",
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

DATASET_CARD = Path("docs/huggingface_dataset_card.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        choices=sorted(RELEASE_FILES),
        default="v0.2",
        help="Which release version's file list to stage (default v0.2).",
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
    # Audit-only side-files (per-row LLM-pipeline reasoning traces). Paths
    # under benchmark_dir are preserved in the staging dir so the layout is
    # `staging/key_facts/<file>` mirroring the working tree.
    for name in spec.get("audit_files", []):  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / name, staging_dir / name))
    for name in spec["manifests"]:  # type: ignore[union-attr]
        plan.append((ROOT / benchmark_dir / "manifests" / name, staging_dir / "manifests" / name))
    plan.append((schema_json, staging_dir / "schema" / schema_json.name))
    plan.append((schema_md, staging_dir / "schema" / schema_md.name))

    # Tree plan: (src_dir_or_file, dst_under_staging). Items in spec["prompts"]
    # name either files OR directories under ./prompts/; directories are
    # copied recursively so the modular obgyn_classifier/ folder lands intact.
    tree_plan: list[tuple[Path, Path]] = []
    for name in spec.get("prompts", []):  # type: ignore[union-attr]
        src = ROOT / "prompts" / name
        dst = staging_dir / "prompts" / name
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
