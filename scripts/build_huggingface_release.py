#!/usr/bin/env python3
"""Stage the v0.x release payload for upload to nmrenyi/mamabench on HF.

Writes a directory tree mirroring the layout the HF dataset will have:

    <staging_dir>/
        README.md
        data/
            medmcqa.jsonl
            medqa_usmle.jsonl
            afrimedqa.jsonl
        manifests/
            medmcqa_manifest.json
            medqa_usmle_manifest.json
            afrimedqa_manifest.json
            release_manifest.json
        schema/
            mamabench_v0.3.md
            mamabench_v0.3.schema.json

The script does NOT perform the upload. Inspect the staging directory, then
push it with `hf upload` and tag the release with `hf repos tag create`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.config import load_project_config  # noqa: E402


SOURCE_JSONLS: list[tuple[str, Path]] = [
    ("medmcqa.jsonl", Path("benchmark/v0.1/medmcqa.jsonl")),
    ("medqa_usmle.jsonl", Path("benchmark/v0.1/medqa_usmle.jsonl")),
    ("afrimedqa.jsonl", Path("benchmark/v0.1/afrimedqa.jsonl")),
]

SOURCE_MANIFESTS: list[tuple[str, Path]] = [
    ("medmcqa_manifest.json", Path("benchmark/v0.1/manifests/medmcqa_manifest.json")),
    ("medqa_usmle_manifest.json", Path("benchmark/v0.1/manifests/medqa_usmle_manifest.json")),
    ("afrimedqa_manifest.json", Path("benchmark/v0.1/manifests/afrimedqa_manifest.json")),
    ("release_manifest.json", Path("benchmark/v0.1/manifests/release_manifest.json")),
]

DATASET_CARD = Path("docs/huggingface_dataset_card.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    config = load_project_config(ROOT / "mamabench.json")
    # schema files are `mamabench_v0.3.schema.json` and `mamabench_v0.3.md`.
    schema_md = config.schema_file.parent / (
        config.schema_file.name.replace(".schema.json", ".md")
    )

    plan: list[tuple[Path, Path]] = []
    plan.append((ROOT / DATASET_CARD, staging_dir / "README.md"))
    for name, relpath in SOURCE_JSONLS:
        plan.append((ROOT / relpath, staging_dir / "data" / name))
    for name, relpath in SOURCE_MANIFESTS:
        plan.append((ROOT / relpath, staging_dir / "manifests" / name))
    plan.append((config.schema_file, staging_dir / "schema" / config.schema_file.name))
    plan.append((schema_md, staging_dir / "schema" / schema_md.name))

    missing = [src for src, _ in plan if not src.is_file()]
    if missing:
        print("error: missing required files:", file=sys.stderr)
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

    _print_staged_layout(staging_dir)
    _print_publish_commands(staging_dir, benchmark_version=config.benchmark_version)
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
