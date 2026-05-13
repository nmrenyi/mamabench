#!/usr/bin/env python3
"""Write pretty-printed first-row inspection files for each JSONL in a directory.

For every ``*.jsonl`` in ``--input-dir``, reads the first non-empty line and
writes it as pretty JSON to ``<output-dir>/<basename>_first.json``. Skips
files in ``--exclude``. Used to produce ``benchmark/v0.x/inspection/`` so
reviewers can eyeball the row shape per source without having to grep the
full artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for `<basename>_first.json` outputs.",
    )
    parser.add_argument(
        "--also-from",
        action="append",
        default=[],
        help="Additional subdirectory of --input-dir to scan (e.g. side_tables). "
        "Repeat for multiple. classification_verdicts and inspection are never "
        "scanned to avoid pulling in non-mamabench rows.",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"error: --input-dir not a directory: {args.input_dir}", file=sys.stderr)
        return 2

    jsonl_paths = sorted(args.input_dir.glob("*.jsonl"))
    for sub_name in args.also_from:
        sub_dir = args.input_dir / sub_name
        if not sub_dir.is_dir():
            print(f"warning: --also-from skipped ({sub_dir} not found)", file=sys.stderr)
            continue
        jsonl_paths.extend(sorted(sub_dir.glob("*.jsonl")))

    if not jsonl_paths:
        print(f"error: no .jsonl files under {args.input_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    for src_path in jsonl_paths:
        first_row = None
        with src_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                first_row = json.loads(line)
                break
        if first_row is None:
            print(f"  skipped (empty): {src_path}", file=sys.stderr)
            continue
        out_path = args.output_dir / f"{src_path.stem}_first.json"
        out_path.write_text(
            json.dumps(first_row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append((src_path.name, out_path.name))

    print(f"wrote {len(written)} inspection files to {args.output_dir}:")
    for src_name, out_name in written:
        print(f"  {src_name} → {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
