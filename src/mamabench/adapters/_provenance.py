"""Shared prepared-input provenance helpers for source adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping


def build_prepared_input_metadata(
    input_tsv: str | Path | None,
    *,
    expected_repository: str,
    expected_path: str,
    base_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return prepared-input provenance for a manifest source block.

    When the input file lives in a clean clone of the expected repository at the
    expected path, the result records repository, path, commit, and dirty
    status. Otherwise it records nested expected/actual details with
    verified=False so a manifest reader can audit the mismatch.
    """

    expected = {"repository": expected_repository, "path": expected_path}
    git_metadata = _git_metadata(
        input_tsv,
        expected_repository=expected_repository,
        expected_path=expected_path,
        expected=expected,
    )
    if git_metadata is None:
        git_metadata = {"expected": dict(expected), "verified": False}
    return {**dict(base_metadata), **git_metadata}


def _git_metadata(
    input_tsv: str | Path | None,
    *,
    expected_repository: str,
    expected_path: str,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    if input_tsv is None:
        return None

    input_path = Path(input_tsv).resolve()
    git_directory = input_path.parent if input_path.is_file() else input_path

    try:
        repo_root = Path(
            _git_output(git_directory, "rev-parse", "--show-toplevel")
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        relative_path = input_path.relative_to(repo_root).as_posix()
        try:
            repository = _git_output(repo_root, "remote", "get-url", "origin")
        except (OSError, subprocess.CalledProcessError):
            return _unverified(expected, actual={"path": relative_path})
        if (
            relative_path != expected_path
            or not _repositories_match(repository, expected_repository)
        ):
            return _unverified(
                expected,
                actual={"repository": repository, "path": relative_path},
            )

        commit = _git_output(repo_root, "rev-parse", "HEAD")
        dirty = bool(_git_output(repo_root, "status", "--porcelain"))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None

    return {
        "repository": expected_repository,
        "path": relative_path,
        "commit": commit,
        "git_dirty": dirty,
        "verified": True,
    }


def _unverified(
    expected: Mapping[str, Any], *, actual: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "expected": dict(expected),
        "actual": {key: value for key, value in actual.items() if value},
        "verified": False,
    }


def _repositories_match(actual: str, expected: str) -> bool:
    return _canonical_repository_url(actual) == _canonical_repository_url(expected)


def _canonical_repository_url(value: str) -> str:
    repository = value.strip().rstrip("/")
    if repository.endswith(".git"):
        repository = repository[: -len(".git")]
    if repository.startswith("git@github.com:"):
        repository = f"github.com/{repository.removeprefix('git@github.com:')}"
    for prefix in ("https://", "http://"):
        if repository.startswith(prefix):
            repository = repository.removeprefix(prefix)
            break
    return repository.removeprefix("www.").rstrip("/")


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
