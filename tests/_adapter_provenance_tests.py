"""Shared assertion helpers for adapter prepared-input provenance tests.

Each adapter test module exercises the same four-or-five scenarios against its
own ``build_<dataset>_source_metadata`` function and its own
``PREPARED_INPUT_PATH``. The helpers below parameterize those scenarios so
each adapter's test class can call them with adapter-specific arguments and
keep failure attribution per-adapter.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Mapping


EXPECTED_REPOSITORY = "https://github.com/nmrenyi/obgyn-qa-collection"
EXPECTED_REPOSITORY_GIT = f"{EXPECTED_REPOSITORY}.git"
UNEXPECTED_REPOSITORY_GIT = "https://github.com/example/obgyn-qa-collection.git"
NONCANONICAL_REPO_PATH = "other/source.tsv"


def assert_verified_prepared_input(
    test_case: unittest.TestCase,
    *,
    dataset_name: str,
    expected_path: str,
    tsv_header: str,
    build_metadata: Callable[[Path], Mapping[str, Any]],
) -> None:
    """Happy path: input lives in a clean clone of the expected repo and path."""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "obgyn-qa-collection"
        tsv = _write_fixture_tsv(repo, expected_path, tsv_header)
        commit = _init_repo_with_remote(repo, EXPECTED_REPOSITORY_GIT)

        prepared = build_metadata(tsv)[dataset_name]["prepared_input"]

        test_case.assertEqual(prepared["repository"], EXPECTED_REPOSITORY)
        test_case.assertEqual(prepared["path"], expected_path)
        test_case.assertEqual(prepared["commit"], commit)
        test_case.assertFalse(prepared["git_dirty"])
        test_case.assertTrue(prepared["verified"])
        test_case.assertNotIn("expected", prepared)
        test_case.assertNotIn("actual", prepared)


def assert_unexpected_remote(
    test_case: unittest.TestCase,
    *,
    dataset_name: str,
    expected_path: str,
    tsv_header: str,
    build_metadata: Callable[[Path], Mapping[str, Any]],
) -> None:
    """Origin URL points to a different repository."""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "obgyn-qa-collection"
        tsv = _write_fixture_tsv(repo, expected_path, tsv_header)
        _init_repo_with_remote(repo, UNEXPECTED_REPOSITORY_GIT)

        prepared = build_metadata(tsv)[dataset_name]["prepared_input"]

        test_case.assertEqual(
            prepared["expected"]["repository"], EXPECTED_REPOSITORY
        )
        test_case.assertEqual(prepared["expected"]["path"], expected_path)
        test_case.assertEqual(
            prepared["actual"]["repository"], UNEXPECTED_REPOSITORY_GIT
        )
        test_case.assertEqual(prepared["actual"]["path"], expected_path)
        test_case.assertFalse(prepared["verified"])
        test_case.assertNotIn("repository", prepared)
        test_case.assertNotIn("path", prepared)
        test_case.assertNotIn("commit", prepared)
        test_case.assertNotIn("git_dirty", prepared)


def assert_missing_remote(
    test_case: unittest.TestCase,
    *,
    dataset_name: str,
    expected_path: str,
    tsv_header: str,
    build_metadata: Callable[[Path], Mapping[str, Any]],
) -> None:
    """Repo exists but no origin remote is configured."""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "obgyn-qa-collection"
        tsv = _write_fixture_tsv(repo, expected_path, tsv_header)
        _init_repo_without_remote(repo)

        prepared = build_metadata(tsv)[dataset_name]["prepared_input"]

        test_case.assertEqual(prepared["expected"]["path"], expected_path)
        test_case.assertEqual(prepared["actual"]["path"], expected_path)
        test_case.assertFalse(prepared["verified"])
        test_case.assertNotIn("repository", prepared["actual"])
        test_case.assertNotIn("repository", prepared)
        test_case.assertNotIn("path", prepared)
        test_case.assertNotIn("commit", prepared)
        test_case.assertNotIn("git_dirty", prepared)


def assert_noncanonical_path(
    test_case: unittest.TestCase,
    *,
    dataset_name: str,
    expected_path: str,
    tsv_header: str,
    build_metadata: Callable[[Path], Mapping[str, Any]],
) -> None:
    """Input lives in a git repo but not at the expected adapter path."""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "obgyn-qa-collection"
        tsv = _write_fixture_tsv(repo, NONCANONICAL_REPO_PATH, tsv_header)
        _init_repo_without_remote(repo)

        prepared = build_metadata(tsv)[dataset_name]["prepared_input"]

        test_case.assertEqual(prepared["expected"]["path"], expected_path)
        test_case.assertEqual(
            prepared["actual"]["path"], NONCANONICAL_REPO_PATH
        )
        test_case.assertFalse(prepared["verified"])
        test_case.assertNotIn("path", prepared)
        test_case.assertNotIn("commit", prepared)
        test_case.assertNotIn("git_dirty", prepared)


def assert_random_local_input_path(
    test_case: unittest.TestCase,
    *,
    dataset_name: str,
    expected_path: str,
    tsv_header: str,
    build_metadata: Callable[[Path], Mapping[str, Any]],
) -> None:
    """Input file is not inside any git repository."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv = Path(tmpdir) / "source.tsv"
        tsv.write_text(tsv_header)

        prepared = build_metadata(tsv)[dataset_name]["prepared_input"]

        test_case.assertEqual(prepared["expected"]["path"], expected_path)
        test_case.assertFalse(prepared["verified"])
        test_case.assertNotIn("actual", prepared)
        test_case.assertNotIn("path", prepared)
        test_case.assertNotIn("commit", prepared)
        test_case.assertNotIn("git_dirty", prepared)


def _write_fixture_tsv(repo: Path, path_in_repo: str, header: str) -> Path:
    tsv = repo / path_in_repo
    tsv.parent.mkdir(parents=True)
    tsv.write_text(header)
    return tsv


def _init_repo_with_remote(repo: Path, origin_url: str) -> str:
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        ["git", "remote", "add", "origin", origin_url], cwd=repo, check=True
    )
    return _commit_and_get_head(repo)


def _init_repo_without_remote(repo: Path) -> str:
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    return _commit_and_get_head(repo)


def _commit_and_get_head(repo: Path) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=mamabench",
            "-c",
            "user.email=mamabench@example.test",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
