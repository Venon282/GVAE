"""Consistency checks for the changelog documentation (`docs/changelog/`, `mkdocs.yml`).

The changelog is one file per released version, `CHANGELOG_<version>_<date>.md`, plus
`CHANGELOG.md` (the running log, with the `[Unreleased]` section) and
`docs/changelog/SUMMARY.md`, the hand-maintained list that `literate-nav` turns into the
site sidebar. The list drifted from the files on disk: releases 1.1.1 and 1.2.0 existed
without being listed, so their pages were built but orphaned, with no sidebar entry and no
link to them. One file name also used underscores in its date (`2026_09_27`) where the
others use hyphens. Nothing failed, because nothing checked. This file is that check, the
changelog counterpart of `test_docs_adr_navigation.py` (which already guards the directory
form of the `Changelog` nav entry): it fails as soon as a release file is added without
being listed, a listed link points nowhere, a file name breaks the naming convention, or the
list, the file name and the heading inside the file disagree about a version or its date.

Most checks are plain string/regex checks over the Markdown files, so they run in the same
environment as the rest of the test suite. The last test builds the site and inspects the
real sidebar; it is skipped when the optional `docs` toolchain is not installed.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG_DIR = _REPO_ROOT / "docs" / "changelog"
_RELEASE_FILE_PATTERN = re.compile(r"^CHANGELOG_(\d+)\.(\d+)\.(\d+)_(\d{4}-\d{2}-\d{2})\.md$")
_MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
_SUMMARY_ENTRY_PATTERN = re.compile(r"^- \[(?P<label>[^\]]+)\]\((?P<target>[^)\s]+\.md)\)\s*$")
_RELEASE_HEADING_PATTERN = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\s*$")


def _releaseFiles() -> list[Path]:
    """Every per-version changelog file, in no particular order."""
    return sorted(
        path for path in _CHANGELOG_DIR.iterdir() if _RELEASE_FILE_PATTERN.match(path.name)
    )


def _versionAndDate(path: Path) -> tuple[str, str]:
    match = _RELEASE_FILE_PATTERN.match(path.name)
    assert match is not None
    major, minor, patch, date = match.groups()
    return f"{major}.{minor}.{patch}", date


def _versionKey(path: Path) -> tuple[int, int, int]:
    match = _RELEASE_FILE_PATTERN.match(path.name)
    assert match is not None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _linkedTargets(markdown_path: Path) -> list[str]:
    return _MARKDOWN_LINK_PATTERN.findall(markdown_path.read_text(encoding="utf-8"))


def test_every_changelog_file_follows_the_naming_convention() -> None:
    """`CHANGELOG_<major>.<minor>.<patch>_<YYYY-MM-DD>.md`, with hyphens in the date.

    `CHANGELOG_1.1.1_2026_09_27.md` was once written with underscores. Besides being
    inconsistent, a name that does not match the convention is invisible to every check below.
    """
    offenders = [
        path.name
        for path in _CHANGELOG_DIR.glob("CHANGELOG_*.md")
        if not _RELEASE_FILE_PATTERN.match(path.name)
    ]
    assert not offenders, (
        f"changelog file name(s) not matching CHANGELOG_<x.y.z>_<YYYY-MM-DD>.md: {offenders}. "
        f"Use hyphens in the date."
    )


def test_the_changelog_directory_has_release_files() -> None:
    assert _releaseFiles(), "docs/changelog/ holds no CHANGELOG_<version>_<date>.md file."


def test_every_release_is_listed_in_the_sidebar_summary() -> None:
    linked = set(_linkedTargets(_CHANGELOG_DIR / "SUMMARY.md"))
    missing = [path.name for path in _releaseFiles() if path.name not in linked]
    assert not missing, (
        f"docs/changelog/SUMMARY.md (the sidebar) does not list: {missing}. Add one "
        f"'- [x.y.z (YYYY-MM-DD)](file.md)' line per release."
    )


def test_summary_starts_with_the_running_changelog() -> None:
    first_target = _linkedTargets(_CHANGELOG_DIR / "SUMMARY.md")[0]
    assert first_target == "CHANGELOG.md"


def test_summary_lists_the_releases_newest_first() -> None:
    listed = [
        target
        for target in _linkedTargets(_CHANGELOG_DIR / "SUMMARY.md")
        if target != "CHANGELOG.md"
    ]
    expected = [path.name for path in sorted(_releaseFiles(), key=_versionKey, reverse=True)]
    assert listed == expected, (
        f"docs/changelog/SUMMARY.md must list releases newest first: expected {expected}, "
        f"got {listed}."
    )


def test_summary_lists_no_release_twice() -> None:
    targets = _linkedTargets(_CHANGELOG_DIR / "SUMMARY.md")
    assert len(targets) == len(set(targets))


def test_every_link_in_the_summary_resolves_to_a_file() -> None:
    for target in _linkedTargets(_CHANGELOG_DIR / "SUMMARY.md"):
        if target.startswith(("http://", "https://", "..")):
            continue
        assert (_CHANGELOG_DIR / target).is_file(), f"SUMMARY.md links to missing '{target}'."


def test_summary_labels_state_the_version_and_date_of_the_file_they_link_to() -> None:
    """A label `1.1.0 (2026-09-16)` must describe the file it points at, not a neighbour."""
    lines = (_CHANGELOG_DIR / "SUMMARY.md").read_text(encoding="utf-8").splitlines()
    for line in lines:
        entry = _SUMMARY_ENTRY_PATTERN.match(line)
        if entry is None or not _RELEASE_FILE_PATTERN.match(entry.group("target")):
            continue
        version, date = _versionAndDate(_CHANGELOG_DIR / entry.group("target"))
        assert entry.group("label") == f"{version} ({date})", (
            f"SUMMARY.md labels {entry.group('target')} as '{entry.group('label')}', "
            f"expected '{version} ({date})'."
        )


@pytest.mark.parametrize("path", _releaseFiles(), ids=lambda path: path.name)
def test_release_heading_matches_the_file_name(path: Path) -> None:
    """The `## [x.y.z] - YYYY-MM-DD` heading inside a release file agrees with its name."""
    headings = [
        match.groups()
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := _RELEASE_HEADING_PATTERN.match(line))
    ]
    assert headings == [_versionAndDate(path)], (
        f"{path.name} must contain exactly one heading '## [x.y.z] - YYYY-MM-DD' equal to "
        f"its file name, found {headings}."
    )


_DOCS_TOOLCHAIN = ("mkdocs", "mkdocs_literate_nav", "mkdocs_gen_files", "mkdocstrings")


@pytest.mark.slow
@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in _DOCS_TOOLCHAIN),
    reason="the optional 'docs' extra (mkdocs, literate-nav, ...) is not installed",
)
def test_built_site_sidebar_lists_every_release(tmp_path: Path) -> None:
    """Build the real site and check what a reader actually sees in the sidebar."""
    site_dir = tmp_path / "site"
    completed = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--quiet", "-d", str(site_dir)],
        cwd=_REPO_ROOT,
        env={**os.environ, "DISABLE_MKDOCS_2_WARNING": "true"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr

    # `CHANGELOG.md` is the first sidebar entry but not an `index.md`, so it is rendered at
    # `changelog/CHANGELOG/`. Every sidebar link is relative to that page.
    page = (site_dir / "changelog" / "CHANGELOG" / "index.html").read_text(encoding="utf-8")
    sidebar_hrefs = set(re.findall(r'href="([^"#]+)"', page))
    missing = [path.stem for path in _releaseFiles() if f"../{path.stem}/" not in sidebar_hrefs]
    assert not missing, f"changelog page(s) missing from the built sidebar: {missing}"
    assert not any("SUMMARY" in href for href in sidebar_hrefs), (
        "a SUMMARY page is being rendered as an ordinary page instead of being expanded"
    )
