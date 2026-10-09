r"""Typography guard (spec section 10, roadmap P0-3).

The house style bans em dashes (U+2014) and unicode arrows (for example U+2192) in code,
comments, docstrings and project documentation. Use a period, a colon, parentheses or two
sentences instead of an em dash, and `->` instead of an arrow. No linter checks this, so
this module does: it scans every tracked or new (not ignored) `.py`, `.md`, `.yaml`, `.yml`
and `.toml` file and lists each offending character with its file, line and column.

A character counts as an arrow when its Unicode name contains "ARROW", which covers every
arrow block (U+2190 to U+21FF, the supplemental arrows, the dingbat arrows) without a
hand-kept range table. En dashes, hyphens and the minus sign are allowed.

This file never contains a banned character itself: the detector tests below build their
inputs from `\u` escapes, so the scan can run over the whole repository, this file included.
Commit messages are not scanned (history cannot be rewritten), so review those by hand.
"""

import os
import re
import subprocess
import unicodedata
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_SUFFIXES = frozenset({".py", ".md", ".yaml", ".yml", ".toml"})
# Directories that hold tooling output or third-party code, never project text. Only used by
# the directory walk, which stands in for `git ls-files` outside a git checkout.
_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "build",
        "dist",
        "site",
    }
)
_EM_DASH = "\u2014"
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


class Finding(NamedTuple):
    """One banned character found in a text."""

    line: int
    column: int
    description: str
    excerpt: str


@cache
def _bannedName(character: str) -> str | None:
    """Return the Unicode name of `character` if the house style bans it, else `None`."""
    if character == _EM_DASH:
        return "EM DASH"
    name = unicodedata.name(character, "")
    return name if "ARROW" in name else None


def _findInText(text: str) -> list[Finding]:
    """Return every banned character in `text`, in reading order."""
    findings: list[Finding] = []
    for match in _NON_ASCII.finditer(text):
        name = _bannedName(match.group())
        if name is None:
            continue
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        findings.append(
            Finding(
                line=text.count("\n", 0, start) + 1,
                column=start - line_start + 1,
                description=f"U+{ord(match.group()):04X} {name}",
                excerpt=text[line_start : len(text) if line_end == -1 else line_end].strip(),
            )
        )
    return findings


def _isScanned(path: Path) -> bool:
    return path.suffix in _SCANNED_SUFFIXES


def _gitTrackedFiles(root: Path) -> list[Path]:
    """Return the tracked and new (not ignored) files under `root`, or `[]` if git cannot say."""
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others"]
    command.append("--exclude-standard")
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    names = completed.stdout.decode("utf-8").split("\0")
    return sorted(root / name for name in names if name)


def _walkedFiles(root: Path) -> list[Path]:
    """Return every file under `root`, skipping tooling and third-party directories."""
    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name not in _SKIPPED_DIRECTORIES and not name.endswith(".egg-info")
        )
        found.extend(Path(directory) / name for name in sorted(filenames))
    return found


def _scannedFiles(root: Path) -> list[Path]:
    """Return the tracked text files the rule applies to.

    Uses `git ls-files` so ignored files (virtual environments, build output, local notes) are
    never scanned, while a new file is scanned before its first commit. Falls back to a
    directory walk when `root` is not a git checkout, for example an unpacked source archive.
    """
    candidates = _gitTrackedFiles(root) or _walkedFiles(root)
    return [path for path in candidates if _isScanned(path) and path.is_file()]


def _scan(paths: Iterable[Path], root: Path) -> list[str]:
    """Return one `path:line:column: character: excerpt` report line per banned character."""
    report: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        report.extend(
            f"{relative}:{finding.line}:{finding.column}: {finding.description}: {finding.excerpt}"
            for finding in _findInText(text)
        )
    return report


def test_no_scanned_file_contains_a_banned_character() -> None:
    report = _scan(_scannedFiles(_REPO_ROOT), _REPO_ROOT)
    assert not report, (
        "Spec section 10 bans em dashes and unicode arrows in code, comments, docstrings and "
        "project documentation. Use a period, a colon, parentheses or two sentences instead of "
        "an em dash, and '->' instead of an arrow. Found:\n" + "\n".join(report)
    )


def test_the_scan_covers_the_files_the_rule_is_about() -> None:
    """An empty or partial file list would make the guard pass for the wrong reason."""
    scanned = {path.relative_to(_REPO_ROOT).as_posix() for path in _scannedFiles(_REPO_ROOT)}
    expected = {
        "README.md",
        "pyproject.toml",
        "mkdocs.yml",
        "docs/global-vae-project-specification.md",
        "docs/roadmap.md",
        "docs/adr/index.md",
        "src/global_vae/__init__.py",
        "tests/unit/test_typography.py",
    }
    assert expected <= scanned, f"not scanned: {sorted(expected - scanned)}"
    assert any(name.endswith(".yaml") for name in scanned)


@pytest.mark.parametrize(
    ("text", "description"),
    [
        ("a \u2014 b", "U+2014 EM DASH"),
        ("a \u2192 b", "U+2192 RIGHTWARDS ARROW"),
        ("a \u21d2 b", "U+21D2 RIGHTWARDS DOUBLE ARROW"),
        ("a \u2194 b", "U+2194 LEFT RIGHT ARROW"),
        ("a \u27f6 b", "U+27F6 LONG RIGHTWARDS ARROW"),
        ("a \u2b05 b", "U+2B05 LEFTWARDS BLACK ARROW"),
        ("a \u279c b", "U+279C HEAVY ROUND-TIPPED RIGHTWARDS ARROW"),
    ],
)
def test_detector_flags_em_dashes_and_every_kind_of_arrow(text: str, description: str) -> None:
    findings = _findInText(text)
    assert [finding.description for finding in findings] == [description]


@pytest.mark.parametrize(
    "text",
    [
        "a -> b",
        "a - b",
        "1\u20132",  # en dash: the spec bans the em dash only
        "a \u2212 b",  # minus sign
        "a \u2264 b",  # less-than or equal
        "section \u00a710, caf\u00e9, \u00d7, \u2022",
        "",
    ],
)
def test_detector_leaves_allowed_characters_alone(text: str) -> None:
    assert _findInText(text) == []


def test_detector_reports_line_column_and_excerpt() -> None:
    text = "clean line\n  see \u2014 here  \nlast \u2192 line"
    first, second = _findInText(text)
    assert (first.line, first.column, first.excerpt) == (2, 7, "see \u2014 here")
    assert (second.line, second.column, second.excerpt) == (3, 6, "last \u2192 line")


def test_detector_finds_every_occurrence_on_one_line() -> None:
    assert len(_findInText("\u2014 and \u2014 and \u2192")) == 3


def test_walk_skips_tooling_directories_and_unscanned_suffixes(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / "pkg.egg-info").mkdir()
    (tmp_path / "docs" / "kept.md").write_text("bad \u2014 dash\n", encoding="utf-8")
    (tmp_path / "docs" / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "docs" / "image.png").write_bytes(b"\x89PNG\r\n\xe2\x80\x94")
    (tmp_path / ".venv" / "lib" / "vendored.md").write_text("bad \u2014\n", encoding="utf-8")
    (tmp_path / "pkg.egg-info" / "PKG-INFO.md").write_text("bad \u2192\n", encoding="utf-8")

    scanned = _scannedFiles(tmp_path)  # tmp_path is not a git checkout: the walk is used

    assert [path.relative_to(tmp_path).as_posix() for path in scanned] == [
        "docs/clean.py",
        "docs/kept.md",
    ]
    report = _scan(scanned, tmp_path)
    assert report == ["docs/kept.md:1:5: U+2014 EM DASH: bad \u2014 dash"]
