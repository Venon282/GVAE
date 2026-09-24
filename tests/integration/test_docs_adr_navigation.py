"""Consistency checks for the ADR documentation (`docs/adr/`, `mkdocs.yml`).

Two hand-maintained lists describe the ADRs: the table in `docs/adr/index.md` (number,
title, one-line summary) and `docs/adr/SUMMARY.md` (what `literate-nav` turns into the
site sidebar, one entry per ADR). Both drifted from the files on disk before: ADRs 0017
and 0018 existed without being listed in the index, and two index rows had been written
as guesses without reading the ADR they described. Nothing failed, because nothing
checked. This file is that check: it fails as soon as an ADR is added without being
listed in both places, or a listed link points nowhere, so the omission is caught by the
test suite instead of by someone noticing an empty sidebar.

Most checks are plain string/regex checks over the Markdown files, so they run in the
same environment as the rest of the test suite (the docs toolchain, `mkdocs`,
`mkdocstrings`, ..., is an optional `docs` extra). The last test does build the site and
inspects the real sidebar; it is skipped when that toolchain is not installed. It exists
because a correct-looking `mkdocs.yml` was not enough: pointing the nav entry at
`changelog/SUMMARY.md` (the file) rendered one ordinary page titled "SUMMARY" instead of
expanding it, and no individual entry ever reached the sidebar. `literate-nav` only expands
a `SUMMARY.md` when the nav entry points at its *directory* (`adr/`).
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADR_DIR = _REPO_ROOT / "docs" / "adr"
_ADR_FILE_PATTERN = re.compile(r"^(\d{4})-.+\.md$")
_MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")


def _adrFiles() -> list[Path]:
    return sorted(path for path in _ADR_DIR.iterdir() if _ADR_FILE_PATTERN.match(path.name))


def _linkedTargets(markdown_path: Path) -> list[str]:
    return _MARKDOWN_LINK_PATTERN.findall(markdown_path.read_text(encoding="utf-8"))


def test_adr_numbers_are_contiguous_and_unique() -> None:
    numbers = [int(_ADR_FILE_PATTERN.match(path.name).group(1)) for path in _adrFiles()]  # type: ignore[union-attr]
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR numbers must run 0001..NNNN with no gap and no duplicate, got {numbers}."
    )


def test_every_adr_is_listed_in_the_index_table() -> None:
    linked = set(_linkedTargets(_ADR_DIR / "index.md"))
    missing = [path.name for path in _adrFiles() if path.name not in linked]
    assert not missing, f"docs/adr/index.md does not list: {missing}"


def test_every_adr_is_listed_in_the_sidebar_summary() -> None:
    linked = set(_linkedTargets(_ADR_DIR / "SUMMARY.md"))
    missing = [path.name for path in _adrFiles() if path.name not in linked]
    assert not missing, (
        f"docs/adr/SUMMARY.md (the sidebar) does not list: {missing}. Add one "
        f"'- [NNNN: title](file.md)' line per ADR."
    )


def test_summary_lists_the_adrs_in_numerical_order() -> None:
    order = [target for target in _linkedTargets(_ADR_DIR / "SUMMARY.md") if target != "index.md"]
    assert order == sorted(order)


def test_summary_starts_with_the_index_page() -> None:
    first_target = _linkedTargets(_ADR_DIR / "SUMMARY.md")[0]
    assert first_target == "index.md"


def test_every_link_in_the_index_and_the_summary_resolves_to_a_file() -> None:
    for markdown_name in ("index.md", "SUMMARY.md"):
        for target in _linkedTargets(_ADR_DIR / markdown_name):
            if target.startswith(("http://", "https://", "..")):
                continue
            assert (_ADR_DIR / target).is_file(), f"{markdown_name} links to missing '{target}'."


def test_index_rows_are_not_unverified_guesses() -> None:
    """Two rows were once committed as "*Guess:*" placeholders because the ADR they
    described had not been read. A row must describe the ADR as written."""
    index_text = (_ADR_DIR / "index.md").read_text(encoding="utf-8")
    assert "*Guess" not in index_text


@pytest.mark.parametrize(
    ("section", "directory"), [("Architecture Decisions", "adr/"), ("Changelog", "changelog/")]
)
def test_mkdocs_nav_uses_the_directory_form_so_literate_nav_expands_the_summary(
    section: str, directory: str
) -> None:
    """`- Section: adr/` (directory) makes literate-nav expand `adr/SUMMARY.md` into one
    sidebar entry per line; `- Section: adr/SUMMARY.md` or `adr/index.md` (a file) does
    not, and silently leaves the individual pages out of the sidebar."""
    mkdocs_text = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    pattern = rf"^\s*-\s*{re.escape(section)}:\s*{re.escape(directory)}\s*(#.*)?$"
    assert re.search(pattern, mkdocs_text, flags=re.MULTILINE), (
        f"mkdocs.yml nav must contain '- {section}: {directory}' (directory form)."
    )
    assert not re.search(
        rf"^\s*-\s*{re.escape(section)}:\s*{re.escape(directory)}(SUMMARY|index)\.md",
        mkdocs_text,
        flags=re.MULTILINE,
    )


_DOCS_TOOLCHAIN = ("mkdocs", "mkdocs_literate_nav", "mkdocs_gen_files", "mkdocstrings")


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in _DOCS_TOOLCHAIN),
    reason="the optional 'docs' extra (mkdocs, literate-nav, ...) is not installed",
)
def test_built_site_sidebar_lists_every_adr(tmp_path: Path) -> None:
    """Build the real site and check what a reader actually sees in the sidebar."""
    site_dir = tmp_path / "site"
    completed = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--quiet", "-d", str(site_dir)],
        cwd=_REPO_ROOT,
        env={**os.environ, "DISABLE_MKDOCS_2_WARNING": "true"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr

    sidebar_hrefs = set(
        re.findall(r'href="([^"#]+)"', (site_dir / "adr" / "index.html").read_text())
    )
    missing = [path.stem for path in _adrFiles() if f"{path.stem}/" not in sidebar_hrefs]
    assert not missing, f"ADR page(s) missing from the built sidebar: {missing}"
    assert not any("SUMMARY" in href for href in sidebar_hrefs), (
        "a SUMMARY page is being rendered as an ordinary page instead of being expanded"
    )
