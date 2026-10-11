"""Type-check configuration guard (roadmap P0-8).

`mypy` with no arguments must check the same code in CI as it does on a developer machine, and
it must check all of it. Three things can drift without anything failing:

- the scope: `scripts/`, `examples/` and `tests/` were once outside mypy, so 98 errors piled up
  there unseen while `src/` stayed strict clean;
- the relaxations: the only thing tests are excused from is the requirement to annotate every
  function, nothing else of `strict`;
- the interpreter: `python_version` in `[tool.mypy]` and the Python the CI `types` job runs on
  must agree and must be the oldest supported Python. numpy 2.5 requires Python 3.12, and its
  stubs make mypy fail on `python_version = "3.11"`, so moving one of the two alone breaks the job.

These tests read `pyproject.toml` and the workflow file as plain text, so they need no network and
no mypy run. They do not run mypy itself: that is the `types` job of `.github/workflows/ci.yaml`.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yaml"
_CHECKED_TREES = ("src", "scripts", "examples", "tests")
_JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
_PYTHON_VERSION = re.compile(r"""^\s*python-version:\s*["']?(\d+\.\d+)["']?\s*$""")
_MINIMUM_PYTHON = re.compile(r"^>=\s*(\d+\.\d+)")


def _pyproject() -> dict[str, Any]:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _mypyConfig() -> dict[str, Any]:
    config: dict[str, Any] = _pyproject()["tool"]["mypy"]
    return config


def _mypyOverrides() -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = _mypyConfig()["overrides"]
    return overrides


def _ciJobLines(job_name: str) -> list[str]:
    """The lines of one job of the CI workflow, from its header to the next job's header."""
    lines = _CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    in_jobs = False
    collected: list[str] = []
    collecting = False
    for line in lines:
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        header = _JOB_HEADER.match(line)
        if header:
            collecting = header.group(1) == job_name
            continue
        if collecting:
            collected.append(line)
    return collected


def _ciTypesJobPythonVersions() -> list[str]:
    return [
        match.group(1)
        for match in (_PYTHON_VERSION.match(line) for line in _ciJobLines("types"))
        if match
    ]


def _ciTypesJobCommands() -> list[str]:
    commands = []
    for line in _ciJobLines("types"):
        stripped = line.strip()
        if stripped.startswith("run: "):
            command = stripped.removeprefix("run: ").strip()
            if command not in {"|", "|-", ">", ">-"}:  # a block scalar, its lines follow
                commands.append(command)
    return commands


class TestScope:
    def test_mypy_is_strict(self) -> None:
        assert _mypyConfig()["strict"] is True

    def test_default_run_checks_every_tree(self) -> None:
        """`mypy` with no arguments covers `src`, `scripts`, `examples` and `tests`."""
        assert set(_mypyConfig()["files"]) == set(_CHECKED_TREES)

    @pytest.mark.parametrize("tree", _CHECKED_TREES)
    def test_every_checked_tree_exists_and_holds_python_files(self, tree: str) -> None:
        """A renamed or emptied directory must not leave a dead entry in `files`."""
        assert any((_REPO_ROOT / tree).rglob("*.py")), f"{tree}/ holds no Python file"

    def test_packages_is_not_set_alongside_files(self) -> None:
        """`packages` would narrow the default run back to `global_vae` alone."""
        assert "packages" not in _mypyConfig()
        assert "modules" not in _mypyConfig()


class TestRelaxations:
    def test_tests_are_excused_from_annotating_functions_and_nothing_else(self) -> None:
        overrides = [entry for entry in _mypyOverrides() if entry["module"] == "tests.*"]
        assert overrides == [{"module": "tests.*", "disallow_untyped_defs": False}]

    def test_no_override_silences_errors_wholesale(self) -> None:
        """`ignore_errors` on any module would hide exactly what this scope exists to find."""
        assert [entry for entry in _mypyOverrides() if entry.get("ignore_errors")] == []

    def test_untyped_mkdocs_gen_files_is_handled_for_the_docs_script_only(self) -> None:
        """`scripts/gen_ref_pages.py` is the one place `mkdocs_gen_files.Nav` is used."""
        by_module = {entry["module"]: entry for entry in _mypyOverrides()}
        assert by_module["mkdocs_gen_files"].get("implicit_reexport") is True
        assert by_module["gen_ref_pages"].get("disallow_untyped_calls") is False
        assert (_REPO_ROOT / "scripts" / "gen_ref_pages.py").is_file()


class TestInterpreter:
    def test_python_version_is_the_oldest_supported_python(self) -> None:
        requires = _pyproject()["project"]["requires-python"]
        match = _MINIMUM_PYTHON.match(requires)
        assert match is not None, f"cannot read the minimum Python from {requires!r}"
        assert _mypyConfig()["python_version"] == match.group(1)

    def test_ci_types_job_runs_on_the_python_version_mypy_checks_against(self) -> None:
        """The `types` job pins one interpreter, and it is the one `python_version` names."""
        assert _ciTypesJobPythonVersions() == [_mypyConfig()["python_version"]]

    def test_ci_types_job_runs_mypy_with_no_arguments(self) -> None:
        """Scope lives in `pyproject.toml`, so the job must not narrow it on the command line."""
        assert "mypy" in _ciTypesJobCommands()
        assert not [c for c in _ciTypesJobCommands() if c.startswith("mypy ")]


class TestWorkflowReader:
    """The helpers above read YAML as text, so they get their own checks."""

    def test_types_job_is_found(self) -> None:
        assert _ciJobLines("types"), "the `types` job was not found in ci.yaml"

    def test_the_reader_keeps_the_jobs_apart(self) -> None:
        """The `tests` job matrix lists 3.13, and that must not be read as the `types` job's."""
        assert any('"3.13"' in line for line in _ciJobLines("tests"))
        assert not any('"3.13"' in line for line in _ciJobLines("types"))
