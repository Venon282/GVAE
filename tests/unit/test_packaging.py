"""Packaging checks (roadmap P0-6): version, dependencies, `py.typed` and `LICENSE`.

`pip install .` once failed on a clean environment because `data/transforms/resample.py`
imports numpy at module level while `pyproject.toml` never declared it, and `pydantic` was
declared while nothing imported it. The version also lived in two places that disagreed
(`pyproject.toml` said 0.1.0, the latest tag was 1.2.0). Nothing failed, because nothing
checked. These tests read `pyproject.toml`, the package and the changelog directory as plain
files, so they run in the same environment as the rest of the suite and need no build.

The end to end check ("`pip install .` in a clean environment imports `global_vae.config`,
`pip check` is clean") needs a fresh virtual environment and the network, so it stays a manual
step of the release checklist in the README.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import global_vae

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = _REPO_ROOT / "src" / "global_vae"
_CHANGELOG_DIR = _REPO_ROOT / "docs" / "changelog"
_RELEASE_FILE_PATTERN = re.compile(r"^CHANGELOG_(\d+)\.(\d+)\.(\d+)_\d{4}-\d{2}-\d{2}\.md$")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
# Distribution name -> import name, for the declared packages whose two names differ.
_IMPORT_NAMES = {"hydra-core": "hydra", "scikit-learn": "sklearn", "umap-learn": "umap"}
# Extras that only serve development or documentation: nothing in `src/` may rely on them.
_NON_RUNTIME_EXTRAS = frozenset({"dev", "docs"})


def _pyproject() -> dict[str, Any]:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _distributionName(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement.strip())
    assert match is not None, f"cannot read the package name of requirement {requirement!r}"
    return match.group(0).lower().replace("_", "-")


def _declaredRuntimeDistributions() -> set[str]:
    project = _pyproject()["project"]
    requirements = list(project["dependencies"])
    for extra, extraRequirements in project["optional-dependencies"].items():
        if extra not in _NON_RUNTIME_EXTRAS:
            requirements.extend(extraRequirements)
    return {_distributionName(requirement) for requirement in requirements}


def _moduleLevelThirdPartyImports() -> dict[str, set[str]]:
    """Top-level package name -> the files of `src/` importing it at module level.

    Only statements directly in a module body count. An import inside a function, a `try`
    block or an `if TYPE_CHECKING` block is optional or deferred, which is how the soft
    dependencies (tensorboard, scipy, scikit-learn, umap) are meant to be used.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name not in sys.stdlib_module_names and name != "global_vae":
                    found.setdefault(name, set()).add(path.relative_to(_REPO_ROOT).as_posix())
    return found


def _newestReleasedVersion() -> str:
    versions = []
    for path in _CHANGELOG_DIR.iterdir():
        match = _RELEASE_FILE_PATTERN.match(path.name)
        if match:
            versions.append(tuple(int(part) for part in match.groups()))
    assert versions, "docs/changelog/ holds no CHANGELOG_<version>_<date>.md file."
    return ".".join(str(part) for part in max(versions))


def test_version_is_a_plain_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", global_vae.__version__), (
        f"global_vae.__version__ is {global_vae.__version__!r}, expected MAJOR.MINOR.PATCH."
    )


def test_pyproject_reads_the_version_from_the_package() -> None:
    """One place to edit: `global_vae.__version__`. `pyproject.toml` holds no copy of it."""
    config = _pyproject()
    assert "version" not in config["project"], (
        "pyproject.toml has a static `version`; remove it so the version lives only in "
        "global_vae.__version__."
    )
    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "global_vae.__version__"}


def test_version_matches_the_newest_changelog_release() -> None:
    """Release checklist: the version, the newest `CHANGELOG_<x.y.z>_<date>.md` and the tag agree.

    The tag itself is outside the repository files, so the checklist covers it by hand. This
    covers the two files, which is where the 0.1.0 versus 1.2.0 drift happened.
    """
    assert global_vae.__version__ == _newestReleasedVersion(), (
        f"global_vae.__version__ is {global_vae.__version__!r} but the newest release file in "
        f"docs/changelog/ is {_newestReleasedVersion()!r}. Bump them together (see the release "
        f"checklist in README.md)."
    )


def test_numpy_is_a_declared_dependency() -> None:
    """`import global_vae.config` imports `resample.py`, which imports numpy unconditionally."""
    dependencies = {_distributionName(item) for item in _pyproject()["project"]["dependencies"]}
    assert "numpy" in dependencies


def test_pydantic_is_not_a_dependency() -> None:
    """ADR 0011 chose dataclasses and nothing imports pydantic: do not declare it again."""
    assert "pydantic" not in _declaredRuntimeDistributions()
    assert "pydantic" not in _moduleLevelThirdPartyImports()


def test_every_module_level_third_party_import_is_declared() -> None:
    """A package imported at module level must be a dependency or belong to a runtime extra."""
    declared = {
        _IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        for distribution in _declaredRuntimeDistributions()
    }
    undeclared = {
        name: sorted(files)
        for name, files in _moduleLevelThirdPartyImports().items()
        if name not in declared
    }
    assert not undeclared, (
        f"imported at module level in src/ but declared nowhere in pyproject.toml: {undeclared}"
    )


def test_py_typed_marker_ships_with_the_package() -> None:
    """PEP 561: without `py.typed` a downstream mypy ignores the package's type hints."""
    assert (_PACKAGE_DIR / "py.typed").is_file()
    packageData = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "py.typed" in packageData["global_vae"]


def test_license_file_matches_the_declared_license() -> None:
    declared = _pyproject()["project"]["license"]
    assert declared == {"text": "MIT"}
    licenseText = (_REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licenseText.startswith("MIT License")
    assert "Permission is hereby granted, free of charge" in licenseText
    assert "Copyright (c)" in licenseText
