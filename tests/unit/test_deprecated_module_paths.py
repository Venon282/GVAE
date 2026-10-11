"""Deprecated CamelCase module paths (roadmap P0-2, decision D-9).

Eight encoder and decoder modules were named after the class they hold (`OneDCnnEncoder.py`),
which ruff flags as `N999` and which no other module of the package does. They were renamed to
snake_case (`one_d_cnn_encoder.py`). Import paths are public API under semantic versioning, so
each old path stays as a shim until 2.0: importing it still works, gives the very same class and
emits a `DeprecationWarning`. These tests pin that contract, so a shim cannot rot into a broken
import before 2.0, and nothing new can be added under a CamelCase name without a deliberate
choice.
"""

import ast
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from global_vae.decoders.registry import listRegisteredDecoders
from global_vae.encoders.registry import listRegisteredEncoders

_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "global_vae"

# (package, old CamelCase module and class name, new snake_case module name)
_RENAMED = [
    ("encoders", "OneDCnnEncoder", "one_d_cnn_encoder"),
    ("encoders", "OneDCnnResidualEncoder", "one_d_cnn_residual_encoder"),
    ("encoders", "TwoDCnnEncoder", "two_d_cnn_encoder"),
    ("encoders", "TwoDCnnResidualEncoder", "two_d_cnn_residual_encoder"),
    ("decoders", "OneDCnnDecoder", "one_d_cnn_decoder"),
    ("decoders", "OneDCnnResidualDecoder", "one_d_cnn_residual_decoder"),
    ("decoders", "TwoDCnnDecoder", "two_d_cnn_decoder"),
    ("decoders", "TwoDCnnResidualDecoder", "two_d_cnn_residual_decoder"),
]
_IDS = [cls for _, cls, _ in _RENAMED]
_SNAKE_CASE = re.compile(r"^_?[a-z][a-z0-9_]*$")


def _importOldPath(old_name: str) -> ModuleType:
    """Import `old_name` as if for the first time, so its module-level warning runs again."""
    sys.modules.pop(old_name, None)
    return importlib.import_module(old_name)


@pytest.mark.parametrize(("package", "cls", "new"), _RENAMED, ids=_IDS)
def test_old_path_warns_and_names_the_replacement(package: str, cls: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"global_vae\.{package}\.{new}") as recorded:
        _importOldPath(f"global_vae.{package}.{cls}")
    assert any(f"global_vae.{package}.{cls}" in str(w.message) for w in recorded)
    assert any("2.0" in str(w.message) for w in recorded)


@pytest.mark.parametrize(("package", "cls", "new"), _RENAMED, ids=_IDS)
def test_old_path_returns_the_very_same_class(package: str, cls: str, new: str) -> None:
    new_module = importlib.import_module(f"global_vae.{package}.{new}")
    with pytest.warns(DeprecationWarning):
        old_module = _importOldPath(f"global_vae.{package}.{cls}")
    assert getattr(old_module, cls) is getattr(new_module, cls)
    assert old_module.__all__ == [cls]


def _modulesImportedBy(package: str) -> set[str]:
    """The last name of every `import global_vae.<package>.<module>` in the package `__init__`."""
    tree = ast.parse((_PACKAGE_DIR / package / "__init__.py").read_text(encoding="utf-8"))
    return {
        alias.name.rsplit(".", 1)[-1]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith(f"global_vae.{package}.")
    }


@pytest.mark.parametrize("package", ["encoders", "decoders"])
def test_package_init_imports_the_new_modules_and_never_a_shim(package: str) -> None:
    """Importing `global_vae.encoders` must not warn, so it registers through the new paths."""
    imported = _modulesImportedBy(package)
    new_names = {new for pkg, _, new in _RENAMED if pkg == package}
    old_names = {cls for pkg, cls, _ in _RENAMED if pkg == package}
    assert new_names <= imported
    assert not old_names & imported


@pytest.mark.parametrize(("package", "cls", "new"), _RENAMED, ids=_IDS)
def test_registry_resolves_the_class_once_the_old_path_is_imported_again(
    package: str, cls: str, new: str
) -> None:
    """Re-importing a shim must not register a second time (that raises "already registered")."""
    before = (set(listRegisteredEncoders()), set(listRegisteredDecoders()))
    with pytest.warns(DeprecationWarning):
        _importOldPath(f"global_vae.{package}.{cls}")
    assert (set(listRegisteredEncoders()), set(listRegisteredDecoders())) == before


def test_every_module_name_is_snake_case_except_the_shims() -> None:
    """The only CamelCase module files left are the eight shims, and they are the old paths."""
    camel_case = sorted(
        path.relative_to(_PACKAGE_DIR).as_posix()
        for path in _PACKAGE_DIR.rglob("*.py")
        if not _SNAKE_CASE.match(path.stem) and path.stem != "__init__"
    )
    expected = sorted(f"{package}/{cls}.py" for package, cls, _ in _RENAMED)
    assert camel_case == expected


@pytest.mark.parametrize(("package", "cls", "new"), _RENAMED, ids=_IDS)
def test_shim_holds_only_the_alias(package: str, cls: str, new: str) -> None:
    """A shim defines nothing of its own: a class defined there would shadow the real one."""
    source = (_PACKAGE_DIR / package / f"{cls}.py").read_text(encoding="utf-8")
    assert not re.search(r"^(class|def) ", source, flags=re.MULTILINE)
    assert f"from global_vae.{package}.{new} import {cls}" in source
