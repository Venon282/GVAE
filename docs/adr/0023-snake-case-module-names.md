# 0023: Snake_case module names, with deprecation shims

**Status:** accepted
**Date:** 2026-10-11

## Context

Eight modules were named after the class they hold: `encoders/OneDCnnEncoder.py`,
`OneDCnnResidualEncoder.py`, `TwoDCnnEncoder.py`, `TwoDCnnResidualEncoder.py` and the four
matching files in `decoders/`. Spec §8 recorded that as the layout ("named after the class they
hold"), and ADR 0014 accepted the resulting `N999` (invalid module name) findings as "this
project's own established naming convention". Every other module of the package is snake_case
(`poe.py`, `concat_mlp.py`, `latent_plot.py`), and PEP 8 asks for lowercase module names. The
spec's naming rule (§10) is about classes, variables, functions and methods; it never says
anything about files.

The cost showed up when CI was added (roadmap P0-1): `ruff check .` reported exactly these eight
files, and the roadmap held them back as decision D-9 so the lint job could stay advisory. A lint
job that is always red is not a safety net, so the choice had to be made: keep the names and
silence the rule, or rename.

Import paths are public API under semantic versioning. Downstream code may well contain
`from global_vae.encoders.OneDCnnEncoder import OneDCnnEncoder`, so a plain rename is a breaking
change that would need 2.0.

## Decision

- **The eight modules are renamed to snake_case**, with `git mv` so their history is kept:
  `one_d_cnn_encoder.py`, `one_d_cnn_residual_encoder.py`, `two_d_cnn_encoder.py`,
  `two_d_cnn_residual_encoder.py` and `one_d_cnn_decoder.py`, `one_d_cnn_residual_decoder.py`,
  `two_d_cnn_decoder.py`, `two_d_cnn_residual_decoder.py`. Class names, registry names
  (`1d_cnn_encoder_v1` and the rest) and behaviour are unchanged. A module is named after its class
  in snake_case, which keeps the "one class per file, named after the class" reading of spec §8.
- **Each old path stays as a shim until 2.0.** The shim is a CamelCase file that imports the class
  from the new module, lists it in `__all__` and emits a `DeprecationWarning` that names the new
  path and says "removed in 2.0". Importing through the old path returns the very same class
  object, and registers nothing a second time, because the decorator only runs in the new module.
  The shims define no class or function of their own: one defined there would shadow the real one.
- **`encoders/__init__.py` and `decoders/__init__.py` import the new modules only**, so
  `import global_vae.encoders` never warns.
- **The `N999` exception is local.** Each shim carries a file-level `# ruff: noqa: N999` with its
  reason, and the exception disappears with the file at 2.0. No project-wide ignore is added, so
  a new CamelCase module is still flagged.
- **The API reference skips the shims** (`scripts/gen_ref_pages.py`), since each only re-exports a
  class its snake_case module already documents.
- **The CI `lint` job becomes blocking**, as roadmap P0-2 asks, because `ruff check .` and
  `ruff format --check .` now exit 0.
- **The spec is updated with the owner's approval** (decision D-9): §8 names the new files and
  says modules are snake_case, and §10 states the module rule next to the other naming rules.

## Consequences

- Existing imports keep working and warn. The next minor release (1.3.0) carries the rename, and
  2.0 deletes the eight shims and the `_isDeprecatedAlias` skip in `gen_ref_pages.py`.
- `tests/unit/test_deprecated_module_paths.py` pins the contract: each old path warns and names
  its replacement, returns the same class, does not register twice and holds only the alias; the
  package `__init__` files import no shim; and the only CamelCase files left under
  `src/global_vae` are the eight shims, so a new CamelCase module fails the suite as well as ruff.
- Older ADRs and changelog files still name the CamelCase files, and are left as written: they
  describe the repository as it was, and the old paths still import.
- A downstream user who turns `DeprecationWarning` into an error finds out at import time, which
  is the point. The fix is one line: replace the CamelCase module name by its snake_case one.

## Update: removed in 2.0.0

The eight shims, `tests/unit/test_deprecated_module_paths.py` and the `_isDeprecatedAlias` skip in
`scripts/gen_ref_pages.py` were deleted in release 2.0.0, as announced above. The old import paths now raise
`ModuleNotFoundError`; the fix is the one line the Consequences describe. The text above is left as written.
