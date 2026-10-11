# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `tests/unit/test_deprecated_module_paths.py` and `tests/unit/test_ci_workflow.py` (roadmap
  P0-2). The first pins the deprecation shims below: each old import path warns and names its
  replacement, returns the very same class, registers nothing twice and defines nothing of its
  own, the `encoders` and `decoders` `__init__` files import no shim, and the only CamelCase
  module files left under `src/global_vae` are the eight shims. The second fails when
  `continue-on-error` returns to the CI workflow or one of its three jobs is dropped.
- `docs/adr/0023-snake-case-module-names.md`, the decision record of the rename below.

### Changed

- Snake_case module names (roadmap P0-2, decision D-9, ADR 0023). The eight encoder and decoder
  modules named after their class are renamed with `git mv`: `encoders/one_d_cnn_encoder.py`,
  `one_d_cnn_residual_encoder.py`, `two_d_cnn_encoder.py`, `two_d_cnn_residual_encoder.py` and
  `decoders/one_d_cnn_decoder.py`, `one_d_cnn_residual_decoder.py`, `two_d_cnn_decoder.py`,
  `two_d_cnn_residual_decoder.py`. Class names, registry names (`1d_cnn_encoder_v1` and the rest)
  and behaviour are unchanged. Every old path (`global_vae.encoders.OneDCnnEncoder`, ...) still
  imports: it is a shim that returns the same class and emits a `DeprecationWarning` naming the new
  path, and it is removed in 2.0. The shims carry a file-level `# ruff: noqa: N999` with its reason,
  and no project-wide ignore is added. `encoders/__init__.py` and `decoders/__init__.py` import the
  new modules, so `import global_vae.encoders` does not warn. `scripts/gen_ref_pages.py` skips the
  shims, so the API reference documents each class once. Imports in `src/`, `tests/` and the
  docs were updated; spec §8 and §10 and the README now state that modules are snake_case, with
  the owner's approval of D-9. Older ADRs and changelog files keep the CamelCase names they were
  written with.
- The CI `lint` job is blocking (roadmap P0-2). `ruff check .` and `ruff format --check .` exit 0
  now that the eight `N999` findings are gone, so `continue-on-error` is removed. The README and
  `docs/getting-started.md` no longer call the job advisory.

### Removed

### Fixed
