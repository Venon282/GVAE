# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Latent heads (roadmap P1-1, spec §2.2). The new `global_vae.heads` package holds
  `AbstractLatentHead` (`base.py`: built from `in_dim` and `out_dim`, `forward(mu, logvar)` returns
  the adapted `(mu, logvar)`, positive-integer validation, a shape check shared by the concrete
  heads), the registry (`registry.py`: `registerHead`, `getHeadClass`, `listRegisteredHeads`) and
  two built-ins, one class per file. `identity` is the default: it returns the very tensors it
  receives and requires `in_dim == out_dim`. `linear` projects `mu` and `logvar` through two
  separate `nn.Linear` layers (`to_mu`, `to_logvar`); the initialisation is Xavier uniform weights
  and zero biases, so the prior `(0, 0)` maps to itself and two heads on one encoder start
  different. `heads/__init__.py` imports both modules. `GlobalVae` does not use heads yet:
  encoder fan-out is P1-2.
- `tests/unit/test_heads.py`: registry, shape, gradient, initialisation and identity-is-a-no-op
  tests. The latent head registry joins `tests/unit/test_registry_contract.py`, which now covers
  ten registries (`test_nine_registries_are_covered` became `test_ten_registries_are_covered`).
- `docs/how-to/add-a-strategy.md` covers latent heads: the table row, the constructor contract
  (`in_dim` and `out_dim` first) and the `forward` signature. The spec §8 tree no longer marks
  `heads/` as `(planned)`.
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
