# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `tests/integration/test_docs_changelog_navigation.py` (roadmap P0-7), the changelog counterpart of
  `test_docs_adr_navigation.py`. It fails when a `CHANGELOG_<version>_<date>.md` file is not listed
  in `docs/changelog/SUMMARY.md`, when the list is not newest first, when a link or a label
  (`x.y.z (YYYY-MM-DD)`) does not match the file it points to, when the `## [x.y.z] - date` heading
  inside a release file disagrees with its name, or when a changelog file name breaks the
  `CHANGELOG_<x.y.z>_<YYYY-MM-DD>.md` convention. A slow test builds the site with `mkdocs build
  --strict` and checks that every release is in the built sidebar. Run against the previous state of
  the repository, three of its tests fail (the underscore file name, the two missing entries and the
  order).
- `.github/workflows/ci.yaml` (roadmap P0-1, spec section 10): GitHub Actions workflow run on
  every push and pull request with three jobs. `lint` runs `ruff check` and
  `ruff format --check`, and is advisory (`continue-on-error`) until the ruff clean-up of
  P0-2 lands. `types` runs `mypy` on Python 3.11, the version `pyproject.toml` asks mypy to
  check against. `tests` runs `pytest` with coverage on Python 3.11 and 3.13 and fails under
  a 95% coverage floor, one point below the measured 96.25%. All jobs install `.[dev,docs]`
  (CPU build of torch), so the test that builds the real docs site no longer skips. A CI
  status badge is added to the README, and the Setup and Running checks sections of the
  README and of `docs/getting-started.md` now match the workflow.
- Tests for the gaps spec section 10 names (roadmap P0-4), 316 new tests in total. In
  `tests/unit/`: `test_routing_graph_validation.py` covers each of the four rejections of
  `validateRoutingGraph` (a latent space with no encoder, one with no decoder, a decoder over
  several latent spaces with no assembler, `sum` or `average` over different dimensions) and the
  accepting cases; `test_assemblers.py` checks the values, gradients and failure modes of
  `concat`, `sum` and `average` and the assembler registry error paths;
  `test_registry_contract.py` runs one parametrized contract over all nine registries (encoders,
  decoders, fusion, assemblers, regularizers, transforms, beta schedules, callbacks, loggers):
  register, look up, duplicate name raises `ValueError`, unknown name raises `KeyError` listing
  the available names, listing is sorted, built-ins are registered; `test_routing_graph_presets.py`
  covers `buildSingleLatentRoutingGraph` and `buildSharedPrivateRoutingGraph`. In
  `tests/integration/`: `test_configuration_matrix.py` is stage 1 of the configuration matrix
  (roadmap P1-6), the four `EN-*` rows without encoder fan-out built from the real 1D and 2D
  modules, with `concat`, `sum` and `average` on the two multi-latent rows, checking
  reconstruction and latent shapes, what each assembler received and returned, a finite loss and
  a gradient on every parameter, and failing if a row is dropped from its list;
  `test_global_vae_construction_guards.py` checks that `GlobalVae` validates the routing graph at
  construction and raises `NotImplementedError` for an encoder that feeds two latent spaces
  (roadmap P1-2 inverts that one). Coverage of `latent/` goes from 57% to 100% and of
  `assemblers/` from 79% to 100%; overall coverage of the full run is 97.5%.
- `tests/unit/test_typography.py` (roadmap P0-3, spec section 10): a guard that fails when a
  tracked or new (not git-ignored) `.py`, `.md`, `.yaml`, `.yml` or `.toml` file contains an em
  dash or a unicode arrow, so the house typography rule cannot regress. A character counts as an
  arrow when its Unicode name contains "ARROW", which covers every arrow block without a range
  table; en dashes, hyphens and the minus sign are allowed. The failure message lists each hit
  as `path:line:column` with the offending line. Files come from `git ls-files`, with a
  directory walk (skipping virtual environments and build output) outside a git checkout. It
  runs with the rest of `tests/` in the `tests` job of CI. Commit messages are not scanned.

### Changed

- Documentation hygiene (roadmap P0-7). Stale text is refreshed, and no code behaviour changes.
  `losses/NOTE.md` no longer says the reconstruction and regularization modules are deferred or
  cites the removed `computeKlLoss` (it now lists `regularizers/`, the beta schedules and the open
  loss-scale question of roadmap P2-1); `data/NOTE.md` describes `DataConfig.transforms` as the
  per-modality mapping it has been since ADR 0015; `visualization/NOTE.md` no longer says that no
  image decoder exists; `configs/experiment/NOTE.md` no longer cites a README section that is gone.
  The README drops `SignalEncoder` and `computeKlLoss` from its naming examples, refreshes
  "Extending beyond `EN-L1-DN`" (what the routing graph supports and the three remaining limits)
  and links the roadmap. The specification follows the code: section 8 shows the real tree
  (`OneDCnn*` and `TwoDCnn*` modules, `latent/routing_graph_builders/`, `beta_schedules/`,
  `callbacks/`, `loggers/`, `config/`, `evaluation/`, `visualization/`, with the unbuilt `heads/`,
  `weighted_sum` and `attention` marked as planned), section 9 uses real registry names instead of
  `signal_cnn_v1` and `resnet_encoder_v1`, and section 10 now reads "no bare `print`" as a rule for
  library code, so the 9 `print` calls that report results in `scripts/evaluate.py` and examples 01
  and 02 stay (roadmap decision D-11, default taken: this edit needs the owner's approval). The
  stale example names in the `encoders/registry.py` and `decoders/registry.py` docstrings, and the
  stale preset paths in `latent/base.py` and spec section 12, are fixed the same way. The
  `computeKlLoss` mention in the `computeRegularizationLoss` docstring is reworded.
- `configs/model/default.yaml` is now a buildable signal plus image model (`1d_cnn_encoder_v1` and
  `2d_cnn_encoder_v1` fused by PoE into one 32-dimensional latent space, both modalities
  reconstructed) instead of an unbuildable example naming a `resnet_encoder_v1` that was never
  registered. In `tests/integration/test_config.py`, the test that expected `model=default` to raise
  `KeyError` is replaced by tests that build it, run a forward pass with both modalities and with
  one, and check the unknown-name `KeyError` with an override naming an unregistered encoder.
- `docs/changelog/CHANGELOG_1.1.1_2026_09_27.md` is renamed `CHANGELOG_1.1.1_2026-09-27.md` (with
  `git mv`) so that every release file uses hyphens in its date, and releases 1.2.0 and 1.1.1 are
  added to `docs/changelog/SUMMARY.md`: both pages were built but had no sidebar entry.
- `computeTotalReconstructionLoss` now raises `ValueError` when a reconstruction and its target
  differ in shape (roadmap P0-4(g)). `torch.nn.functional.mse_loss` and the other built-in losses
  only warn and then broadcast, so a `(B, 1, H, W)` target against the 2D decoder's `(B, H, W)`
  output silently became a `(B, B, H, W)` comparison of every reconstruction with every target
  in the batch. The error names the modality and both shapes, and is raised before the loss
  function is called. `Trainer.computeLosses` and `evaluate` inherit it. A target with an extra
  channel axis now has to be squeezed to the decoder's output shape before it is used.

- Test layout and markers (roadmap P0-5, spec sections 8 and 10). 18 modules of pure unit tests
  moved from `tests/integration/` to `tests/unit/` with `git mv`, so their history is kept:
  `conv_math`, `conv_blocks`, the 1D and 2D encoders and decoders with their residual variants,
  the beta schedule, regularizer and callback registries, the reconstruction and regularization
  losses, `PlateauTracker`, `setGlobalSeed` and the data transforms. That is 1208 of the 1723
  tests. A module that builds a `GlobalVae`, runs a `Trainer`, composes a config into a model,
  or runs a script, an example or the docs build stays in `tests/integration/`. A `slow` marker
  is registered in `pyproject.toml` next to `--strict-markers` (an unregistered marker is now an
  error) and tags the 20 tests that run a script or an example in a subprocess, build the docs
  site or run UMAP. `pytest tests/unit` (about 5 seconds) and `pytest -m "not slow"` (about 30
  seconds, against about 150 for the full run) are the fast loops, documented in the README and
  in `docs/getting-started.md`. CI is unchanged and still runs the whole suite.

- Ruff clean-up (roadmap P0-2, spec section 10): `ruff check .` goes from 1071 findings to the 8
  `N999` ones (the CamelCase module names), which wait for decision D-9, so the CI `lint` job
  stays advisory until then. `ruff` and `mypy` are pinned in the `dev` extra with
  compatible-release specifiers (`ruff~=0.16.0`, `mypy~=2.4.0`; the `docs` extra gets the same
  `ruff` pin), so the finding counts no longer move with the tool version. `ruff check --fix`
  (85 fixes) and `ruff format` were applied; ruff 0.16 also formats the Python code blocks of
  the two `docs/guide-vae-accessible-*.md` files, which changed by one comment spacing. By hand,
  in `src/`:
  24 `D205` summary lines, 15 `D102` docstrings on overriding methods (the three assemblers'
  `forward`, the `latent_dim`, `modality_name` and `minimal_input_length` properties of the
  encoders and decoders), the `D417` parameter descriptions, the `D107` and `D100` gaps, and
  the garbled class docstring and indentation of `OneDCnnEncoder`. The mutable default
  `pool_kwargs={}` of `OneDCnnEncoder.__init__` (B006) is now `None`, resolved inside the
  constructor like in the other three CNN encoders, so instances no longer share one dict.
  In `scripts/` and `examples/`, the `D205` summaries were split, the four `D301` docstrings
  that contain backslashes became raw strings (the resulting `__doc__`, which is the
  `--help` text, is unchanged) and `main` of the first example got a docstring. In `tests/`,
  `D101` to `D104` are ignored through `[tool.ruff.lint.per-file-ignores]` (decision D-10) and
  the 98 `D205` summaries were split. No new `noqa` was needed.
- Typography sweep (roadmap P0-3, spec section 10): every em dash (93 of them, in 32 files) and
  every unicode arrow (3 in the README, 1 in the spec) is gone from the tracked text. Each was
  rewritten in context with a period, a colon, parentheses or `->`, changing punctuation only.
  The files are the README, the specification, the how-to guides and getting-started page, the
  ADRs, the ADR index and the changelogs, plus a few docstrings and comments in `src/`,
  `examples/` and `tests/`. No ADR decision text changed, so the rule against editing an ADR in
  place is not breached. ADR headings and index entries now read `NNNN: Title`, like
  `docs/adr/SUMMARY.md` and ADRs 0002 and 0016 to 0022 already did, and the changelog sidebar
  entries read `1.1.0 (2026-09-16)`. The Typography bullet of spec section 10, which named the
  banned characters themselves, now cites their code points (U+2014 and U+2192), names the new
  guard test and states the arrow rule it used to garble.

### Removed

### Fixed
