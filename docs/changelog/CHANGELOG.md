# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Changed

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

### Removed

### Fixed
