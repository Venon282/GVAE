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

### Changed

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

### Removed

### Fixed
