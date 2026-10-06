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

### Removed

### Fixed
