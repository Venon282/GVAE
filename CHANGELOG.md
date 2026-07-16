# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial project scaffold: repository structure per spec §8.
- `AbstractEncoder`, `AbstractDecoder`, `AbstractFusion`, `AbstractAssembler`
  interfaces, each with a self-registration registry (`registerEncoder`,
  `registerDecoder`, `registerFusion`, `registerAssembler`).
- `LatentSpace` and `RoutingGraph` abstractions, plus `validateRoutingGraph`
  enforcing the two structural constraints from spec §2.2 (no orphan
  latent spaces; `sum`/`average` assemblers require matching
  dimensionality).
- `single.py` and `factorized.py` convenience presets for building common
  `RoutingGraph` topologies.
- `GlobalVae` model class assembling the `EN-L1-DN` (Phase-1 recommended
  default) configuration from a config dict.
- Unit tests for the registry pattern and for routing-graph validation.
- First of the 8 end-to-end integration tests called for in spec §10
  (`EN-L1-DN`, with dummy encoders/decoders/fusion).
- ADR 0001 documenting the choice of `EN-L1-DN` as the first
  configuration implemented.
- `pyproject.toml` with `ruff`, `mypy` (strict), and `pytest` configured,
  including the naming-convention override (`N802`/`N803`/`N806`
  disabled) required by spec §10.

### Fixed
- `validateRoutingGraph` now rejects a decoder that consumes more than
  one latent space but has no assembler assigned, instead of silently
  skipping the dimensionality check for it.
