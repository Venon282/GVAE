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
- `tests/integration/test_en_l1_dn_default.py` rebuilt against the
  `GlobalVae.createSingleLatent(...)` constructor introduced in ADR
  0002 (the previous version targeted the old `EN-L1-DN`-only
  constructor from ADR 0001 and was flagged as stale by ADR 0002's own
  "Consequences" section). Covers, with dummy per-modality
  encoders/decoders and a dummy PoE fusion: forward-pass shapes for
  every reconstruction and for the fused latent, missing-modality
  robustness (spec §5), KL-loss finiteness, gradient flow into every
  encoder and decoder, and the registry `KeyError` path. Passes `ruff
  check`, `ruff format --check`, and `mypy --strict`.
- `losses/regularizers/` subpackage: `AbstractLatentRegularizer` interface,
  its self-registration registry (`registerRegularizer` /
  `getRegularizerClass`), and the default `kl_standard_normal` strategy,
  extracted from `LatentSpace.klDivergence` so latent regularization is
  pluggable per latent space (spec §2.3), matching the same pattern
  already used for Fusion and Assemblers.
- ADR 0003 documenting the pluggable-regularization retrofit described
  below.
- `tests/integration/test_regularizers.py` (registry pattern and
  `kl_standard_normal` value correctness) and
  `tests/integration/test_regularization_loss.py` (cross-latent-space
  aggregation, beta weighting).
- `GlobalVae.createSingleLatent` gained a `regularizer_strategy`
  parameter (default `"kl_standard_normal"`), and
  `tests/integration/test_en_l1_dn_default.py` gained tests proving the
  default is wired in and that a non-default strategy is actually used,
  not just accepted.

### Changed
- Renamed `latent/factorized.py` to `latent/shared_private.py` and
  `buildFactorizedRoutingGraph` to `buildSharedPrivateRoutingGraph`,
  since "several latent spaces" is a general routing graph, not a
  hardcoded factorization scheme. And deplace them to `routing_graph_builders` directory
- Split `latent/assembler.py` (base class, registry, and all three
  concrete assemblers in one file) into its own `assemblers/`
  subpackage, one class per file, matching the rest of the codebase's
  modularity rule.
- `GlobalVae` is now built from an explicit `RoutingGraph` instead of
  assuming a single fused latent space fed by every encoder.
- `GlobalVae.__init__` takes separate `encoder_configs` and
  `decoder_configs` instead of one `modality_configs` dict, so a single
  shared decoder can be registered under its own name (the `*-D1` rows
  of spec §2.1) instead of reusing a modality name that may not exist.
- Fusion strategy is now selected per latent space (only required for
  latent spaces fed by more than one encoder), not once globally for
  the whole model.
- `GlobalVae.__init__` now accepts `regularizer_strategies` and
  `regularizer_kwargs`, and builds `self.regularizers` (an
  `nn.ModuleDict`, one entry per latent space, symmetric to
  `self.fusions`), instead of the model class going straight to
  `LatentSpace.klDivergence` for every latent space unconditionally.
- `losses/kl.py` is superseded by `losses/regularization.py`:
  `computeTotalRegularizationLoss` aggregates via each latent space's
  own regularizer module instead of calling `LatentSpace.klDivergence`
  directly.
- `GlobalVae.computeKlLoss` renamed to `computeRegularizationLoss`: it
  was never only about KL divergence in spirit (spec §2.3 always
  described the regularization term as pluggable), and now it isn't in
  code either.
- `LatentSpace` (`latent/base.py`) no longer exposes `klDivergence`;
  its module docstring no longer refers to "its own KL term" but "its
  own regularization term, not necessarily KL divergence".
- `models/global_vae.py` no longer imports the unused `LatentSpace`
  name from `latent.base` (a pre-existing dead import, `ruff` F401,
  noticed while editing this file's imports for the change above).

### Fixed
- `validateRoutingGraph` now rejects a decoder that consumes more than
  one latent space but has no assembler assigned, instead of silently
  skipping the dimensionality check for it.
- Fixed a typo in the decoder registry's error message ("Unknow" to
  "Unknown").
- `GlobalVae.__init__` no longer crashes on construction: the fusion
  strategy name was being called as a function before being resolved
  to a class.
- `GlobalVae.__init__` now explicitly rejects, with a clear
  `NotImplementedError`, an encoder assigned to more than one latent
  space, instead of silently reusing its output at the wrong
  dimension for the second latent space. Found by running the
  shared-plus-private preset end to end; see ADR 0002.

### Removed
- `assemblers/assembler.py`: a leftover duplicate of `AbstractAssembler`
  and the assembler registry, left behind at the new subpackage path
  after the split described above under "Changed".
- `losses/kl.py`: superseded by `losses/regularization.py` (see
  "Changed" above). Should be deleted from the repository.
- `LatentSpace.klDivergence`: superseded by
  `losses.regularizers.kl_standard_normal.KlStandardNormalRegularizer`,
  which is now the only place this exact computation lives.
