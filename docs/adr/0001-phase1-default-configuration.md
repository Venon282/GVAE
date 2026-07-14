# 0001 — Phase 1 default configuration: EN-L1-DN

**Status:** accepted
**Date:** 2026-07-14

## Context

The framework supports 8 valid architecture configurations along three
independent axes (encoder cardinality, latent cardinality, decoder
cardinality) — see project spec §2.1. All 8 must remain selectable via
config; none is "the" architecture. Even so, a first configuration
had to be implemented end-to-end before the others, to validate the
registry pattern, the routing-graph validator, and the overall
assembly flow with a real (if minimal) forward/backward pass.

## Decision

Implement `EN-L1-DN` first: per-modality encoders, a single fused
latent space, per-modality decoders. This is the configuration the
spec itself flags as the "recommended Phase-1 default" — the classic
MVAE/MMVAE-style multimodal VAE family.

## Rationale

- It is the best-understood configuration in the literature, which
  lowers the risk of the *implementation* being wrong while the
  *architecture* is still being explored.
- It exercises the full pipeline end-to-end (multiple encoders ->
  Fusion -> one latent space -> multiple decoders) with the minimum
  number of moving parts, making it the fastest path to a working,
  testable skeleton.
- The other 7 configurations reuse the exact same building blocks
  (encoder/decoder/fusion registries, `RoutingGraph`, `LatentSpace`,
  assemblers) — none of them require re-deriving core abstractions,
  only extending `GlobalVae` (or introducing sibling model classes) to
  cover multi-latent routing and non-fused single/shared decoding.

## Consequences

- `models/global_vae.py` currently only assembles and runs
  `EN-L1-DN`. `tests/integration/test_en_l1_dn_default.py` covers it
  with dummy modules.
- The remaining 7 configurations from spec §2.1 are not yet
  implemented. Extending `GlobalVae` (or splitting it into per-topology
  model classes sharing the same building blocks) to cover them is the
  next milestone, tracked against the "integration test for each of
  the 8 architecture combinations" requirement in spec §10.
- This is not a narrowing of the framework's scope — the `RoutingGraph`
  validator in `latent/base.py` and the `latent/factorized.py` preset
  already support multi-latent topologies; only the model-assembly
  code in `GlobalVae` needs to grow to route through them.
