# 0002: Build `GlobalVae` from an explicit routing graph

**Status:** accepted
**Date:** 2026-07-15

## Context

ADR 0001 scoped the first implementation of `GlobalVae` to exactly the
`EN-L1-DN` configuration, with the routing graph and assembler
machinery in `latent/` reachable but unused by the model class itself.
A code review, followed by actually running the code end to end,
surfaced several problems with that first pass:

1. `GlobalVae.__init__` rebuilt a single-latent `RoutingGraph` by hand
   instead of calling the existing `single.buildSingleLatentRoutingGraph`
   preset, duplicating logic that already existed.
2. The model exposed exactly one fusion strategy for the entire model,
   which cannot express a graph with more than one latent space, where
   different latent spaces (or latent spaces fed by only one encoder)
   may need a different, or no, fusion strategy.
3. `GlobalVae.__init__` had a plain defect: it called
   `fusion_strategy(**fusion_kwargs)`, where `fusion_strategy` is a
   string, before looking it up in the registry, which raises
   immediately on construction.
4. `validateRoutingGraph` silently accepted a decoder that consumes
   more than one latent space but has no assembler assigned, instead
   of rejecting it.
5. `latent/assembler.py` held a base class, a registry, and all three
   concrete assemblers in one file, breaking the project's own
   one-class-per-file rule.
6. `modality_configs` assumed one encoder and one decoder always share
   the same key, which makes a single shared decoder (the `*-D1` rows
   of spec §2.1) impossible to express: there is no modality name to
   register it under.
7. Running the shared-plus-private preset end to end (built while
   fixing points 1 to 6) surfaced a further, deeper issue: it assigns
   one encoder to two latent spaces of different dimensionality.
   `AbstractEncoder.forward` returns a single `(mu, logvar)` pair, so
   reusing it for both latent spaces silently produced a latent tensor
   with the wrong dimension instead of two independent posteriors.

Given the project is explicitly meant to support all 8 configurations
in spec §2.1 without hardcoding any one of them into the model class
("Global" in the name), fixing these in place inside the
`EN-L1-DN`-only class would have meant a second rewrite as soon as a
second configuration was needed.

## Decision

Rebuild `GlobalVae` around an arbitrary `RoutingGraph` rather than a
single hardcoded latent space, and make the gaps above fail loudly
where they cannot yet be fixed properly:

- `GlobalVae.__init__(encoder_configs, decoder_configs, routing_graph,
  fusion_strategies, ...)` builds encoders, decoders, one fusion
  module per latent space fed by more than one encoder, and one
  assembler module per decoder consuming more than one latent space.
  `encoder_configs` and `decoder_configs` are separate dicts, so a
  single shared decoder can be registered under its own name.
- `GlobalVae.createSingleLatent(...)` is a convenience classmethod that
  builds the `EN-L1-DN` graph via the existing preset and derives
  `encoder_configs`/`decoder_configs` from a single `modality_configs`
  dict, covering the exact case ADR 0001 scoped as the Phase-1
  default, without duplicating the graph construction.
- `forward()` walks the graph generically: fuse (or pass through) each
  active latent space, sample it, assemble (or pass through) whatever
  latent spaces each decoder consumes, decode.
- `validateRoutingGraph` now rejects a decoder that consumes more than
  one latent space with no assembler assigned.
- `__init__` explicitly rejects, with `NotImplementedError` and a clear
  message, any encoder assigned to more than one latent space, rather
  than silently producing a wrongly shaped latent. This is a real
  capability gap, not a bug to paper over: fixing it properly requires
  `AbstractEncoder` to expose one `(mu, logvar)` pair per latent space
  it feeds (multiple projection heads on a shared trunk), which is a
  separate, larger interface change.
- KL aggregation across latent spaces moved to
  `losses.kl.computeTotalKlLoss`, which sums each latent space's own
  `klDivergence` with an optional per-space weight, instead of the
  model class hardcoding a single KL term.
- `latent/assembler.py` is superseded by an `assemblers/` subpackage,
  one class per file, matching `fusion/`.

## Consequences

- `EN-L1-DN` (the Phase-1 default) and, more generally, any topology
  where every encoder feeds exactly one latent space (`EN-L1-D1`,
  `EN-LN-D1`, `EN-LN-DN`) are reachable through `GlobalVae.__init__`
  today. This was verified by actually instantiating and running a
  model with two independent latent spaces and a single shared decoder
  end to end, gradients included.
- Two things remain unsupported, both by explicit, loud failure rather
  than silent wrong behavior:
  - `E1-*` rows (a single shared encoder fanning out to one or more
    latent spaces): no shared-trunk, multi-head encoder exists yet, as
    ADR 0001 already anticipated.
  - Encoder fan-out within `EN-*` rows, i.e. one encoder assigned to
    more than one latent space. This is exactly what
    `latent/shared_private.py`'s preset needs: the `RoutingGraph` it
    builds is valid, but passing it to `GlobalVae` raises
    `NotImplementedError` until encoders can expose one `(mu, logvar)`
    pair per latent space. Both gaps come from the same root cause:
    `AbstractEncoder` has one output, not one output per latent space.
- ADR 0001 is left unchanged: it correctly documents why `EN-L1-DN` was
  implemented first, which this decision does not revisit, only how
  the model class is built.
- `tests/integration/test_en_l1_dn_default.py` (referenced by ADR
  0001) needs updating to the new constructor signature; it was not in
  scope for this review since its current content was not available.
- `latent/factorized.py` and `latent/assembler.py` are superseded by
  `latent/shared_private.py` and the new `assemblers/` subpackage
  respectively, and should be deleted from the repository.
