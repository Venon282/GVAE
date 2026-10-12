# 0024: Encoder fan-out through latent heads

**Status:** accepted
**Date:** 2026-10-11
**Supersedes:** the encoder fan-out guard of
[ADR 0002](0002-generalize-global-vae-to-routing-graph.md) (the `NotImplementedError` in
`GlobalVae.__init__` and the two "remain unsupported" paragraphs about fan-out). The rest of
ADR 0002 stands, and ADR 0002 itself is left as written.

## Context

ADR 0002 made `GlobalVae.__init__` refuse any graph in which one encoder feeds two latent spaces.
The reason was sound: `AbstractEncoder.forward` returns a single `(mu, logvar)` pair, and reusing
that pair for two independent latent spaces gives them identical posteriors, and a wrongly shaped
tensor when their dimensions differ. The guard turned a silent wrong result into a loud failure.

The spec has always named the proper fix (§2.2, §12): encoder fan-out goes through a **latent
head**, a per-edge module that adapts the encoder's `(mu, logvar)` to the latent space that edge
feeds, "never by giving `AbstractEncoder` multiple output heads". Roadmap P1-1 delivered the heads
(`heads/`: `AbstractLatentHead`, the registry, `identity` and `linear`) as registered, tested
building blocks that nothing used yet. The `shared_private` preset still built a graph the model
refused, and spec §10 still listed a validation rule (a head's output dimension must match its
latent space) that could not be written because the graph had no heads.

Wiring the heads in raises one question the guard hid: what a decoder does when only some of the
latent spaces it consumes exist this pass. Under modality dropout (spec §5), the private latent of
a dropped modality has no encoder to feed it. Before fan-out this was rare, because the graphs the
model could build had no shared latent next to a per-modality private one.

## Decision

**The graph carries a head per edge.** `latent/base.py` gains `HeadSpec(name, out_dim, kwargs)` and
`RoutingGraph.encoder_heads: dict[str, dict[str, HeadSpec]]` (encoder name -> latent name -> head).
`name` is a key of the head registry, `identity` by default. `out_dim` is optional: when given it
is the output width the author intends, and validation compares it with the latent space's `dim`.
`kwargs` are extra constructor arguments for the head. `in_dim` and `out_dim` are never in `kwargs`:
the model derives them (below).

**An edge with no entry is an identity edge, and no module is built for it.** This keeps every
graph without heads building and running exactly as before, including graphs whose declared latent
`dim` was never checked against the encoder's width. Writing `HeadSpec("identity")` opts an edge
in to the width check.

**`validateRoutingGraph` gains three rules**, all about heads:

1. Every entry of `encoder_heads` names an edge that exists in `encoder_to_latents`.
2. A head's `out_dim`, when declared, equals the target latent space's `dim` (spec §10), and
   `kwargs` do not set `in_dim` or `out_dim`.
3. An encoder that feeds several latent spaces has at most one identity edge (no entry, or a head
   whose registry name is in `identity_head_names`, `{"identity"}` by default). Two identity edges
   would hand the same `(mu, logvar)` to two spaces: identical posteriors, the very failure ADR
   0002 guarded against. Heads whose parameters are initialised at random (`linear`) start
   different, so any number of them is accepted.

**`GlobalVae` builds one head module per entry of `encoder_heads` and applies it before fusion.**
Each head is built as `getHeadClass(name)(in_dim=encoder.latent_dim, out_dim=latent.dim, **kwargs)`,
so the input width is the encoder's own `latent_dim` and the output width is the latent space's
`dim`. The modules live in `GlobalVae.heads`, a `ModuleDict` of `ModuleDict`s (encoder -> latent ->
head), so a checkpoint names them `heads.<encoder>.<latent>.to_mu.weight`. An empty `heads` has no
state, so checkpoints written before this change still load. `forward` runs each encoder once,
then, per latent space, adapts every active feeding encoder's output through that edge's head, and
fuses the adapted outputs when several encoders are active. An encoder absent from `inputs` has no
active edges, so its heads do not run: modality dropout removes the edges with the encoder.

**The `shared_private` preset accepts head choices.** `buildSharedPrivateRoutingGraph` gains
`shared_head` (default `"identity"`) and `private_head` (default `"linear"`) and writes an explicit
`HeadSpec` for every edge. The default graph is therefore valid under rule 3 and buildable. This
changes what the preset returns by default: before, it returned two identity edges per encoder.

**A decoder with an incomplete input is skipped unless its assembler tolerates it.**
`AbstractAssembler` gains the property `handlesMissingLatents`, the assembler-side counterpart of
`AbstractFusion.handlesMissingModalities`. It is `True` for `sum` and `average`, which are defined
for any non-empty subset, and `False` by default, so for `concat` and for any assembler that does
not say otherwise. In `forward`, a decoder that consumes several latent spaces and has only some of
them available runs only if its assembler tolerates that; otherwise it is absent from
`"reconstructions"`, like a decoder with no input at all.

## Alternatives considered

- **Names only in the graph, like `decoder_assemblers`.** Simpler, but then the graph never knows a
  head's output width and rule 2 cannot be written in `validateRoutingGraph`, where spec §10 puts
  it. `HeadSpec.out_dim` is what makes the rule checkable without building the model.
- **A head module on every edge, identity included.** It matches "one head per edge" literally, but
  it would make every existing graph run the width check, and reject graphs that declared a latent
  `dim` the encoder never matched. An unmaterialised identity edge is what "identity by default"
  means in spec §2.2.
- **Impute a missing private latent with the prior (`mu = 0`).** It would let `concat` decode a
  dropped modality's reconstruction from the shared latent alone. It is a modelling choice with its
  own consequences for training, and spec §5 asks that a limitation of concatenation be documented,
  not patched silently. Skipping is the conservative default; imputation can be added as a
  strategy later.
- **Skip every decoder with an incomplete input, `sum` and `average` included.** It would change
  behaviour that works today (a sum over the available latent spaces), for no gain.

## Consequences

- The `NotImplementedError` is gone. The shared plus private topology (signal and image,
  `z_shared` through PoE, one private latent per modality through a `linear` head) builds, trains
  and evaluates through the existing `Trainer`, `evaluate` and cross-modal reporting, which needed
  no change: they already treat a latent space or decoder absent from a pass as normal.
- The `E1-*` rows (one shared encoder over several modalities) are still not buildable. They need a
  multi-modality encoder input and a decoder output split (roadmap P1-5); a single encoder that
  feeds several latent spaces through heads does work.
- With `concat` over `{z_shared, z_private_m}`, the decoder of a dropped modality `m` does not run
  in that pass. Cross-modal reports for a shared plus private model therefore show only the
  decoders of the modalities that were fed in, unless the model uses `sum` or `average`. This is
  the documented limitation of concatenation (spec §5), not a defect to patch in the model class.
- A custom assembler that tolerated a partial input before this change now has to say so, by
  overriding `handlesMissingLatents`. The three built-in assemblers are unaffected apart from
  `concat`, whose partial input could only fail on a width mismatch before.
- `fusion_kwargs` of a fused latent space that has heads must use the heads' output width
  (`HeadSpec.out_dim`, that is the latent space's `dim`) for its inputs. The model does not
  cross-check fusion kwargs against head widths.
- Exposing heads in YAML (`latent.mode: several`, `head: linear`) waits on roadmap P1-3. This ADR
  is the Python API; the config layer binds to `HeadSpec` later.
- `tests/integration/test_global_vae_construction_guards.py` inverted its fan-out class: the same
  graphs that raised now build, and the ones that must still fail (two identity edges) fail in
  `validateRoutingGraph` with a `ValueError`.
