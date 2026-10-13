# 0026: Assembler constructor contract and the learned assemblers

**Status:** accepted (the order of implementation is the default proposed for roadmap decision D-3,
`weighted_sum` first and `attention` second; it needs the owner's approval)
**Date:** 2026-10-13
**Supersedes:** the "built and run once on zero tensors" way of deriving a decoder width in
[ADR 0025](0025-config-schema-for-several-latent-spaces.md). The rest of ADR 0025 stands, and
ADR 0025 itself is left as written.

## Context

`GlobalVae` built every assembler with no arguments, `getAssemblerClass(name)()`. That was enough
for `concat`, `sum` and `average`, which have no parameters and no knowledge of the latent spaces
they merge. It cannot work for the two assemblers spec §2.2 still lists as wanted:

- `weighted_sum` holds one learned weight per latent space, so it must know how many there are.
- `attention` holds one projection per latent space, from that space's width to a common token
  width, so it must know every width, and it has an output width and a head count of its own.

Three further problems came with the same gap:

- **A wrong decoder width failed late.** The caller of `GlobalVae` had to add up the widths by hand
  for `concat` and write the sum into the decoder's `latent_dim`. A wrong value was accepted at
  construction and failed at the first forward pass with `mat1 and mat2 shapes cannot be
  multiplied`. ADR 0025 worked around it in the config layer only, by building the assembler and
  running it once on zero tensors to read the width of the result.
- **`validateRoutingGraph` named assemblers.** Its `dimension_locked_assemblers` argument defaulted
  to `{"sum", "average"}`, a hard-coded list of names in the function that is supposed to be
  generic over registered strategies. Spec §10 and §12 ask for the registry pattern: a new strategy
  must not need an edit in the validator. A third dimension-locked assembler, `weighted_sum`, would
  have been the first to need one.
- **A subset of the latent spaces lost its identity.** With a modality left out of a pass (spec
  §5), `forward` received a shorter list. For `sum` and `average` that does not matter. A strategy
  with one parameter per latent space has to know which spaces it was handed.

## Decision

**An assembler is built from the widths it receives.** `AbstractAssembler.__init__(input_dims)`
takes the `dim` of each latent space the decoder consumes, in order, validates them (a non-empty
sequence of positive integers) and exposes them as `input_dims`. Concrete classes accept
`input_dims` first and any further argument with a default, the same rule as latent heads (ADR
0024). `GlobalVae` builds each assembler as `getAssemblerClass(name)(input_dims, **kwargs)`, with
`input_dims` read from the new `RoutingGraph.decoderInputs()` and the kwargs from the existing
`assembler_kwargs` (ADR 0025). Nobody writes `input_dims` in a config.

**An assembler announces its output width.** `output_dim` is an abstract property: the last
dimension of what `forward` returns, a function of `input_dims` and the strategy's own arguments
only. `concat` returns `sum(input_dims)`, `sum`, `average` and `weighted_sum` return the common
width, and `attention` returns its `output_dim` argument, `max(input_dims)` by default.

**The decoder's `latent_dim` is checked, or filled in, when the model is built.** For each decoder,
`GlobalVae` computes the width it receives: the assembler's `output_dim`, or the `dim` of its latent
space when it consumes one and has no assembler. A `latent_dim` in the decoder's kwargs that
differs from it raises a `ValueError` naming the decoder, both widths and the source of the width.
When the kwargs omit it and the decoder's constructor has a `latent_dim` parameter, it is filled
in. A decoder whose constructor has no such parameter is built from its kwargs untouched. The
config layer reads `output_dim` instead of running the assembler on zeros, so the width formula
lives in each assembler and nowhere else; it builds the throwaway instance with the random number
generator state saved and restored, so resolving a config does not shift the initialisation of the
model that follows.

**The dimension rule is a class-level flag read through the registry.**
`AbstractAssembler.requires_matching_dims: ClassVar[bool] = False`. `sum`, `average` and
`weighted_sum` set it to `True`. The base constructor enforces it, so a flagged class refuses
differing widths even when built outside a model, and `validateRoutingGraph` reads it with
`getAssemblerClass(name).requires_matching_dims`. The function no longer names any assembler and
its `dimension_locked_assemblers` argument is removed. An assembler name that is not registered now
fails in the validator with the registry's `KeyError` (it lists the available names), where it used
to fail a few lines later in `GlobalVae`; the config layer looks the names up first, so its errors
keep the path of the field.

**`forward` receives the position of each latent space.** The signature becomes
`forward(latents, present=None)`. `present` is the position in `input_dims` of each tensor of
`latents`, or `None` when `latents` holds every space, in order. A shared helper,
`_resolvePresent`, checks it with the widths and returns the positions, so a wiring mistake
fails with a message naming the expected and received widths. It also refuses a strict subset
when `handlesMissingLatents` is `False`, which makes `concat` fail loudly if someone calls it with
a subset directly, instead of returning a narrower tensor. `GlobalVae.forward` passes `present`
when a decoder runs on a subset.

**An assigned assembler always runs.** `GlobalVae.forward` used to hand a decoder its latent
vector directly when only one of its latent spaces was present. For `sum` and `average` the
result is the same, but `weighted_sum` with `normalize=False` scales it, and `attention` projects
it to `output_dim`, which is the width the decoder was built for. A decoder with an assembler now
goes through it whatever the number of latent spaces present. The result for `sum`, `average` and
`concat` is value-identical, so existing models and checkpoints are unaffected.

**`weighted_sum`.** One learned scalar per latent space, `raw_weights` of shape `(n,)`. By default
(`normalize=True`) the weights are the softmax of the stored values over the spaces present in the
pass, initialised at zero, so the module starts as `average` and a subset keeps the scale the
decoder was trained with. With `normalize=False` the stored values are the weights, initialised at
1, so it starts as `sum`, and a subset is not renormalised. The dimension rule is the one of `sum`
(`requires_matching_dims`). It handles missing latents.

**`attention`.** One `nn.Linear(input_dims[i], output_dim)` per latent space turns each vector into
a token, one learned query attends over the tokens through `nn.MultiheadAttention`
(`batch_first`), and the attended vector is the output. No dimensionality requirement. Arguments:
`output_dim` (default `max(input_dims)`, must be a multiple of `num_heads`) and `num_heads`
(default 1, which divides any width, so the assembler builds with no argument for any `input_dims`).
A missing latent space contributes no token, so the attention runs over the tokens present and the
output keeps its width: it handles missing latents. The per-space projection, bias included, is
what tells the attention which space a token comes from.

**Order.** `weighted_sum` lands first (same rule as `sum`, `n` parameters), `attention` second (it
needs an output width and a head count), the order roadmap D-3 proposes. Both are registered and
imported in `assemblers/__init__.py`.

## Alternatives considered

- **Keep `AbstractAssembler()` argument-free and size lazily at the first forward** (`LazyLinear`
  style). It would hide the width from the model until a batch has been seen, so the decoder width
  could not be checked or filled in at construction, which is the point of the change.
- **A width formula per assembler in the config layer or in `GlobalVae`.** It repeats what
  `output_dim` says and needs an edit for every new strategy, against spec §12 (ADR 0025 already
  rejected it for the same reason).
- **Keep `dimension_locked_assemblers` as an optional override of the registry flag.** Two sources
  of truth for one rule, and the default would still name assemblers if it were kept as a set.
- **A name set on the validator that registered strategies add themselves to.** Global mutable
  state in a validator, and a strategy could be registered without the rule. A class attribute
  lives next to the code it describes and is inherited.
- **Pass latents by name (`dict[str, Tensor]`) instead of a list and `present`.** It would also
  identify the spaces, but changes `forward` for every assembler and ties assemblers to latent
  space names, which the same assembler class reuses across models with different names.
- **Raw, unconstrained weights as the only `weighted_sum` mode.** Simplest, but a subset would
  silently shrink the decoder input by the weights of the missing spaces. The softmax mode keeps the
  scale, and `normalize=False` remains for those who want the raw form.
- **Several learned queries, layer norm or a feed-forward block in `attention`.** More capacity, but
  also more choices to expose before anyone has asked for them. The module is one cross-attention
  pooling, which `output_dim` and `num_heads` already scale.

## Consequences

- All five assemblers named in spec §2.2 exist, and spec §11's second bullet is answered. `GlobalVae`
  can build `EN-LN-DN` and `EN-LN-D1` with a learned assembler, from Python or from YAML
  (`decoders_consume.<decoder>.assembler` and `kwargs`, ADR 0025).
- **Breaking, for code that builds assemblers directly.** `getAssemblerClass(name)()` and
  `ConcatAssembler()` now need `input_dims`; a custom assembler must accept `input_dims`, implement
  `output_dim` and take `present` in `forward`. `validateRoutingGraph` no longer accepts
  `dimension_locked_assemblers`. Built-in assemblers have the same behaviour for the same inputs.
  Models and checkpoints built through `GlobalVae` are unaffected: the parameter-free assemblers
  have no state, and the model's constructor arguments did not change.
- A decoder `latent_dim` that disagrees with the routing graph is now a construction error for a
  single latent space too, not only for assemblers. A model that passed this check before and
  worked is not affected, since a wrong width failed at the first forward pass.
- An assembler entry for a decoder that consumes no latent space is skipped (no module is built):
  there is nothing to size it from, and it never ran.
- `weighted_sum` and `attention` add parameters to the model (`assemblers.<decoder>.raw_weights`,
  `assemblers.<decoder>.projections.<i>.*`, `.query`, `.attention.*`), so a checkpoint carries them.
- `docs/how-to/add-a-strategy.md` documents the contract for anyone adding an assembler.
- Roadmap P1-4 is done. P1-6 stage 1 now runs the `LN` rows once per registered assembler, five in
  all.
