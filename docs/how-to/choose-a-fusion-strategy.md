# Choose and configure a fusion strategy

A latent space fed by more than one encoder needs a **fusion strategy** (spec §4): a module
that combines each encoder's `(mu, logvar)` into one posterior, *before* sampling. A latent
space fed by exactly one encoder needs none. The strategy is picked by registry name, per
latent space, and changing it never requires touching `GlobalVae`.

Four strategies are built in. To write your own, see
[Add a fusion / assembler / ...](add-a-strategy.md).

## At a glance

| Name | Idea | Missing modalities (spec §5) | Extra constructor arguments |
|---|---|---|---|
| `poe` | Multiply each modality's Gaussian expert (MVAE). Precision-weighted average; the prior is added as an extra expert by default. | Native: the missing expert's term is dropped. | none required (`eps`, `include_prior_expert`) |
| `moe` | Mixture of the experts (MMVAE), reduced to one Gaussian by moment matching. | Native: the missing component is dropped. | none required (`modality_weights`, `eps`) |
| `concat_mlp` | Concatenate every `(mu, logvar)` and project through an MLP. Simple baseline. | **Not native.** Explicit scheme: zero imputation plus presence mask. | `latent_dim`, `modality_dims` (required) |
| `cross_attention` | One token per active modality, fused by a Transformer encoder, mean-pooled. | Native: the missing token is omitted. | `latent_dim` (required) |

`AbstractFusion.handlesMissingModalities` reports the "native" column programmatically.

## Which one should I use?

- **Start with `poe`.** It has no parameters to learn, works with any subset of modalities,
  and is what the milestone-2 example uses.
- **`moe` if the modalities can genuinely disagree.** PoE's fused variance only ever shrinks
  as experts are added, so a confident but wrong modality is not visible in the output. `moe`
  widens the fused variance when the experts' means disagree.
- **`concat_mlp` as a learned baseline** when you want to check whether a learned combination
  beats the closed-form ones. It is the only strategy that needs the modality set fixed up
  front, and it only learns to cope with a missing modality if you train with
  `modality_dropout_p > 0`.
- **`cross_attention` when modalities may interact non-trivially**, or when you expect to
  scale toward larger backbones (spec §7). It has the most parameters and the most
  hyperparameters, so it is the one most worth tuning.

## Configuration

`fusion.kwargs` is forwarded unchanged to the strategy's constructor. Unlike an encoder's or
decoder's `latent_dim`, the fusion's is **not** filled in for you (`poe` and `moe` do not
accept it), so repeat it where a strategy needs it. In every example below the latent
dimension is `128` and the modalities are `signal` and `image`.

```yaml
model:
  modalities:
    # Decoder kwargs such as output_length / output_shape are data-dependent and must be
    # given too (omitted here to keep the focus on fusion).
    signal: {encoder: {name: 1d_cnn_encoder_v1}, decoder: {name: 1d_cnn_decoder_v1}}
    image:  {encoder: {name: 2d_cnn_encoder_v1}, decoder: {name: 2d_cnn_decoder_v1}}
  latent_mode: single
  single_latent:
    dim: 128
    fusion:
      strategy: moe            # poe | moe | concat_mlp | cross_attention
      kwargs: {}
```

The three strategies with extra arguments:

```yaml
# concat_mlp: every modality's encoder output width must be declared up front.
fusion:
  strategy: concat_mlp
  kwargs:
    latent_dim: 128
    modality_dims: {signal: 128, image: 128}   # slot order = this order
    hidden_dims: [256]
    use_presence_mask: true

# cross_attention: known_modalities adds a learned embedding per modality.
fusion:
  strategy: cross_attention
  kwargs:
    latent_dim: 128
    num_heads: 4               # must divide d_model (default: latent_dim)
    num_layers: 2
    known_modalities: [signal, image]

# moe: optional fixed weights, renormalized over the modalities present in each call.
fusion:
  strategy: moe
  kwargs:
    modality_weights: {signal: 2.0, image: 1.0}
```

From Python, the same choice is one argument:

```python
model = GlobalVae.createSingleLatent(
    modality_configs=modality_configs,
    latent_dim=128,
    fusion_strategy="cross_attention",
    fusion_kwargs={"z_fused": {"latent_dim": 128, "num_heads": 4}},
)
```

With an explicit `RoutingGraph`, pass `fusion_strategies={"z_shared": "moe"}` and
`fusion_kwargs={"z_shared": {...}}`: fusion is chosen per latent space, so two latent spaces
in the same model can use different strategies.

## Residual connection (`residual: true`)

Spec §4 lets any fusion assignment use a residual connection, as one flag next to the strategy.
It works the same way for all four strategies, because it wraps whichever one you chose:

```yaml
fusion:
  strategy: moe
  residual: true          # off by default
```

```python
model = GlobalVae.createSingleLatent(..., fusion_strategy="moe", fusion_residual=True)
# or, with an explicit routing graph: GlobalVae(..., fusion_residual={"z_shared": True})
```

The fused posterior then starts as the plain **mean of the active experts** and two learned
gates (one for `mu`, one for `logvar`, both starting at `0`) open the strategy's own correction:
`fused = mean + gate * (strategy - mean)`. A gate of `1` is exactly the strategy alone.

- **Why not `strategy + mean`?** Every strategy returns an absolute posterior, so adding the mean
  would count the experts twice (two identical `poe` experts of mean `m` would give `2m`).
  ADR 0021 explains the choice and the alternatives.
- **It changes the architecture.** The strategy moves under `inner` and two gate parameters are
  added, so a checkpoint saved with the flag off will not load with it on, and the reverse.
- **It is only valid where a fusion exists.** Enabling it on a latent space fed by fewer than two
  encoders raises `ValueError` at construction.
- **It needs the experts and the fused posterior to share one width.** A `concat_mlp` whose
  `latent_dim` differs from its experts' width has nowhere to add the skip path and raises at
  `forward`.
- **Missing modalities still work**: the mean is taken over the modalities present, and support
  is inherited from the wrapped strategy.
- **Prefer Adam over plain SGD** when it is on: with the gates at `0` the strategy's own
  parameters get no gradient on the first step and then a small one, which Adam rescales and SGD
  does not (measured in ADR 0021).

## Behaviour worth knowing

- **`moe` returns a single Gaussian**, the one matching the mixture's mean and variance. It
  cannot represent a bimodal posterior. See ADR 0020 for why.
- **With one active modality**, `GlobalVae` does not call fusion at all: that modality's own
  `(mu, logvar)` is used directly (this is `GlobalVae.forward`, not the strategy). Fusion only
  runs when two or more encoders are active. `moe` with a single expert would return it
  unchanged anyway.
- **`concat_mlp` rejects an undeclared modality** with `KeyError`, and `cross_attention` with
  `known_modalities` rejects one with `ValueError`. Both are configuration errors surfaced
  early rather than silently ignored.
- **Modality dropout** (`training.modality_dropout_p`, spec §5) is the training technique that
  makes any of these robust to missing inputs, and the only thing that teaches `concat_mlp`
  to use its presence mask.

## Comparing strategies

Only `fusion.strategy` (and its `kwargs`) changes between runs, so the simplest comparison is
one experiment file per strategy, each composed from the same `model`/`data`/`training`
groups. Then run `scripts/evaluate.py` on each checkpoint: for any model with more than one
modality it also writes a cross-modal reconstruction matrix
(`docs/adr/0016-cross-modal-reconstruction-reporting.md`), which shows directly how each
strategy behaves when a modality is withheld.

See also: [ADR 0020](../adr/0020-additional-fusion-strategies.md) and spec §4, §5.
