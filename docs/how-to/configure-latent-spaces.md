# Configure encoders, decoders and latent spaces

A model config says which encoders and decoders exist, how many latent spaces there are, and what
feeds or reads each one. The schema is `ModelConfig`
([ADR 0025](../adr/0025-config-schema-for-several-latent-spaces.md)); `configs/model/` has a
working file for each case below, and `buildModelFromConfig` turns any of them into a
`GlobalVae`.

## Pick the shape

| You want | Write | Shipped file |
|---|---|---|
| One latent space; every modality is encoded and decoded back to itself | `latent_mode: single`, `modalities`, `single_latent` | `signal_single_latent.yaml`, `default.yaml` |
| One latent space; a decoder whose target is not an encoder input | `latent_mode: single`, `encoders`, `decoders`, `single_latent` | `signal_image_to_image.yaml` |
| Several independent latent spaces | `latent_mode: several`, `latent` | `signal_image_independent_latents.yaml` |
| Shared plus private latent spaces (encoder fan-out) | `latent_mode: several`, `latent` with a `head` on some `fed_by` edges | `signal_image_shared_private.yaml` |

`latent_mode` is explicit. `single` takes `single_latent` and refuses `latent`; `several` takes
`latent` and refuses `single_latent`. A section that would be ignored is an error.

## Name encoders and decoders

`modalities.<name>` declares one encoder and one decoder, both called `<name>`: the name is the
batch key the encoder reads and the decoder is trained against. When the two sides do not pair
up, declare them apart:

```yaml
encoders:
  signal:   {name: 1d_cnn_resnet_encoder_v1}
  image_in: {name: 2d_cnn_resnet_encoder_v1}
decoders:
  image_out: {name: 2d_cnn_resnet_decoder_v1}   # trained against batch["image_out"]
```

The two sections and `modalities` can be mixed in one file, as long as a name is not used twice
on the same side. A batch may carry keys with no encoder (here `image_out`): they are never
encoded, only used as targets.

## Several latent spaces

```yaml
latent_mode: several
latent:
  spaces:
    z_signal: {dim: 16, fed_by: [signal]}
    z_image:  {dim: 16, fed_by: [image]}
  decoders_consume:
    image_out: {spaces: [z_signal, z_image], assembler: concat}
```

- `fed_by` lists the encoders feeding a space. More than one encoder needs a `fusion`
  (`fusion: {strategy: poe}`, see [choose a fusion strategy](choose-a-fusion-strategy.md)); one
  encoder refuses it, since it would never run.
- `decoders_consume.<decoder>.spaces` lists what a decoder reads. More than one space needs an
  `assembler` (`concat`, `sum`, `average`, `weighted_sum` or `attention`; `sum`, `average` and
  `weighted_sum` need equal dimensions), and `kwargs` go to the assembler's constructor, next to the
  widths of the spaces, which the model supplies (see [the assemblers](#choose-an-assembler)). Every declared decoder needs an entry, and every
  space needs a decoder that reads it.
- Each space takes its own `regularizer` (`strategy` and `kwargs`), `kl_standard_normal` by
  default.

## One encoder, several latent spaces

An encoder produces one `(mu, logvar)`. Feeding it to two spaces as it is would give both the
same posterior, so each edge but one carries a latent head. Write the edge as a mapping:

```yaml
spaces:
  z_shared: {dim: 32, fed_by: [signal, image], fusion: {strategy: poe}}
  z_signal_private:
    dim: 8
    fed_by:
      signal: {head: linear}          # kwargs: {...} goes to the head's constructor
```

A list entry is an identity edge. A mapping entry builds a head (`identity` or `linear`; see
[add a strategy](add-a-strategy.md) for new ones). Two identity edges from one encoder are
refused. The shipped shared plus private model has one identity edge (to `z_shared`) and one
`linear` edge per encoder.

## Choose an assembler

A decoder that reads several latent spaces needs an `assembler` that merges the sampled vectors into
its one input. The model builds it from the widths of those spaces, so `kwargs` holds only the
strategy's own options.

| `assembler` | What it does | Equal widths | `kwargs` (defaults) | Learned |
|---|---|---|---|---|
| `concat` | joins the vectors side by side | no | none | no |
| `sum` | adds them | yes | none | no |
| `average` | takes their mean | yes | none | no |
| `weighted_sum` | adds them, each scaled by a learned weight per space | yes | `normalize: true` | `n` weights |
| `attention` | pools one token per space with a learned-query cross-attention | no | `output_dim` (the widest space), `num_heads: 1` | projections, query, attention |

```yaml
decoders_consume:
  image_out:
    spaces: [z_signal, z_image]
    assembler: attention
    kwargs: {output_dim: 32, num_heads: 4}   # output_dim must be a multiple of num_heads
```

- `weighted_sum` starts as `average` (`normalize: true`: the weights are a softmax over the spaces
  present, so they sum to 1) or as `sum` (`normalize: false`: raw weights starting at 1).
- With a modality left out of a pass (modality dropout, cross-modal reports), `sum`, `average`,
  `weighted_sum` and `attention` still decode from the spaces that remain. `concat` cannot, and its
  decoder is left out of that pass.
- The decoder's width is the assembler's `output_dim` (next section), so for `attention` the width
  you set is the width the decoder gets.

## Widths are derived

Leave `latent_dim` out of encoder and decoder `kwargs`.

- An encoder takes the `dim` of the space it feeds through an identity edge. If every edge goes
  through a head, set `latent_dim` on the encoder yourself.
- A decoder takes the width of what its assembler announces as its `output_dim`: the space's `dim`
  for one space, the sum for `concat`, the common `dim` for `sum`, `average` and `weighted_sum`,
  the `output_dim` kwarg of `attention` (the widest space by default).

An explicit value is kept, and in `several` mode a value that disagrees with the derived one is an
error.

## Beta per latent space

Beta is not part of the model config. It stays in `training.beta_schedules`, keyed by the latent
space names:

```yaml
training:
  beta_schedules:
    z_shared: {strategy: linear_warmup, kwargs: {warmup_steps: 1000}}
    z_signal_private: {strategy: linear_warmup, kwargs: {warmup_steps: 1000}}
```

A space without an entry uses `training.beta`. A key that is not a space of the model is logged
as a warning and has no effect. `configs/training/several_latents.yaml` has no schedule of its
own for that reason: the experiment file adds them.

## Read the errors

Every error starts with the dotted path of the field to edit:

```text
model.latent.spaces.z_shared.fed_by: unknown encoder 'audio'. Declared encoders (...): ['image', 'signal'].
model.latent.decoders_consume.joint.assembler: required, since decoder 'joint' consumes 2 latent spaces ...
model.encoders.signal.kwargs.latent_dim: 7 differs from dim=32 of latent space 'z_shared', ...
```

Unknown keys and wrong types are reported by OmegaConf with its own `full_key`. A registry name
that is not registered raises `KeyError` with the path of the field that holds it.
`validateModelConfig(config)` checks a config without building anything.

## Train it

Each pattern has an experiment file, with the data contract `configs/data/signal_image.yaml`:

```bash
python scripts/train.py --config-name experiment/signal_image_shared_private \
    data.train_path=/path/to/pairs \
    data.loader_factory=my_project.data:buildPairedDataloaders
```

Your `loader_factory` yields batches with `signal` and `image` (and `image_out` for
`signal_image_independent_latents`). See [train a model](train.md).
