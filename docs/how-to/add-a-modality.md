# Add a new modality

A modality is a kind of data (a signal, an image, a table, a graph...). To add one you write an
encoder and a decoder, register them by name, and point a config file at them. The framework does
the rest.

!!! important "No core file should change"
    Adding a modality must never require touching the core (`models/global_vae.py`, `latent/`,
    `losses/`). If it does, the registry pattern is being bypassed somewhere: flag it instead of
    special-casing your modality in.

**On this page**

- [The five steps](#the-five-steps)
- [Which example should I copy?](#which-example-should-i-copy)
- [The worked examples](#the-worked-examples)
- [One encoder or decoder for several modalities](#one-encoder-or-decoder-for-several-modalities)

## The five steps

1. **Write the encoder.** Subclass [`AbstractEncoder`](../reference/global_vae/encoders/base.md) in
   `encoders/`. It maps your modality's raw input to a `(mu, logvar)` pair. See
   `encoders/one_d_cnn_encoder.py` for a worked example (length-agnostic, through adaptive pooling).
2. **Write the decoder.** Subclass [`AbstractDecoder`](../reference/global_vae/decoders/base.md) in
   `decoders/`. It maps a latent vector back to a reconstruction of your modality.
3. **Register both.** Decorate them with `@registerEncoder("your_name")` and
   `@registerDecoder("your_name")`, and add the imports to `encoders/__init__.py` and
   `decoders/__init__.py`. A registered class only exists once its module has been imported.
4. **Add a config.** A model file that names the two registered classes, plus a data file and an
   experiment if you want a ready-to-run recipe. Start from `configs/model/signal_single_latent.yaml`.
5. **Add tests.** A unit test for the forward-pass shapes and the gradient flow, and an entry in
   the relevant integration test if your modality changes which of the 8 architecture
   configurations (spec §2.1) are exercised end to end.

## Which example should I copy?

Six modalities were added after the core was written, each with **zero core changes**, and each
shows a different situation. Find yours in the first column.

| Your situation | Copy | What it teaches | ADR | Guide |
|---|---|---|---|---|
| Nothing like it is registered yet | Tabular | The simplest complete case: two new classes and a `loader_factory` | [0039](../adr/0039-tabular-modality.md) | [Train on a table](train-on-a-table.md) |
| An existing family nearly fits | Audio | Check the registered classes first, and add only what they lack | [0040](../adr/0040-audio-modality.md) | [Train on audio](train-on-audio.md) |
| The tensor layout is the hard part | Text | The decoder's output and the target must share one shape | [0041](../adr/0041-text-modality.md) | [Train on text](train-on-text.md) |
| Only the encoder is missing | Time series | Carry a mask inside the batch tensor (`NaN`) | [0042](../adr/0042-time-series-modality.md) | [Train on time series](train-on-time-series.md) |
| The registered family already fits | Graph | Measure first, then add one thin class for the loss | [0043](../adr/0043-graph-modality.md) | [Train on graphs](train-on-graphs.md) |
| The loss is the hard part | Point cloud | Put an order-free loss in `utils/` and set it from Python | [0044](../adr/0044-point-cloud-modality.md) | [Train on point clouds](train-on-point-clouds.md) |

Whatever you copy, **two habits** carry over:

- **Measure before you build.** Each ADR above compares the registered classes with new designs
  before adding anything, and says plainly where the result departs from the checklist.
- **Test that the core did not change.** Every integration test below reads the source of the core
  packages and checks that none of them names the new classes or the new batch key. It is a useful
  test to copy for your own modality.

## The worked examples

### Tabular data: a modality that is not a grid

The signal and the image were written together with the core, so they did not test the steps above.
Tabular data ([ADR 0039](../adr/0039-tabular-modality.md)) is the first modality that followed them
afterwards.

| Step | File |
|---|---|
| 1 to 3 | `encoders/tabular_mlp_encoder.py`, `decoders/tabular_mlp_decoder.py` and their two import lines |
| 4 | `configs/model/tabular_single_latent.yaml`, with a data file and an experiment |
| 5 | `tests/unit/test_tabular_backbone.py` and `tests/integration/test_tabular_model_configs.py` |

[Train on a table](train-on-a-table.md) shows the part that is not in the two classes: the
`loader_factory` that builds the rows.

### Audio: when an existing family nearly fits

A waveform is a 1D series, so the first thing to check is whether a class that is already registered
does the job ([ADR 0040](../adr/0040-audio-modality.md)). Here the 1D classes were not enough, for
three reasons:

- a decoder must reach `output_length` exactly;
- the output must be bounded;
- the stages need one dilation each.

So the modality adds `encoders/audio_waveform_encoder.py` and `decoders/audio_waveform_decoder.py`
and leaves the 1D classes as they were. It is also the example of a modality whose most common
alternative needs no class at all: a **spectrogram is an image**, and
`tests/integration/test_audio_model_configs.py` runs it through the 2D classes.
[Train on audio](train-on-audio.md) shows the `loader_factory` that cuts clips to one length and keeps
them within `[-1, 1]`.

### Text: when the batch layout is the hard part

For text, the first thing to settle is not the network but the shape of the tensors. A decoder's
reconstruction and its target (the batch tensor of the same name) must have one shape. So a decoder
that gives a distribution over the vocabulary at every position needs a one-hot target,
`(batch, length, vocab_size)`, and the encoder reads that same tensor.

The text modality ([ADR 0041](../adr/0041-text-modality.md)) adds:

- `encoders/text_transformer_encoder.py` and `decoders/text_transformer_decoder.py`, built from the
  transformer backbones' pieces (`utils/transformer_blocks.py`);
- `utils/text_blocks.py`, with the token embedding (which takes ids or one-hot rows) and the padding
  mask.

It is the example of a modality whose loss the YAML cannot name exactly (a cross-entropy is not in
the loss table), and of how far a registered class and a Python loss function get without touching
the core. [Train on text](train-on-text.md) shows the `loader_factory` that tokenizes, builds the
vocabulary, pads and one-hot encodes.

### Time series: when the decoder already exists

A multichannel series of any length is already covered by the registered 1D classes, so the useful
question ([ADR 0042](../adr/0042-time-series-modality.md)) was what they leave out: **values that were
not observed**. The answer is one encoder, `encoders/time_series_encoder.py`, and one utility module,
`utils/series_blocks.py`. The decoder stays `1d_cnn_decoder_v1`: the checklist asks for a decoder,
and an existing one met it because it measured clearly better than the alternative.

What to copy is **where the mask travels**. The loss sees only a reconstruction and the batch tensor,
so a `NaN` in that tensor is the mask, for the encoder and, through `computeMaskedSquaredError`, for
the loss. There is no second tensor and no change to the core.

Two more lessons:

- The YAML experiment is only half of the story: the `mse` of a YAML file is `NaN` on a gap, so
  series with gaps train through a Python `Trainer`.
- A measurement went against the first guess: the new encoder matched a zero-filled one on what was
  observed and led only on the padded tail.

[Train on time series](train-on-time-series.md) shows the `loader_factory` that cuts windows and
standardizes per channel, and the Python route.

### Graphs: when the family already fits

A graph of at most `N` nodes is a dense adjacency matrix `(batch, N, N)`, which is an image to the 2D
convolutions. So the first thing to do ([ADR 0043](../adr/0043-graph-modality.md)) was to measure what
the registered 2D classes already do, against graph-specific encoders and decoders. **None beat the
registered pair.** The encoder is `2d_cnn_encoder_v1` as it is, and the one class added is
`decoders/graph_cnn_decoder.py`, a thin subclass of `TwoDCnnDecoder` that symmetrizes, applies a
sigmoid and zeroes the diagonal.

That class is justified by the loss and not by the accuracy: the raw decoder cannot use the `bce` of
a YAML file, which measured better than its `mse`. It is the example to copy when an existing family
nearly fits: measure the existing classes first, add only what the measurements ask for, and say
plainly where the result departs from the checklist (here, no new encoder).

It also shows a property no class can supply. The model compares matrices position by position, so
**the order of the nodes matters as much as the choice of architecture**, and that order is the
loader's job. [Train on graphs](train-on-graphs.md) shows the `loader_factory` that orders and pads
the nodes, and how to score the edges.

### Point clouds: when the loss is the hard part

A point cloud is a set of `N` points. The tensor `(batch, N, 3)` fits the one-tensor-per-modality
contract, but the comparison does not: every loss a YAML file can name pairs row `i` of the
reconstruction with row `i` of the target, and for a set those are unrelated points
([ADR 0044](../adr/0044-point-cloud-modality.md)). Measured, the model trained with a YAML loss ended
farther from the clouds than one that ignores its input.

So the modality has:

- two registered classes: `encoders/point_cloud_encoder.py` (a pooling over the points that leaves
  out the rows of `NaN`, exactly invariant to their order) and `decoders/point_cloud_decoder.py` (a
  learned template folded by the latent vector), built from `utils/mlp_blocks.py` and
  `utils/transformer_blocks.py`;
- one function that is not a class, `utils/point_cloud_metrics.computeChamferDistance`.

The trainer takes any Python function as its reconstruction loss, so the core did not change. The
experiment file only composes the model, the data and the optimizer, and its header says its own loss
is a placeholder.

Copy this example when the hard part is the loss and not the network:

1. Measure what the loss of a YAML file does before building anything.
2. Put the loss in `utils/` as a function, with its tests.
3. Flag the missing table entry instead of editing the core.
4. Compare the new classes with the registered ones honestly. Here the registered 1D pair came
   within a few percent, and the new classes are justified by the invariance, the `NaN` convention
   and the cost.

[Train on point clouds](train-on-point-clouds.md) shows the `loader_factory` and the Python route.

## One encoder or decoder for several modalities

The steps above give each modality its own encoder and decoder (`EN-*` and `*-DN` in spec §2.1). To
have **one encoder ingest several modalities**, or **one decoder reconstruct several**, subclass the
shared interfaces instead ([ADR 0027](../adr/0027-shared-encoders-and-decoders.md)). Everything else
stays: the same registries, the same decorators, the same `__init__.py` imports.

### The steps

1. **Subclass [`AbstractSharedEncoder`](../reference/global_vae/encoders/shared_base.md).** Declare
   `input_modalities` (the batch keys it reads) and `latent_dim`. `forward` receives a
   `dict[str, Tensor]` and returns `(mu, logvar)`.
2. **Subclass [`AbstractSharedDecoder`](../reference/global_vae/decoders/shared_base.md).** Declare
   `output_modalities`. `forward` receives one latent tensor and returns a `dict[str, Tensor]` with
   exactly those keys.
3. **Register** with `@registerEncoder` and `@registerDecoder`, as above. A shared class shares the
   namespace of the one-modality ones, so its name must be unused.
4. **Name them under `encoders` and `decoders`**, not under `modalities`, which pairs one encoder and
   one decoder under one name. The routing, the latent spaces and the fusion are written as for any
   other model (see [configure latent spaces](configure-latent-spaces.md)).

```yaml
latent_mode: single
encoders:
  backbone:
    name: my_shared_encoder_v1
    kwargs:
      input_dims: {signal: 128, image: 1024}
decoders:
  backbone_out:
    name: my_shared_decoder_v1
    kwargs:
      output_shapes: {signal: [128], image: [32, 32]}
single_latent: {dim: 32}
```

`backbone` and `backbone_out` are instance names. The batch keys are the modalities the classes
declare (`signal` and `image`), which no instance name has to match.

### What the model asks of the two classes

| Requirement | Meaning |
|---|---|
| **Any non-empty subset** | Modality dropout and the cross-modal reports hand a shared encoder only the modalities that are present, with the others omitted from the dict. It must encode any non-empty subset of `input_modalities`. How (a mask, a mean over the tokens present, a learned placeholder) is its choice, since with one encoder there is nothing to fuse. |
| **Tokenisation lives in your class** | The model never concatenates or reshapes modalities. A shared encoder turns each tensor into tokens, and a shared decoder splits its output back into one tensor per modality. |
| **`latent_dim`** | The constructor takes it, as for any encoder or decoder, and the config layer fills it in from the latent space. Leave it out of `kwargs`. |
| **Exact keys** | A decoder that returns other keys than `output_modalities` is refused at the first forward pass. A modality that two encoders ingest, or two decoders reconstruct, is refused when the model is built. |
| **The batch key is the modality name** | The modality `signal` is read from `batch["signal"]` and reconstructed against `batch["signal"]`, whether a shared module or a one-modality one handles it. |

### Examples to start from

- **A real one:** `encoders/shared_transformer_encoder.py` and `decoders/shared_transformer_decoder.py`
  (ADR 0037). Each modality becomes patch tokens with its own embedding and a learned modality
  vector, the tokens of the modalities present form one sequence for one transformer, and a missing
  modality is simply a shorter sequence.
- **A much smaller one:** `tests/integration/_shared_modules.py` has a linear token per modality,
  averaged over the ones present, and one linear layer per output. It is the starting point for
  something simpler. The four `E1-*` rows of the configuration matrix are built from it in
  `tests/integration/test_configuration_matrix.py`.

See the [project specification, §6](../global-vae-project-specification.md) for the full modality
roadmap (1D signals and images are Phase 1; anything else is intentionally open-ended).
