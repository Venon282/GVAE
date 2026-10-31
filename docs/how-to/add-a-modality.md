# Add a new modality

Adding a modality should never require touching core framework code
(`models/global_vae.py`, `latent/`, `losses/`). If it does, that's a sign the
registry pattern is being bypassed somewhere. Flag it rather than
special-casing the new modality in.

1. **Subclass [`AbstractEncoder`](../reference/global_vae/encoders/base.md)**
   in `encoders/`, mapping your modality's raw input to a `(mu, logvar)`
   pair. See `encoders/one_d_cnn_encoder.py` for a worked example
   (length-agnostic, via adaptive pooling).
2. **Subclass [`AbstractDecoder`](../reference/global_vae/decoders/base.md)**
   in `decoders/`, mapping a latent vector back to a reconstruction of your
   modality.
3. **Register both** with `@registerEncoder("your_name")` /
   `@registerDecoder("your_name")`, and add the import to
   `encoders/__init__.py` / `decoders/__init__.py` so the decorator actually
   runs. A registered class only exists once its module has been imported.
4. **Add a config entry** referencing the two registry names. See
   `configs/model/signal_single_latent.yaml` for the shape.
5. **Add a test**: a unit test for your encoder/decoder's forward-pass
   shapes and gradient flow, plus an entry in the relevant integration test
   if your modality changes which of the 8 architecture configurations
   (spec §2.1) are exercised end to end.

## A worked example that is not a grid: tabular data

The signal and the image were written with the core, so they did not test the steps above. The
tabular modality ([ADR 0039](../adr/0039-tabular-modality.md)) is the first that followed them
afterwards, with no change to the core: `encoders/tabular_mlp_encoder.py` and
`decoders/tabular_mlp_decoder.py` (steps 1 to 3, plus their two lines in the `__init__.py` files),
`configs/model/tabular_single_latent.yaml` with a data file and an experiment (step 4), and
`tests/unit/test_tabular_backbone.py` and `tests/integration/test_tabular_model_configs.py` (step 5).
That integration test also checks that no core package names the new classes or the `tabular` batch
key, which is a useful test to copy for your own modality. [Train on a table](train-on-a-table.md)
shows the part that is not in the two classes: the `loader_factory` that builds the rows.

## A worked example where an existing family nearly fits: audio

A waveform is a 1D series, so the first thing to check for a new modality is whether a class that is
already registered does the job. The audio modality ([ADR 0040](../adr/0040-audio-modality.md)) checked,
and found three reasons the 1D classes were not enough (a decoder that must reach `output_length` exactly,
no bounded output, one dilation per stage), so it added `encoders/audio_waveform_encoder.py` and
`decoders/audio_waveform_decoder.py` and left the 1D ones as they were. It is also the example of a
modality whose most common alternative needs no class at all: a spectrogram is an image, and
`tests/integration/test_audio_model_configs.py` runs it through the 2D classes. [Train on
audio](train-on-audio.md) shows the `loader_factory` that cuts clips to one length and keeps them within
`[-1, 1]`.

## A worked example where the batch layout is the hard part: text

Text is a sequence of categorical tokens, and the first thing to settle for it is not the network but the
shape of the tensors. A decoder's reconstruction and its target (the batch tensor of the same name) must have
one shape, so a decoder that gives a distribution over the vocabulary at every position needs a one-hot
target, `(batch, length, vocab_size)`, and the encoder reads that same tensor. The text modality
([ADR 0041](../adr/0041-text-modality.md)) adds `encoders/text_transformer_encoder.py` and
`decoders/text_transformer_decoder.py`, built from the pieces of the transformer backbones
(`utils/transformer_blocks.py`) and a few of its own (`utils/text_blocks.py`: the token embedding, which
takes ids or one-hot rows, and the padding mask). It is the example of a modality whose loss the YAML cannot
name exactly (a cross-entropy is not in the loss table), and of how far a registered class and a Python loss
function get without touching the core. [Train on text](train-on-text.md) shows the `loader_factory` that
tokenizes, builds the vocabulary, pads and one-hot encodes.

## A worked example where the decoder already exists: time series

A multichannel series of any length is already covered by the registered 1D classes, so the useful question for the
time series modality ([ADR 0042](../adr/0042-time-series-modality.md)) was what they leave out: values that were not
observed. The answer is one encoder, `encoders/time_series_encoder.py`, and one utility module, `utils/series_blocks.py`,
with the decoder left as `1d_cnn_decoder_v1`: the checklist asks for a decoder, and it was met by an existing one because it
measured clearly better than the alternative. Where a mask travels is the part to copy. The loss sees only a
reconstruction and the batch tensor, so a `NaN` in that tensor is the mask for the encoder and, through
`computeMaskedSquaredError`, for the loss, with no second tensor and no change to the core. It is also the example of a
modality whose YAML experiment is only half of the story (the `mse` of a YAML file is `NaN` on a gap, so series with gaps train
through a Python `Trainer`), and of a measurement that went against the first guess: the new encoder matched a zero-filled
one on what was observed and led only on the padded tail. [Train on time series](train-on-time-series.md) shows the
`loader_factory` that cuts windows and standardizes per channel, and the Python route.

## A worked example where the family already fits: graphs

A graph of at most `N` nodes is a dense adjacency matrix `(batch, N, N)`, which is an image to the 2D convolutions,
so the first thing to do for the graph modality ([ADR 0043](../adr/0043-graph-modality.md)) was to measure what the
registered 2D classes already do, and to try graph-specific encoders and decoders against them. None beat the
registered pair, so the encoder is `2d_cnn_encoder_v1` as it is, and the one class added is
`decoders/graph_cnn_decoder.py`, a thin subclass of `TwoDCnnDecoder` that symmetrizes, applies a sigmoid and zeroes the
diagonal. That class is justified by the loss and not by the accuracy: the raw decoder cannot use the `bce` of a YAML
file, which measured better than its `mse`. It is the example to copy when an existing family nearly fits: measure the
existing classes first, add only what the measurements ask for, and say plainly where the result departs from the
checklist (here, no new encoder). It is also the example of a property no class can supply: the model compares matrices
position by position, so the order of the nodes matters as much as the whole choice of architecture, and that order is
the loader's job. [Train on graphs](train-on-graphs.md) shows the `loader_factory` that orders and pads the nodes, and
how to score the edges.

## A worked example where the loss is the hard part: point clouds

A point cloud is a set of `N` points, so `(batch, N, 3)` fits the one-tensor-per-modality contract and the comparison does
not: every loss a YAML file can name pairs row `i` of the reconstruction with row `i` of the target, and for a set those are
unrelated points ([ADR 0044](../adr/0044-point-cloud-modality.md)). The first thing to do was to measure that, and the model
trained with the loss of a YAML file ended farther from the clouds than one that ignores its input. So the modality has two
registered classes, `encoders/point_cloud_encoder.py` (a pooling over the points that leaves out the rows of `NaN`, exactly
invariant to their order) and `decoders/point_cloud_decoder.py` (a learned template folded by the latent vector), built from
`utils/mlp_blocks.py` and `utils/transformer_blocks.py`, and one function that is not a class,
`utils/point_cloud_metrics.computeChamferDistance`. The trainer takes any Python function as its reconstruction loss, so the
core did not change, and the file of the experiment only composes the model, the data and the optimizer, with a header that
says its own loss is a placeholder. It is the example to copy when the hard part is the loss and not the network: measure
what the loss of a YAML file does before building anything, put the loss in `utils/` as a function with its tests, flag the
missing table entry instead of editing the core, and compare the new classes with the registered ones honestly, since here
the registered 1D pair came within a few percent and the new classes are justified by the invariance, the `NaN` convention and
the cost. [Train on point clouds](train-on-point-clouds.md) shows the `loader_factory` and the Python route.

## One encoder or decoder for several modalities

The steps above give each modality its own encoder and decoder (`EN-*` and `*-DN` in spec §2.1).
To have one encoder ingest several modalities, or one decoder reconstruct several, subclass the
shared interfaces instead ([ADR 0027](../adr/0027-shared-encoders-and-decoders.md)). Everything
else stays: the same registries, the same decorators, the same `__init__.py` imports.

1. **Subclass [`AbstractSharedEncoder`](../reference/global_vae/encoders/shared_base.md)**.
   Declare `input_modalities` (the batch keys it reads) and `latent_dim`. `forward` receives a
   `dict[str, Tensor]` and returns `(mu, logvar)`.
2. **Subclass [`AbstractSharedDecoder`](../reference/global_vae/decoders/shared_base.md)**.
   Declare `output_modalities`. `forward` receives one latent tensor and returns a
   `dict[str, Tensor]` with exactly those keys.
3. **Register** with `@registerEncoder` and `@registerDecoder`, as above. A shared class shares
   the namespace of the one-modality ones, so its name must be unused.
4. **Name them under `encoders` and `decoders`**, not under `modalities`, which pairs one encoder
   and one decoder under one name. The routing, the latent spaces and the fusion are written as
   for any other model (see [configure latent spaces](configure-latent-spaces.md)).

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

What the model asks of the two classes:

- **Any non-empty subset.** Modality dropout and the cross-modal reports hand a shared encoder only
  the modalities that are present, with the others omitted from the dict. It must encode any
  non-empty subset of `input_modalities`. How it does so (a mask, a mean over the tokens present,
  a learned placeholder) is its choice, since with one encoder there is nothing to fuse.
- **Tokenisation lives in your class.** The model never concatenates or reshapes modalities. A
  shared encoder turns each tensor into tokens, and a shared decoder splits its output back into
  one tensor per modality.
- **`latent_dim`.** The constructor takes it, as for any encoder or decoder, and the config layer
  fills it in from the latent space (leave it out of `kwargs`).
- **Exact keys.** A decoder that returns other keys than `output_modalities` is refused at the
  first forward pass, and a modality that two encoders ingest, or two decoders reconstruct, is
  refused when the model is built.
- **The batch key is the modality name.** The modality `signal` is read from `batch["signal"]` and
  reconstructed against `batch["signal"]`, whether a shared module or a one-modality one handles it.

`encoders/shared_transformer_encoder.py` and `decoders/shared_transformer_decoder.py` are a real
one (ADR 0037): each modality becomes patch tokens with its own embedding and a learned modality
vector, the tokens of the modalities present form one sequence for one transformer, and a missing
modality is simply a shorter sequence. `tests/integration/_shared_modules.py` has a pair of much
smaller modules (a linear token per modality averaged over the ones present, and one linear layer
per output) that are a starting point for something simpler. The four `E1-*` rows of the
configuration matrix are built from them in `tests/integration/test_configuration_matrix.py`.

See the [project specification, §6](../global-vae-project-specification.md)
for the full modality roadmap (1D signals and images are Phase 1; anything
else is intentionally open-ended).
