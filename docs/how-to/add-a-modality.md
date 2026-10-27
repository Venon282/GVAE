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
