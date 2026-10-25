# 0037: Transformer and ViT backbones

**Status:** accepted (roadmap item P3-5; it adds backbones through the registries and changes no
core module)
**Date:** 2026-10-24

## Context

Spec §6 names a small transformer over the series and a Vision Transformer for images as the next
backbones after the 1D and 2D CNNs and their residual variants
([ADR 0017](0017-2d-cnn-encoder-decoder.md), [ADR 0018](0018-2d-residual-encoder-decoder.md)). The roadmap makes P3-5 a test of the registry design:
each backbone follows the new-modality checklist with zero core changes, and a core edit that looks
necessary is flagged and not made.

Three other items meet here:

- **Tokenisation.** [ADR 0027](0027-shared-encoders-and-decoders.md) left the stream of a shared
  encoder to "the concrete shared module" and said P3-5 writes the first one. A series and an image
  have to become tokens of one width before one transformer can read both.
- **`cross_attention` fusion.** It has been tested on tiny encoders only. The roadmap expects the
  transformer backbones to give it a meaningful scale.
- **The loss scale.** [ADR 0029](0029-loss-scale-convention.md) found that the default `mean`
  reduction collapses the VAEs on these data, and that `sum` with weight 0.5 is the ELBO of a
  unit-variance Gaussian decoder. A new backbone is the place to check the same on a different
  architecture.

## Decision

**1. Six registered classes and one module of building blocks.** Nothing else changes:

| Registry name | Class | Reads or writes |
|---|---|---|
| `1d_transformer_encoder_v1` | `OneDTransformerEncoder` | a series, `(batch, length)` or `(batch, channels, length)` |
| `1d_transformer_decoder_v1` | `OneDTransformerDecoder` | a series of fixed `output_length` |
| `2d_vit_encoder_v1` | `TwoDVitEncoder` | an image, `(batch, height, width)` or `(batch, channels, height, width)` |
| `2d_vit_decoder_v1` | `TwoDVitDecoder` | an image of fixed `output_shape` |
| `shared_transformer_encoder_v1` | `SharedTransformerEncoder` | any non-empty subset of a series and an image, as a dict |
| `shared_transformer_decoder_v1` | `SharedTransformerDecoder` | one tensor per modality, as a dict |

`utils/transformer_blocks.py` holds what they share: patch embedding and un-embedding, the two
kinds of positions, the transformer stack and the pooling head. The classes are registered by the
imports of `encoders/__init__.py` and `decoders/__init__.py`, and `models/`, `latent/`, `losses/`,
`training/`, `evaluation/` and `config/` are untouched.

**2. Patches are cut by a strided convolution, and any size is allowed.** A series is cut into
patches of `patch_size` values and an image into `patch_size` by `patch_size` pixels (a single
integer or a `[height, width]` pair), each projected to `d_model` values by a `Conv1d` or `Conv2d`
whose kernel and stride are the patch size. A size that is not a multiple of the patch is padded at
the end by repeating the edge value (an edge, not an invented zero), and a decoder crops its output
to the requested length or shape, so the number of tokens is `ceil(size / patch_size)` on each
axis. This is what lets a 100-point series go through a model whose decoder reconstructs 256
points, as with the CNNs.

**3. Positions are sinusoidal by default, learned on request.** Sinusoidal positions have no
parameters and are computed for whatever number of tokens arrives, so an encoder built with them
accepts any series length and any image size, like the adaptive pooling of the CNN encoders.
`positional_encoding: learned` holds one trained vector per position up to `max_length` (series) or
`image_size` (image) and refuses a longer input with an error naming that setting. An image uses a
2D form of the sinusoid, half of `d_model` for the row and half for the column, so `d_model` must
be a multiple of 4 whenever an image is involved; the error says so.

**4. An encoder is a pre-norm transformer with a pooled output.** A stack of `nn.TransformerEncoder`
layers (`norm_first=True`, GELU or ReLU, a final LayerNorm), a `[CLS]` token prepended (default) or
the mean of the tokens (`pooling: mean`), and a linear head, optionally with hidden layers
(`head_hidden_dims`), that returns `(mu, logvar)`. Attention is quadratic in the number of tokens,
so a long series or a large image wants a larger `patch_size`, not more layers.

**5. A decoder gives every token the latent vector and its position.** The tokens are the position
vectors plus one linear projection of `z`, the same for every token, so they differ only by
position and the layers of self-attention turn one vector into a series or an image. Each output
token is projected to the values of its patch, the patches are laid side by side and the last row
or column is cropped.

**6. The shared pair is one stream in and one stream out.** `SharedTransformerEncoder` takes
`modalities: {name: {kind: series | image, patch_size, in_channels}}`, tokenizes each present
modality with its own patch embedding, adds a learned embedding per modality to its tokens, joins
all of them (and a `[CLS]` token) into one sequence and runs one transformer over it. **A missing
modality is a shorter sequence**, so any non-empty subset is encoded with no mask and no
placeholder, which is the omission rule of ADR 0027. `SharedTransformerDecoder` takes
`modalities: {name: {kind, output_length | output_shape, patch_size, out_channels}}`, builds the
tokens of every modality (positions, modality embedding, projected `z`), runs one transformer over
the joined sequence and splits the output back into one tensor per modality. The attention of the
shared encoder is across the patches of both modalities, which is the cross-modal interaction a
fusion strategy cannot give (point 7).

**7. `cross_attention` keeps its interface, and so its scale.** A fusion receives `(mu, logvar)`
per encoder, so what `cross_attention` attends over is one token per modality, two tokens for a
signal and an image, whatever the backbone. A transformer backbone changes the encoders and leaves
the fusion's parameters as they were (a test builds the model with 3 and with 6 encoder layers and
compares them). `configs/model/signal_image_transformer_cross_attention.yaml` ships the pairing,
but the meaningful scale the roadmap hoped for does not come from the backbones. Attention over
patch tokens across modalities exists in the shared transformer of point 6. Giving a fusion strategy
the patch tokens would change `AbstractFusion` and `AbstractEncoder` (an encoder would have to
return tokens as well as `(mu, logvar)`), which is a core change and is **not made here**.

**8. The experiments set the loss scale, with the numbers behind it.** The four experiment files
carry `reconstruction_reduction: sum` and `reconstruction_weight: 0.5` ([ADR 0029](0029-loss-scale-convention.md), [ADR 0030](0030-per-modality-reconstruction-loss-and-weight.md)), as the
image experiments of P3-3 do. On the synthetic pairs of `examples/04_paired_loader.py` (about 800
training samples, 40 to 60 epochs, one seed), `mean` left every one of the new models at 0 active
latent units and an R² against the batch mean near 0, and `sum` at 0.5 trained them:

| Experiment | Signal R² | Image R² | Active latent units |
|---|---|---|---|
| `signal_transformer_vae` | 0.99 (the 1D CNN: 0.98) | | not recorded |
| `image_vit_vae` | | 0.92 (the 2D CNN: 0.65) | 4 of 16 (the CNN: 13 of 16) |
| `signal_image_shared_transformer_vae` | 0.99 | 0.45 | 9 of 32 |
| `signal_image_transformer_vae` | 0.98 | 0.24 | 1 of 32 |

These come from synthetic data and are not a tuned value. The choice of beta stays roadmap decision
D-6, which waits for a real run, and `signal_vae.yaml`, `image_vae.yaml` and the other
experiments that predate this ADR are not changed.

## Consequences

- **The registry design held.** The six classes, the module and the configs are all that was added,
  and every existing test passes without change. No core edit was necessary for the backbones.
- **Finding, not fixed: the loss settings of a shared decoder.** The config layer
  (`resolveReconstructionWeight`, `resolveReconstructionReduction`, `resolveReconstructionLoss`)
  validates a per-decoder mapping against the *decoder instance names* (`backbone_out`), and
  `computeTotalReconstructionLoss` looks the keys up by *reconstruction name*, which for a shared
  decoder are the modalities (`signal`, `image`). Results: a mapping by modality is refused; a
  weight or reduction mapping by decoder name is accepted and **silently not applied**; and a loss
  function mapping by decoder name raises `KeyError` at the first step. A single value works and
  covers every modality, which is what the shipped experiment uses. This predates P3-5 (ADR 0027
  flattens the reconstructions by modality, and the config checks were written for one-tensor
  decoders) and the fix is a core and config change, so it is flagged in the roadmap.
  `tests/integration/test_transformer_model_configs.py` states the behavior a mapping should have
  in two strict `xfail` tests, which start to fail when the mismatch is fixed and then ask for the
  marker to be removed.
- **A shared encoder or decoder takes one setting per class.** Heads, width and depth are shared
  by every modality in the stream. A model that wants a deeper transformer for the image than for
  the series uses the per-modality pairs of point 1 and a fusion.
- **Learned positions fix the size.** An encoder with `positional_encoding: learned` refuses a
  longer series or a larger image, and a decoder's tokens are fixed by its output size either way.
- **Sizes of the shipped models** (a 256-point series and a 64 by 64 image): the cross-attention
  model has 653,104 parameters and the shared one 417,168, with `d_model` 64 and a latent width of 32.
- **Nothing here was run on real data.** The R² values are on synthetic pairs, one seed and a
  short run; they show that the models train, not which backbone is better.
