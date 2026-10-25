# Status

`configs/experiment/signal_vae.yaml` (spec §6.1 milestone 1: single-modality signal VAE)
now exists and is runnable end to end via `scripts/train.py`, composing
`configs/model/signal_single_latent.yaml` + `configs/data/signal.yaml` +
`configs/training/default.yaml`. See `docs/adr/0011-hydra-config-layer.md`.

`configs/experiment/signal_resnet_vae.yaml` is the same shape, composing
`configs/model/signal_resnet_single_latent.yaml` (the residual "ResNet-style" 1D
encoder/decoder, spec §7) instead. Selectable either as its own named experiment file
(`--config-name experiment/signal_resnet_vae`) or as a `model=
signal_resnet_single_latent` override directly on top of `signal_vae.yaml`; both are
runnable end to end via `scripts/train.py`. See
`docs/adr/0014-residual-1d-encoder-decoder.md`.

`configs/experiment/signal_image_vae.yaml`, `signal_image_independent_latents.yaml` and
`signal_image_shared_private.yaml` are one experiment for each pattern of spec §9 (single latent
space, several independent latent spaces, shared plus private), composing a model from
`configs/model/`, the paired data contract `configs/data/signal_image.yaml` and a training
config (`configs/training/default.yaml` for the single latent space,
`configs/training/several_latents.yaml` plus one `beta_schedules` entry per latent space for the
others). All three train through `scripts/train.py`, which `tests/integration/test_train_script.py`
checks on dummy paired batches. See `docs/adr/0025-config-schema-for-several-latent-spaces.md`.

`configs/experiment/image_vae.yaml` and `image_resnet_vae.yaml` (roadmap P3-3) are the image
counterparts of `signal_vae.yaml` and `signal_resnet_vae.yaml`: a single-modality image VAE,
`image -> z -> image`, composing `configs/model/image_single_latent.yaml` (the plain 2D pair,
ADR 0017) or `image_resnet_single_latent.yaml` (the residual pair, ADR 0018), the data contract
`configs/data/image.yaml` and `configs/training/default.yaml`. Both train through
`scripts/train.py` (`tests/integration/test_image_model_configs.py`), and `model=` swaps one for
the other. Unlike the signal ones they set `training.reconstruction_reduction: sum` and
`reconstruction_weight: 0.5`: under the default `mean` and `beta = 1` the plain image model
collapsed on synthetic images (the comment of each file has the numbers, and says they are not a
tuned value, roadmap decision D-6).

`signal_transformer_vae.yaml`, `image_vit_vae.yaml`, `signal_image_transformer_vae.yaml` and
`signal_image_shared_transformer_vae.yaml` (roadmap P3-5, ADR 0037) compose the transformer
models of `configs/model/`: a small transformer over the series, a ViT over the image, both fused
by `cross_attention`, and both through one shared transformer with no fusion (the
`encoders`/`decoders` form of the schema, ADR 0027). They read the data contracts of the CNN
experiments they mirror (`signal.yaml`, `image.yaml`, `signal_image.yaml`), train through
`scripts/train.py` (`tests/integration/test_transformer_model_configs.py`) and set the `sum`
loss with weight 0.5 like the image ones, with the numbers in the comment of each file. For the
shared decoder the loss settings stay single values (the comment of that file says why).

# Still open

The paired data itself: `configs/data/signal_image.yaml` only states the contract (a
`loader_factory` of your own that yields `signal`, `image` and, for the translation-style models,
`image_out`), and pairing stays the caller's `loader_factory` (spec §6.2).
`examples/04_paired_loader.py` is a reference one, run through `scripts/train.py` on
`experiment/signal_image_vae` (roadmap P3-2); the real dataset is what is still open. The
image-only model configs of roadmap P3-3 are the two experiments above. `signal_image_vae.yaml`
still trains on the default `mean` loss scale, which collapsed on the synthetic pairs (roadmap P3-3
status): the loss scale of a model with an image and a signal is a choice for decision D-6.

A model whose decoder target is not an encoder input (signal and image to image, as in
`examples/03_signal_image_to_image.py`) is `configs/model/signal_image_to_image.yaml`: the
`encoders` and `decoders` sections name instances independently of modalities. It has no
experiment file, since it needs a data contract of its own (three batch keys, 64 point signals
and 32 x 32 images).
