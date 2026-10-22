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

# Still open

The paired data itself: `configs/data/signal_image.yaml` only states the contract (a
`loader_factory` of your own that yields `signal`, `image` and, for the translation-style models,
`image_out`), and pairing stays the caller's `loader_factory` (spec §6.2).
`examples/04_paired_loader.py` is a reference one, run through `scripts/train.py` on
`experiment/signal_image_vae` (roadmap P3-2); the real dataset is what is still open. The
image-only model configs and the review of `configs/model/default.yaml` are roadmap P3-3.

A model whose decoder target is not an encoder input (signal and image to image, as in
`examples/03_signal_image_to_image.py`) is `configs/model/signal_image_to_image.yaml`: the
`encoders` and `decoders` sections name instances independently of modalities. It has no
experiment file, since it needs a data contract of its own (three batch keys, 64 point signals
and 32 x 32 images).
