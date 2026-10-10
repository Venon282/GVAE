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

# Still open

A second experiment file for spec §6.1 milestone 2 (paired signal+image, exercising
Fusion) is not written yet. The image encoder and decoder it needs exist
(`docs/adr/0017-2d-cnn-encoder-decoder.md`, `docs/adr/0018-2d-residual-encoder-decoder.md`),
and `configs/model/default.yaml` is a buildable signal plus image model (PoE fusion, both
modalities reconstructed). What is missing is a paired `configs/data/` file and the pairing
mechanism itself (spec §11, roadmap P3-2): pairing stays the caller's `loader_factory`. The
experiment file is roadmap P3-3, which proposes the name `signal_image_vae.yaml`.

A model whose decoder target is not an encoder input (signal and image to image, as in
`examples/03_signal_image_to_image.py`) cannot be written as a `configs/model/` file yet:
`buildModelFromConfig` supports `latent_mode: single` only, which ties one decoder to each
encoder name (roadmap P1-3).
