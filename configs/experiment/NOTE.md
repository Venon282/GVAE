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
Fusion) depends on an image encoder/decoder existing first (see README.md's "What's
deliberately not built yet") and on the still-open pairing mechanism (spec §11).
`configs/model/default.yaml` is schema-valid for that future case already, but not yet
buildable.
