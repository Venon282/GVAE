# Status

`configs/experiment/signal_vae.yaml` (spec §6.1 milestone 1: single-modality signal VAE)
now exists and is runnable end to end via `scripts/train.py`, composing
`configs/model/signal_single_latent.yaml` + `configs/data/signal.yaml` +
`configs/training/default.yaml`. See `docs/adr/0011-hydra-config-layer.md`.

# Still open

A second experiment file for spec §6.1 milestone 2 (paired signal+image, exercising
Fusion) depends on an image encoder/decoder existing first (see README.md's "What's
deliberately not built yet") and on the still-open pairing mechanism (spec §11).
`configs/model/default.yaml` is schema-valid for that future case already, but not yet
buildable.
