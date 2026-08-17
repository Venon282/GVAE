# Status

`global_vae/config/data.py`'s `DataConfig` schema (spec §10 "Config management") and
`configs/data/signal.yaml` (an illustrative, schema-valid data config for the signal-VAE
milestone, spec §6.1 milestone 1) now exist. See `docs/adr/0011-hydra-config-layer.md`.

# Still deferred

No dataset/transform *code* exists in this framework, and none is planned to: per an
explicit scope decision, dataset loading, preprocessing, and train/val/test splitting
stay entirely the user's own responsibility (see `src/global_vae/data/NOTE.md`).
`configs/data/*.yaml` only ever describes that information (paths, batch size, split,
named transforms) and a `loader_factory` reference to the user's own
`"module.path:function_name"` callable; `global_vae.config.data.buildDataloadersFromConfig`
does nothing but resolve and call it. The first concrete dataset/task (spec §11's
still-open question) determines what a real `loader_factory` implementation looks like,
not anything in this repository.
