# Status

`global_vae/config/data.py`'s `DataConfig` schema (spec §10 "Config management")
and `configs/data/signal.yaml` (an illustrative, schema-valid data config for the
signal-VAE milestone, spec §6.1 milestone 1) exist, as before. As of spec §6.2 /
`docs/adr/0012-generic-data-transforms.md`, `DataConfig.transforms` is no longer
purely decorative: it is a **per-modality** mapping, modality name -> ordered
list of `TransformConfig` entries (name + kwargs), each resolved through the
`data.transforms` registry (`log`, `standardize`, `resample`), and
`global_vae.config.data.buildTransformPipeline(config)` turns it into one real,
composed, invertible `ComposeTransform` per modality. `DataConfig.sequence_length`
is keyed the same way (modality name -> target length). See
`docs/adr/0015-per-modality-data-transforms.md` for why: a second 1D-signal-family
dataset (spec §6's own "only preprocessing differing" example) needs its own
steps, its own statistics, and its own resampled length, independently of any
other modality already configured. `configs/data/signal.yaml` ships a working
`log` + `standardize` example for its one `signal` modality (see that file's own
comments for how to fill in real statistics).

# Still deferred / permanent scope boundary

No dataset/pairing/splitting *code* exists in this framework, and none is planned:
per an explicit, permanent scope decision (spec §6.2), dataset loading,
preprocessing orchestration, and train/val/test splitting stay entirely the user's
own responsibility. `configs/data/*.yaml` only ever describes that information
(paths, batch size, split, the generic transform pipeline) and a `loader_factory`
reference to the user's own `"module.path:function_name"` callable;
`global_vae.config.data.buildDataloadersFromConfig` does nothing but resolve and
call it. The first concrete dataset/task (spec §11's still-open pairing-mechanism
question) determines what a real `loader_factory` implementation looks like, not
anything in this repository; that question is independent of, and does not block,
the transform pipeline above, which is already fully implemented.
