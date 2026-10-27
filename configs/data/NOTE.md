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

`configs/data/signal_image.yaml` is the same kind of contract for the models that take a signal
and an image (`configs/experiment/signal_image_*.yaml`, `docs/adr/0025-config-schema-for-several-latent-spaces.md`):
a `loader_factory` of your own, per-modality `transforms` for `signal` and `image`, and
`sequence_length.signal`. It describes the batches the experiments expect (`signal`, `image`, and
`image_out` for the translation-style models); pairing a signal with its image stays the
`loader_factory`'s job. `examples/04_paired_loader.py` is a reference for it (roadmap P3-2): it
pairs two folders of files by sample ID, leaves out the samples that have one modality only,
splits by sample ID with `seed`, applies `transforms` and checks `sequence_length`. It is an
example, not framework code.

`configs/data/image.yaml` is the contract of the single-modality image models
(`configs/experiment/image_vae.yaml` and `image_resnet_vae.yaml`, roadmap P3-3): a
`loader_factory` of your own whose batches carry one key, `image`, of shape `(batch, 64, 64)`, and
a per-modality `transforms` entry for `image` (an identity `standardize` to fill in). It has no
`sequence_length`, which is the length of a 1D signal: an image's size is the decoder's
`output_shape`. A loader that yields more keys, such as the paired one of
`examples/04_paired_loader.py`, serves these experiments as well, since a key with no encoder or
decoder is ignored.

`configs/data/tabular.yaml` is the contract of the tabular model (`configs/experiment/tabular_vae.yaml`, roadmap
P3-8): a `loader_factory` of your own whose batches carry one key, `tabular`, of shape
`(batch, num_features)` (numeric columns standardized, then one-hot blocks, no missing value), and a per-modality
`transforms` entry for `tabular` (an identity `standardize` to replace). Per-column statistics are tensors, which
YAML cannot write, so the loader builds the transform in code (`docs/how-to/train-on-a-table.md`). It has no
`sequence_length`: the width of a row is `num_features` in the model file.

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
