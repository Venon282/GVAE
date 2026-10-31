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

`configs/data/audio.yaml` is the contract of the audio model (`configs/experiment/audio_vae.yaml`, roadmap P3-8): a
`loader_factory` of your own whose batches carry one key, `audio`, of shape `(batch, samples)` (or `(batch, channels,
samples)`), float, within `[-1, 1]`, at one sample rate, with `samples` equal to `sequence_length.audio`. Its
`transforms` entry for `audio` is empty on purpose (the decoder's `tanh` bounds the waveform, which a `standardize`
would leave), and decoding, resampling, normalizing the level and cutting to a length are the loader's
(`docs/how-to/train-on-audio.md`). The sample rate is not a field of `DataConfig`: it is the loader's choice.

`configs/data/text.yaml` is the contract of the text model (`configs/experiment/text_vae.yaml`, roadmap P3-8): a
`loader_factory` of your own whose batches carry one key, `text`, a one-hot float tensor `(batch, length,
vocab_size)` (a row of the pad class or of zeros is padding), with `length` equal to `sequence_length.text`. It is
one-hot because the loss compares the decoder's output with this very tensor, which must have its shape. Its
`transforms` entry for `text` is empty and has to stay so (every transform acts on numbers, and a one-hot row is
not a quantity). Tokenizing, the vocabulary, cutting or padding to a length and the one-hot encoding are the
loader's (`docs/how-to/train-on-text.md`). The vocabulary size is not a field of `DataConfig`: it is the last axis
of the tensor and the `vocab_size` of the encoder and the decoder.

`configs/data/time_series.yaml` is the contract of the time series model (`configs/experiment/time_series_vae.yaml`,
roadmap P3-7 and P3-8): a `loader_factory` of your own whose batches carry one key, `time_series`, a float tensor
`(batch, channels, length)` (or `(batch, length)` for one channel), with `length` equal to `sequence_length.time_series`.
A `NaN` is a value that was not observed: a gap, or the padding of a series shorter than `length`. The encoder reads the
mask from it, and the masked loss of the Python route counts the observed values only. Its `transforms` entry for
`time_series` is empty, because the standardization is per channel and a YAML file can write only one number for
`standardize`: the loader builds a `StandardizeTransform` from tensors of shape `(channels, 1)`, which leaves a `NaN` a
`NaN` (`docs/how-to/train-on-time-series.md`). Cutting windows, padding them to one length and putting the observations on a
regular grid are the loader's too. The number of channels is not a field of `DataConfig`: it is the second axis of the
tensor, the encoder's `in_channels` and the decoder's `out_channels`.

`configs/data/graph.yaml` is the contract of the graph model (`configs/experiment/graph_vae.yaml`, roadmap P3-8): a
`loader_factory` of your own whose batches carry one key, `graph`, a float tensor `(batch, N, N)` (or `(batch, C, N, N)`
for several kinds of edge): the 0/1 adjacency matrix of each graph, padded with isolated nodes (rows and columns of zeros,
not `NaN`) up to `N`, which is `sequence_length.graph`. The nodes have to come in a meaningful order, because the model
reconstructs the matrix position by position: ordering them is the loader's, and a number of nodes per graph can ride
in another key (`num_nodes`), which the trainer and `evaluate` ignore (ADR 0019). Its `transforms` entry for `graph` is empty, because an adjacency
matrix is not a quantity to standardize (`docs/how-to/train-on-graphs.md`). The padded size is not a field of `DataConfig`
beyond `sequence_length`: it is also the decoder's `output_shape`, which has to agree with it.

`configs/data/point_cloud.yaml` is the contract of the point cloud model (`configs/experiment/point_cloud_vae.yaml`, roadmap
P3-8): a `loader_factory` of your own whose batches carry one key, `point_cloud`, a float tensor `(batch, N, 3)` (or
`(batch, N, 3 + k)` for points with normals or colors): one row per point, in no particular order, with a row of `NaN` for a
point that does not exist (a cloud with fewer than `N` points is padded with them, not with zeros, since the origin is a
point). `N` is `sequence_length.point_cloud`, which is also the decoder's `num_points` and has to equal it, since the trainer
refuses a reconstruction of another shape. Every cloud is centered and scaled by the loader, so its `transforms` entry for
`point_cloud` is empty: the transforms of this framework use one mean and one scale for the whole dataset, and a cloud is
normalized one at a time. A number of points per cloud can ride in another key, which the trainer and `evaluate` ignore
(ADR 0019). Sampling, normalization and padding are the loader's (`docs/how-to/train-on-point-clouds.md`).

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
