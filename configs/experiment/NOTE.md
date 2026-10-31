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

`tabular_vae.yaml` (roadmap P3-8, ADR 0039) is the single-modality VAE of a table, `tabular -> z -> tabular`,
composing `configs/model/tabular_single_latent.yaml`, the data contract `configs/data/tabular.yaml` and
`configs/training/default.yaml`. It trains through `scripts/train.py`
(`tests/integration/test_tabular_model_configs.py`) and sets `training.reconstruction_reduction: sum` and
`reconstruction_weight: 0.5` like the image and transformer ones, with the numbers in the comment of the file.
`num_features` has to be set in the encoder and the decoder to the width of your rows.

`audio_vae.yaml` (roadmap P3-8, ADR 0040) is the single-modality VAE of a raw waveform, `audio -> z -> audio`,
composing `configs/model/audio_waveform_single_latent.yaml`, the data contract `configs/data/audio.yaml` and
`configs/training/default.yaml`. It trains through `scripts/train.py` (`tests/integration/test_audio_model_configs.py`)
and sets `training.reconstruction_reduction: sum` and `reconstruction_weight: 0.5` like the image, transformer and
tabular ones, with the numbers in the comment of the file. The clip length is the `data.sequence_length.audio` of the
data file and the `output_length` of the decoder, which have to agree.

`text_vae.yaml` (roadmap P3-8, ADR 0041) is the single-modality VAE of a text, `text -> z -> text`, composing
`configs/model/text_transformer_single_latent.yaml`, the data contract `configs/data/text.yaml` and
`configs/training/default.yaml`. It trains through `scripts/train.py` (`tests/integration/test_text_model_configs.py`)
and sets `training.reconstruction_loss: bce` with `reconstruction_reduction: sum` and `reconstruction_weight: 2.0`,
not the `mse` of the others: the decoder returns probabilities over the vocabulary, and the numbers in the comment of
the file say why `bce` and not `mse`. The length is the `data.sequence_length.text` of the data file and the
`output_length` of the decoder, and the vocabulary size is the `vocab_size` of the encoder and of the decoder, which
all have to agree with the loader.

`time_series_vae.yaml` (roadmap P3-7 and P3-8, ADR 0042) is the single-modality VAE of a window of one or several channels,
`time_series -> z -> time_series`, composing `configs/model/time_series_single_latent.yaml`, the data contract
`configs/data/time_series.yaml` and `configs/training/default.yaml`. It trains through `scripts/train.py`
(`tests/integration/test_time_series_model_configs.py`) and sets `training.reconstruction_loss: mse` with
`reconstruction_reduction: sum` and `reconstruction_weight: 50.0`, for complete series: the `mse` of a YAML file is `NaN` as
soon as one value of a batch is a `NaN`, so series with gaps or padding train through Python with
`computeMaskedSquaredError` (`docs/how-to/train-on-time-series.md`). The weight is the lever (a weight `w` is a `beta` of
`1 / w`) and scales as `0.5 / sigma^2` for a noise `sigma` after standardization, with the numbers in the comment of the
file. The length is the `data.sequence_length.time_series` of the data file and the `output_length` of the decoder (whose
`seed_length`, `kernel_sizes` and `paddings` have to land on it exactly), and the number of channels is the encoder's
`in_channels` and the decoder's `out_channels`, which all have to agree with the loader.

`graph_vae.yaml` (roadmap P3-8, ADR 0043) is the single-modality VAE of a graph as a dense adjacency matrix,
`graph -> z -> graph`, composing `configs/model/graph_cnn_single_latent.yaml`, the data contract `configs/data/graph.yaml`
and `configs/training/default.yaml`. It trains through `scripts/train.py` (`tests/integration/test_graph_model_configs.py`)
and sets `training.reconstruction_loss: bce` with `reconstruction_reduction: sum` and `reconstruction_weight: 4.0`: the
decoder (`graph_cnn_decoder_v1`) returns the probability of each edge, so `bce` is the Bernoulli likelihood of the graph,
and the weight is the lever (a weight `w` is a `beta` of `1 / w`), with the numbers in the comment of the file. The padded
number of nodes is the `data.sequence_length.graph` of the data file and the `output_shape` of the decoder (whose
`seed_shape`, `kernel_sizes` and `paddings` have to land on it exactly), which have to agree with the loader. The nodes
have to come in a meaningful order: the model compares matrices position by position, and ordering them is the loader's
(`docs/how-to/train-on-graphs.md`).

`point_cloud_vae.yaml` (roadmap P3-8, ADR 0044) is the single-modality VAE of a point cloud, `point_cloud -> z -> point_cloud`,
composing `configs/model/point_cloud_single_latent.yaml`, the data contract `configs/data/point_cloud.yaml` and
`configs/training/default.yaml`. **Its `training` block is a placeholder.** A YAML file can name only `mse`, `l1`, `bce` and
`smooth_l1`, each of which pairs point `i` of a reconstruction with point `i` of the target, which for a set is an unrelated
point, and on synthetic clouds the model `scripts/train.py` trains that way ended farther from the clouds than one that ignores
its input (`tests/integration/test_point_cloud_model_configs.py` pins it, and pins `NaN` on a padded cloud). The loss that works
is the Chamfer distance of `global_vae.utils.point_cloud_metrics`, a Python function: build the trainer from this file (the
optimizer, the beta schedule and the weight `100.0`, a `beta` of `1 / w`) and set `trainer.reconstruction_loss_fn` to it and
`trainer.reconstruction_reduction` to `"mean"`, since it is already a sum over points. The number of points is the
`data.sequence_length.point_cloud` of the data file and the `num_points` of the decoder, which have to be equal
(`docs/how-to/train-on-point-clouds.md`).

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
