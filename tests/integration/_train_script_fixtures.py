"""Dummy `loader_factory` for `test_train_script.py`'s end-to-end subprocess test.

Named without a `test_`/`_test` pattern so pytest's own test discovery never imports
this file directly (see `_script_fixtures.py`'s own docstring for the identical
reasoning). Resolved by `scripts/train.py` (running as a real subprocess) via
`global_vae.utils.imports.importCallable`, given as
`"tests.integration._train_script_fixtures:buildDummyDataloaders"` on the command line,
so this module must be importable as a dotted path, which requires the repository
root (not just `tests/integration/`) on `PYTHONPATH` in the subprocess.
"""

import torch
import torch.nn.functional as functional

from global_vae.config.data import DataConfig, DataloaderBundle

BATCH_SIZE = 4
IMAGE_SIZE = 64
TABULAR_FEATURES = 16
AUDIO_SAMPLES = 2048
TEXT_LENGTH = 8
TEXT_VOCAB = 32
SERIES_LENGTH = 96
SERIES_CHANNELS = 3
GRAPH_NODES = 16
POINT_CLOUD_POINTS = 64


def buildDummyDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny, deterministic in-memory dataloaders, ignoring `config.train_path`.

    This fixture never touches disk: the whole point is exercising the `loader_factory` wiring,
    not any real data pipeline.
    """
    torch.manual_seed(config.seed)
    length = (config.sequence_length or {}).get("signal", 256)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [{"signal": torch.randn(BATCH_SIZE, length)} for _ in range(num_batches)]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummySignalImageDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the models that take a signal and an image.

    Serves every shipped signal plus image experiment (`configs/experiment/signal_image_*.yaml`)
    with one set of batches: `signal` of shape `(batch, length)`, `image` of shape `(batch, 64,
    64)`, and `image_out`, the target of the translation-style models whose decoder is named
    that (a key with no encoder is never encoded, ADR 0019; the models that have none leave it
    unused). Like `buildDummyDataloaders`, it never touches disk.

    Args:
        config: The data config; `seed` and `sequence_length["signal"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    length = (config.sequence_length or {}).get("signal", 256)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        batches = []
        for _ in range(num_batches):
            image = torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
            batches.append(
                {
                    "signal": torch.randn(BATCH_SIZE, length),
                    "image": image,
                    "image_out": image.clone(),
                }
            )
        return batches

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyImageDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the image-only models (`experiment/image_*.yaml`).

    The batches carry one key, `image`, of shape `(batch, 64, 64)`: the data contract of
    `configs/data/image.yaml`. Like the other builders of this module, it never touches disk.

    Args:
        config: The data config; `seed` is read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [
            {"image": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)} for _ in range(num_batches)
        ]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyTabularDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the tabular model (`experiment/tabular_vae.yaml`).

    The batches carry one key, `tabular`, of shape `(batch, 16)`: the data contract of
    `configs/data/tabular.yaml` and the `num_features` of
    `configs/model/tabular_single_latent.yaml`.
    Like the other builders of this module, it never touches disk.

    Args:
        config: The data config; `seed` is read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [{"tabular": torch.randn(BATCH_SIZE, TABULAR_FEATURES)} for _ in range(num_batches)]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyAudioDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the audio model (`experiment/audio_vae.yaml`).

    The batches carry one key, `audio`, of shape `(batch, 2048)` with values within `[-1, 1]`: the
    data contract of `configs/data/audio.yaml`, whose `sequence_length.audio` is the
    `output_length` of `configs/model/audio_waveform_single_latent.yaml`. Like the other builders
    of this module, it never touches disk.

    Args:
        config: The data config; `seed` and `sequence_length["audio"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    length = (config.sequence_length or {}).get("audio", AUDIO_SAMPLES)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [{"audio": torch.tanh(torch.randn(BATCH_SIZE, length))} for _ in range(num_batches)]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyTextDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the text model (`experiment/text_vae.yaml`).

    The batches carry one key, `text`, a one-hot tensor of shape `(batch, 8, 32)`: the data
    contract of `configs/data/text.yaml`, whose `sequence_length.text` is the `output_length` of
    `configs/model/text_transformer_single_latent.yaml` and whose `32` is its `vocab_size`. Each
    text has between three and eight real tokens (ids `1` to `31`) and the padding token (id `0`)
    after them. Like the other builders of this module, it never touches disk.

    Args:
        config: The data config; `seed` and `sequence_length["text"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    length = (config.sequence_length or {}).get("text", TEXT_LENGTH)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        batches = []
        for _ in range(num_batches):
            ids = torch.randint(1, TEXT_VOCAB, (BATCH_SIZE, length))
            real = torch.randint(min(3, length), length + 1, (BATCH_SIZE, 1))
            ids = torch.where(torch.arange(length) < real, ids, torch.zeros_like(ids))
            batches.append({"text": functional.one_hot(ids, TEXT_VOCAB).float()})
        return batches

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyTimeSeriesDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the time series model.

    It serves `experiment/time_series_vae.yaml`.

    The batches carry one key, `time_series`, of shape `(batch, 3, 96)`: the data contract of
    `configs/data/time_series.yaml`, whose `sequence_length.time_series` is the `output_length` of
    `configs/model/time_series_single_latent.yaml` and whose `3` is its `in_channels`. The series
    are complete (no `NaN`), because the loss of the experiment is a plain mean squared error, which
    a gap would turn into `NaN` (ADR 0042). Like the other builders of this module, it never touches
    disk.

    Args:
        config: The data config; `seed` and `sequence_length["time_series"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    length = (config.sequence_length or {}).get("time_series", SERIES_LENGTH)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [
            {"time_series": torch.randn(BATCH_SIZE, SERIES_CHANNELS, length)}
            for _ in range(num_batches)
        ]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyGraphDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the graph model (`experiment/graph_vae.yaml`).

    The batches carry one key, `graph`, of shape `(batch, 16, 16)`: the data contract of
    `configs/data/graph.yaml`, whose `sequence_length.graph` is the side of the `output_shape` of
    `configs/model/graph_cnn_single_latent.yaml`. Each matrix is a random undirected graph with a
    zero diagonal, of 8 to 16 nodes, padded with isolated nodes. Like the other builders of this
    module, it never touches disk.

    Args:
        config: The data config; `seed` and `sequence_length["graph"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    size = (config.sequence_length or {}).get("graph", GRAPH_NODES)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        batches = []
        for _ in range(num_batches):
            upper = torch.triu((torch.rand(BATCH_SIZE, size, size) < 0.3).float(), diagonal=1)
            nodes = torch.randint(min(8, size), size + 1, (BATCH_SIZE, 1))
            exists = torch.arange(size)[None, :] < nodes
            adjacency = upper + upper.transpose(1, 2)
            adjacency = adjacency * (exists[:, :, None] & exists[:, None, :]).float()
            batches.append({"graph": adjacency})
        return batches

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))


def buildDummyPointCloudDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny in-memory dataloaders for the point cloud models.

    Serves `configs/experiment/point_cloud_vae.yaml` with batches of one key, `point_cloud`, of
    shape `(batch, num_points, 3)`: random points in the unit ball, in no particular order. Every
    cloud is complete, with no row of `NaN`, because the position-by-position loss of a YAML file
    is `NaN` on one. The number of points is read from `sequence_length["point_cloud"]`. Like
    `buildDummyDataloaders`, it never touches disk.

    Args:
        config: The data config; `seed` and `sequence_length["point_cloud"]` are read.

    Returns:
        A `DataloaderBundle` of 3 train, 1 validation and 1 test batches.
    """
    torch.manual_seed(config.seed)
    points = (config.sequence_length or {}).get("point_cloud", POINT_CLOUD_POINTS)

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        batches = []
        for _ in range(num_batches):
            clouds = torch.randn(BATCH_SIZE, points, 3)
            clouds = clouds / clouds.norm(dim=-1, keepdim=True).clamp_min(1.0)
            batches.append({"point_cloud": clouds})
        return batches

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))
