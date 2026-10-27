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

from global_vae.config.data import DataConfig, DataloaderBundle

BATCH_SIZE = 4
IMAGE_SIZE = 64
TABULAR_FEATURES = 16


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
