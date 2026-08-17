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


def buildDummyDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build tiny, deterministic in-memory dataloaders, ignoring `config.train_path`
    (this fixture never touches disk: the whole point is exercising the
    `loader_factory` wiring, not any real data pipeline).
    """
    torch.manual_seed(config.seed)
    length = config.sequence_length or 256

    def make(num_batches: int) -> list[dict[str, torch.Tensor]]:
        return [{"signal": torch.randn(BATCH_SIZE, length)} for _ in range(num_batches)]

    return DataloaderBundle(train=make(3), val=make(1), test=make(1))
