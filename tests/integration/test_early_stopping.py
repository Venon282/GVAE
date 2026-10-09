"""Integration tests for `training.callbacks.early_stopping.EarlyStopping`.

See spec §10 and ADR 0022.

`PlateauTracker`'s own value-correctness is covered by `test_plateau_tracker.py`; this
file covers what `EarlyStopping` itself is responsible for: reading `self.monitor`
from the epoch metrics dict, raising a clear `KeyError` when it is absent, setting
`trainer.should_stop` exactly on the epochs `PlateauTracker` reports a plateau, and
that `Trainer.fit` actually honors `should_stop` (stops before `num_epochs` epochs
have run).

As in `test_reduce_lr_on_plateau.py`, the "full training run" tests never rely on the
*actual* trajectory of a real loss: a small `_ForceMetric` callback, placed before
`EarlyStopping` in `Trainer.callbacks`, overwrites one metric key with a fully
deterministic, hand-picked sequence before `EarlyStopping.onEpochEnd` ever reads it.
"""

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.early_stopping import EarlyStopping
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_early_stopping_test")
class _DummySignalEncoder(AbstractEncoder):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_mu(x), self.to_logvar(x)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return "signal"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_signal_decoder_early_stopping_test")
class _DummySignalDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


class _ForceMetric(TrainerCallback):
    """Overwrites one epoch-metrics key with a hand-picked, deterministic sequence."""

    def __init__(self, key: str, values: list[float]) -> None:
        self.key = key
        self._values = iter(values)

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        metrics[self.key] = next(self._values)


def _buildSingleModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_early_stopping_test",
                "decoder": "dummy_signal_decoder_early_stopping_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int = 2, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


def _buildTrainer(**kwargs) -> Trainer:
    return Trainer(_buildSingleModalityModel(), device="cpu", **kwargs)


class TestDirectCall:
    def test_missing_monitor_key_raises_key_error(self) -> None:
        trainer = _buildTrainer()
        callback = EarlyStopping(monitor="val/loss/total")
        with pytest.raises(KeyError, match="val/loss/total"):
            callback.onEpochEnd(trainer, 0, {"train/loss/total": 1.0})

    def test_sets_should_stop_once_patience_exceeded(self) -> None:
        trainer = _buildTrainer()
        callback = EarlyStopping(monitor="loss", mode="min", patience=1, threshold=0.0)
        for value in [10.0, 10.0]:  # improves, then one non-improving epoch (within patience)
            callback.onEpochEnd(trainer, 0, {"loss": value})
        assert trainer.should_stop is False

        callback.onEpochEnd(trainer, 1, {"loss": 10.0})  # second non-improving epoch
        assert trainer.should_stop is True

    def test_does_not_stop_while_improving(self) -> None:
        trainer = _buildTrainer()
        callback = EarlyStopping(monitor="loss", mode="min", patience=1)
        for value in [10.0, 9.0, 8.0, 7.0]:
            callback.onEpochEnd(trainer, 0, {"loss": value})
        assert trainer.should_stop is False

    def test_max_mode_tracks_the_opposite_direction(self) -> None:
        trainer = _buildTrainer()
        callback = EarlyStopping(monitor="metric", mode="max", patience=0, threshold=0.0)
        callback.onEpochEnd(trainer, 0, {"metric": 1.0})
        assert trainer.should_stop is False
        callback.onEpochEnd(trainer, 1, {"metric": 1.0})  # no improvement, patience=0
        assert trainer.should_stop is True


class TestFullTrainingRun:
    def test_stops_before_num_epochs_once_the_forced_metric_plateaus(self) -> None:
        trainer = _buildTrainer()
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 10.0, 10.0, 10.0, 10.0]),
            EarlyStopping(monitor="train/loss/total", patience=1, threshold=0.0),
        ]

        history = trainer.fit(_fixedDataset(), num_epochs=5)

        # Epoch 0 establishes the best value; epoch 1 is tolerated (patience=1);
        # epoch 2 exceeds patience and stops the run, so only 3 epochs ever run.
        assert len(history) == 3
        assert trainer.should_stop is True

    def test_runs_every_epoch_if_the_metric_keeps_improving(self) -> None:
        trainer = _buildTrainer()
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 9.0, 8.0, 7.0, 6.0]),
            EarlyStopping(monitor="train/loss/total", patience=1),
        ]

        history = trainer.fit(_fixedDataset(), num_epochs=5)

        assert len(history) == 5
        assert trainer.should_stop is False

    def test_should_stop_resets_at_the_start_of_a_fresh_fit_call(self) -> None:
        trainer = _buildTrainer()
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 10.0, 10.0]),
            EarlyStopping(monitor="train/loss/total", patience=0, threshold=0.0),
        ]
        trainer.fit(_fixedDataset(), num_epochs=3)
        assert trainer.should_stop is True

        # A second fit() call resets should_stop, so training can resume; this
        # second run has no _ForceMetric callback left (it was exhausted above,
        # and a fresh one is easiest), so the real (unconstrained) training loss
        # is used and the run completes its requested epochs normally.
        trainer.callbacks = []
        history_before = len(trainer.history)
        trainer.fit(_fixedDataset(), num_epochs=2)
        assert trainer.should_stop is False
        assert len(trainer.history) == history_before + 2

    def test_callback_listed_after_early_stopping_still_runs_on_the_stopping_epoch(
        self,
    ) -> None:
        """`should_stop` is only checked once every onEpochEnd callback has run."""
        calls: list[int] = []

        class _RecordsCalls(TrainerCallback):
            def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
                calls.append(epoch)

        trainer = _buildTrainer()
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 10.0]),
            EarlyStopping(monitor="train/loss/total", patience=0, threshold=0.0),
            _RecordsCalls(),
        ]

        trainer.fit(_fixedDataset(), num_epochs=5)

        # Epoch 0 establishes the best; epoch 1 plateaus and stops the run, but
        # _RecordsCalls (listed after EarlyStopping) still observed both epochs.
        assert calls == [0, 1]
