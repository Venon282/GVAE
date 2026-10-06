"""Integration tests for `training.callbacks.reduce_lr_on_plateau.ReduceLrOnPlateau`
(spec §10, ADR 0022).

`PlateauTracker`'s own value-correctness (mode/threshold/threshold_mode/cooldown) is
covered by `test_plateau_tracker.py`; this file covers what `ReduceLrOnPlateau` itself
is responsible for: reading `self.monitor` from the epoch metrics dict, raising a
clear `KeyError` when it is absent, and reducing `trainer.optimizer`'s learning rate
(respecting `min_lr` and `eps`) exactly on the epochs `PlateauTracker` reports a
plateau.

The "full training run" tests below drive a real `Trainer.fit()` call, but never rely
on the *actual* trajectory of a real loss (which this codebase's own dummy
encoder/decoder make no promise about epoch to epoch): a small `_ForceMetric`
callback, placed before `ReduceLrOnPlateau` in `Trainer.callbacks`, overwrites one
metric key with a fully deterministic, hand-picked sequence before
`ReduceLrOnPlateau.onEpochEnd` ever reads it (callbacks are called in order against
the exact same, mutable epoch-metrics dict; see `TrainerCallback.onEpochEnd`'s own
docstring), so these tests stay exactly reproducible like every other test in this
suite instead of depending on real training dynamics.
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
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_reduce_lr_test")
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


@registerDecoder("dummy_signal_decoder_reduce_lr_test")
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
    """Overwrites one epoch-metrics key with a hand-picked, deterministic sequence.

    Placed before `ReduceLrOnPlateau`/`EarlyStopping` in `Trainer.callbacks` so
    those callbacks observe exactly these values instead of whatever a real
    (unpredictable, epoch-to-epoch) training loss happens to produce.
    """

    def __init__(self, key: str, values: list[float]) -> None:
        self.key = key
        self._values = iter(values)

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        metrics[self.key] = next(self._values)


def _buildSingleModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_reduce_lr_test",
                "decoder": "dummy_signal_decoder_reduce_lr_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int = 2, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


def _buildTrainer(**kwargs) -> Trainer:
    return Trainer(_buildSingleModalityModel(), device="cpu", **kwargs)


class TestConstructorValidation:
    def test_factor_must_be_in_open_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="factor"):
            ReduceLrOnPlateau(factor=0.0)
        with pytest.raises(ValueError, match="factor"):
            ReduceLrOnPlateau(factor=1.0)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            ReduceLrOnPlateau(mode="sideways")


class TestDirectCall:
    """Calls `onEpochEnd` directly against crafted metrics, for precise assertions."""

    def test_missing_monitor_key_raises_key_error(self) -> None:
        trainer = _buildTrainer()
        callback = ReduceLrOnPlateau(monitor="val/loss/total")
        with pytest.raises(KeyError, match="val/loss/total"):
            callback.onEpochEnd(trainer, 0, {"train/loss/total": 1.0})

    def test_reduces_lr_once_patience_exceeded(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        callback = ReduceLrOnPlateau(
            monitor="loss", mode="min", factor=0.5, patience=1, threshold=0.0
        )
        for value in [10.0, 10.0, 10.0]:  # improves, then two non-improving epochs
            callback.onEpochEnd(trainer, 0, {"loss": value})
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_does_not_reduce_lr_while_improving(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        callback = ReduceLrOnPlateau(monitor="loss", mode="min", factor=0.5, patience=1)
        for value in [10.0, 9.0, 8.0, 7.0]:
            callback.onEpochEnd(trainer, 0, {"loss": value})
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_min_lr_floors_the_reduction(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        callback = ReduceLrOnPlateau(
            monitor="loss", factor=0.5, patience=0, threshold=0.0, min_lr=0.08
        )
        callback.onEpochEnd(trainer, 0, {"loss": 10.0})  # first value: establishes best
        callback.onEpochEnd(trainer, 1, {"loss": 10.0})  # plateau: would-be new lr = 0.05
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.08)  # floored

    def test_eps_skips_a_negligible_update(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        callback = ReduceLrOnPlateau(
            monitor="loss", factor=0.999, patience=0, threshold=0.0, eps=1.0
        )
        callback.onEpochEnd(trainer, 0, {"loss": 10.0})
        callback.onEpochEnd(trainer, 1, {"loss": 10.0})  # new lr would be 0.0999: skipped
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_min_lr_list_applies_per_parameter_group(self) -> None:
        model = _buildSingleModalityModel()
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoders.parameters(), "lr": 0.1},
                {"params": model.decoders.parameters(), "lr": 0.2},
            ]
        )
        trainer = Trainer(model, optimizer=optimizer, device="cpu")
        callback = ReduceLrOnPlateau(
            monitor="loss", factor=0.5, patience=0, threshold=0.0, min_lr=[0.06, 0.0]
        )
        callback.onEpochEnd(trainer, 0, {"loss": 10.0})
        callback.onEpochEnd(trainer, 1, {"loss": 10.0})
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.06)  # floored
        assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(0.1)  # 0.2 * 0.5

    def test_mismatched_min_lr_list_length_raises(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        callback = ReduceLrOnPlateau(monitor="loss", patience=0, threshold=0.0, min_lr=[0.01, 0.02])
        with pytest.raises(ValueError, match="min_lr"):
            callback.onEpochEnd(trainer, 0, {"loss": 10.0})
            callback.onEpochEnd(trainer, 1, {"loss": 10.0})


class TestFullTrainingRun:
    def test_lr_drops_on_the_forced_plateau_epoch(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 10.0, 10.0, 10.0, 10.0]),
            ReduceLrOnPlateau(monitor="train/loss/total", factor=0.5, patience=1, threshold=0.0),
        ]

        trainer.fit(_fixedDataset(), num_epochs=5)

        # Epoch 0 establishes best=10; epochs 1-2 are non-improving (patience=1
        # tolerates one), so the reduction fires at epoch 2, and again at epoch 4
        # after patience resets.
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)

    def test_never_reduces_if_the_metric_keeps_improving(self) -> None:
        trainer = _buildTrainer(optimizer_kwargs={"lr": 0.1})
        trainer.callbacks = [
            _ForceMetric("train/loss/total", [10.0, 9.0, 8.0, 7.0, 6.0]),
            ReduceLrOnPlateau(monitor="train/loss/total", factor=0.5, patience=1),
        ]

        trainer.fit(_fixedDataset(), num_epochs=5)

        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
