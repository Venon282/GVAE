"""Integration tests for `Trainer` (spec §10, §6.1 milestone 1).

Uses trivial linear dummy encoders/decoders (single-modality for most
tests, matching spec §6.1 milestone 1's `signal -> z -> signal` case;
two modalities plus a dummy PoE fusion for the modality-dropout tests,
spec §5), built through `GlobalVae.createSingleLatent`, since `Trainer`
itself is what is under test here, not any real modality architecture.
Dummy registry names are suffixed `_trainer_test` so they cannot
collide with dummy fixtures in sibling integration tests.
"""

import logging

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.constant import ConstantBetaSchedule
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.callbacks import TrainerCallback
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_trainer_test")
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


@registerDecoder("dummy_signal_decoder_trainer_test")
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


@registerEncoder("dummy_image_encoder_trainer_test")
class _DummyImageEncoder(AbstractEncoder):
    """A second modality, used only by the modality-dropout tests."""

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
        return "image"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_image_decoder_trainer_test")
class _DummyImageDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "image"


@registerFusion("dummy_poe_trainer_test")
class _DummyProductOfExperts(AbstractFusion):
    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_mu_sum: torch.Tensor | None = None
        total_precision: torch.Tensor | None = None
        for mu, logvar in params.values():
            precision = torch.exp(-logvar)
            weighted_mu_sum = (
                precision * mu if weighted_mu_sum is None else weighted_mu_sum + precision * mu
            )
            total_precision = precision if total_precision is None else total_precision + precision
        assert weighted_mu_sum is not None
        assert total_precision is not None
        return weighted_mu_sum / total_precision, -torch.log(total_precision)

    @property
    def handlesMissingModalities(self) -> bool:
        return True


def _buildSingleModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_trainer_test",
                "decoder": "dummy_signal_decoder_trainer_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _buildTwoModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_trainer_test",
                "decoder": "dummy_signal_decoder_trainer_test",
            },
            "image": {
                "encoder": "dummy_image_encoder_trainer_test",
                "decoder": "dummy_image_decoder_trainer_test",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="dummy_poe_trainer_test",
    )


def _fixedDataset(
    num_batches: int, input_dim: int = INPUT_DIM, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    """A small, deterministic, re-iterable "dataset" (a plain list of batches)."""
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, input_dim)} for _ in range(num_batches)]


class _RecordingCallback(TrainerCallback):
    """Records every hook call for assertion, without touching training itself."""

    def __init__(self) -> None:
        self.train_begin_calls = 0
        self.epoch_begin_calls: list[int] = []
        self.step_end_calls: list[tuple[int, dict[str, float]]] = []
        self.epoch_end_calls: list[tuple[int, dict[str, float]]] = []
        self.train_end_calls = 0

    def onTrainBegin(self, trainer: Trainer) -> None:
        self.train_begin_calls += 1

    def onEpochBegin(self, trainer: Trainer, epoch: int) -> None:
        self.epoch_begin_calls.append(epoch)

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        self.step_end_calls.append((step, metrics))

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.epoch_end_calls.append((epoch, metrics))

    def onTrainEnd(self, trainer: Trainer) -> None:
        self.train_end_calls += 1


class _RaisingCallback(TrainerCallback):
    """Raises partway through training, to prove onTrainEnd still fires (finally-block)."""

    def __init__(self, raise_at_step: int) -> None:
        self.raise_at_step = raise_at_step
        self.train_end_calls = 0

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        if step == self.raise_at_step:
            raise RuntimeError("boom")

    def onTrainEnd(self, trainer: Trainer) -> None:
        self.train_end_calls += 1


class TestForwardAndLosses:
    def test_total_loss_is_reconstruction_plus_regularization(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        batch = {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}
        losses = trainer.computeLosses(batch, step=0)
        assert torch.allclose(
            losses.total, losses.reconstruction + losses.regularization, atol=1e-6
        )

    def test_as_metrics_keys(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        batch = {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}
        metrics = trainer.computeLosses(batch, step=0).asMetrics()
        assert set(metrics) == {"loss/total", "loss/reconstruction", "loss/regularization"}

    def test_empty_batch_raises(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        with pytest.raises(ValueError, match="empty batch"):
            trainer.computeLosses({}, step=0)


class TestFitReducesLoss:
    def test_loss_decreases_over_epochs_on_a_fixed_dataset(self) -> None:
        torch.manual_seed(42)
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 0.05})
        dataset = _fixedDataset(num_batches=4)

        history = trainer.fit(dataset, num_epochs=30)

        first_few = sum(entry["train/loss/total"] for entry in history[:5]) / 5
        last_few = sum(entry["train/loss/total"] for entry in history[-5:]) / 5
        assert last_few < first_few


class TestOptimizerConfigurability:
    def test_optimizer_class_is_instantiated_with_kwargs(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(
            model, optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.1}, device="cpu"
        )
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_optimizer_instance_is_used_as_is(self) -> None:
        model = _buildSingleModalityModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        trainer = Trainer(model, optimizer=optimizer, device="cpu")
        assert trainer.optimizer is optimizer

    def test_default_optimizer_is_adam(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        assert isinstance(trainer.optimizer, torch.optim.Adam)


class TestDeviceHandling:
    def test_explicit_cpu_device_is_used(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        assert trainer.device == torch.device("cpu")
        assert next(trainer.model.parameters()).device == torch.device("cpu")

    def test_default_device_resolves_without_error(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model)
        assert trainer.device in (torch.device("cpu"), torch.device("cuda"))


class TestBetaSchedules:
    def test_zero_constant_schedule_gives_zero_regularization_loss(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(
            model, device="cpu", beta_schedules={"z_fused": ConstantBetaSchedule(value=0.0)}
        )
        history = trainer.fit(_fixedDataset(num_batches=2), num_epochs=1)
        assert history[-1]["train/loss/regularization"] == pytest.approx(0.0, abs=1e-6)

    def test_schedule_overrides_base_beta_for_that_latent_space(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(
            model,
            device="cpu",
            beta=5.0,  # would give a large, clearly non-zero regularization loss if not overridden
            beta_schedules={"z_fused": ConstantBetaSchedule(value=0.0)},
        )
        history = trainer.fit(_fixedDataset(num_batches=2), num_epochs=1)
        assert history[-1]["train/loss/regularization"] == pytest.approx(0.0, abs=1e-6)

    def test_compute_beta_reflects_the_schedule_at_different_steps(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(
            model,
            device="cpu",
            beta_schedules={
                "z_fused": LinearWarmupBetaSchedule(
                    warmup_steps=100, start_value=0.0, end_value=1.0
                )
            },
        )
        assert trainer._computeBeta(0)["z_fused"] == pytest.approx(0.0)
        assert trainer._computeBeta(100)["z_fused"] == pytest.approx(1.0)

    def test_latent_space_without_a_schedule_falls_back_to_base_beta(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", beta=0.25, beta_schedules=None)
        assert trainer._computeBeta(0)["z_fused"] == pytest.approx(0.25)


class TestCallbacks:
    def test_hooks_fire_the_expected_number_of_times_with_expected_metric_keys(self) -> None:
        model = _buildSingleModalityModel()
        callback = _RecordingCallback()
        trainer = Trainer(model, device="cpu", callbacks=[callback])
        dataset = _fixedDataset(num_batches=3)

        trainer.fit(dataset, num_epochs=2)

        assert callback.train_begin_calls == 1
        assert callback.epoch_begin_calls == [0, 1]
        assert len(callback.step_end_calls) == 6  # 3 batches * 2 epochs
        for _, metrics in callback.step_end_calls:
            assert set(metrics) == {"loss/total", "loss/reconstruction", "loss/regularization"}
        assert [epoch for epoch, _ in callback.epoch_end_calls] == [0, 1]
        for _, metrics in callback.epoch_end_calls:
            assert "train/loss/total" in metrics
        assert callback.train_end_calls == 1

    def test_on_train_end_still_fires_when_training_raises(self) -> None:
        model = _buildSingleModalityModel()
        callback = _RaisingCallback(raise_at_step=1)
        trainer = Trainer(model, device="cpu", callbacks=[callback])
        dataset = _fixedDataset(num_batches=3)

        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(dataset, num_epochs=2)

        assert callback.train_end_calls == 1

    def test_progress_is_logged_via_standard_logging_not_print(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", log_every_n_steps=1)
        with caplog.at_level(logging.INFO, logger="global_vae.training.trainer"):
            trainer.fit(_fixedDataset(num_batches=2), num_epochs=1)
        assert any("epoch" in record.message for record in caplog.records)


class TestModalityDropout:
    def test_disabled_by_default_leaves_batch_unchanged(self) -> None:
        model = _buildTwoModalityModel()
        trainer = Trainer(model, device="cpu")
        batch = {
            "signal": torch.randn(BATCH_SIZE, INPUT_DIM),
            "image": torch.randn(BATCH_SIZE, INPUT_DIM),
        }
        assert trainer._applyModalityDropout(batch) is batch

    def test_full_dropout_still_keeps_at_least_one_modality(self) -> None:
        model = _buildTwoModalityModel()
        trainer = Trainer(model, device="cpu", modality_dropout_p=1.0)
        batch = {
            "signal": torch.randn(BATCH_SIZE, INPUT_DIM),
            "image": torch.randn(BATCH_SIZE, INPUT_DIM),
        }
        for _ in range(10):
            kept = trainer._applyModalityDropout(batch)
            assert 1 <= len(kept) <= 2

    def test_single_modality_model_ignores_dropout(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", modality_dropout_p=1.0)
        batch = {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}
        assert trainer._applyModalityDropout(batch) is batch

    def test_two_modality_model_still_trains_with_dropout_enabled(self) -> None:
        model = _buildTwoModalityModel()
        trainer = Trainer(model, device="cpu", modality_dropout_p=0.5)
        dataset = [
            {
                "signal": torch.randn(BATCH_SIZE, INPUT_DIM),
                "image": torch.randn(BATCH_SIZE, INPUT_DIM),
            }
            for _ in range(3)
        ]
        history = trainer.fit(dataset, num_epochs=2)
        assert len(history) == 2
        for entry in history:
            assert torch.isfinite(torch.tensor(entry["train/loss/total"]))

    def test_invalid_modality_dropout_p_raises(self) -> None:
        model = _buildSingleModalityModel()
        with pytest.raises(ValueError, match="modality_dropout_p"):
            Trainer(model, modality_dropout_p=1.5)


class TestGradientClipping:
    def test_clips_the_gradient_norm(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", grad_clip_norm=0.01, optimizer_kwargs={"lr": 0.0})
        batch = {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}

        trainer.fitEpoch([batch], epoch=0)

        total_norm_sq = sum(
            p.grad.pow(2).sum() for p in trainer.model.parameters() if p.grad is not None
        )
        assert torch.sqrt(total_norm_sq) <= 0.01 + 1e-4

    def test_invalid_grad_clip_norm_raises(self) -> None:
        model = _buildSingleModalityModel()
        with pytest.raises(ValueError, match="grad_clip_norm"):
            Trainer(model, grad_clip_norm=-1.0)


class TestHistoryAndResume:
    def test_history_has_one_entry_per_epoch(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        history = trainer.fit(_fixedDataset(num_batches=2), num_epochs=3)
        assert len(history) == 3
        assert history is trainer.history
        for entry in history:
            assert "train/loss/total" in entry

    def test_fit_called_twice_continues_epoch_and_step_numbering(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        dataset = _fixedDataset(num_batches=2)

        trainer.fit(dataset, num_epochs=2)
        assert trainer.start_epoch == 2
        assert trainer.global_step == 4

        trainer.fit(dataset, num_epochs=1)
        assert trainer.start_epoch == 3
        assert trainer.global_step == 6
        assert len(trainer.history) == 3

    def test_validation_metrics_are_merged_into_epoch_history(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        history = trainer.fit(
            _fixedDataset(num_batches=2), num_epochs=1, val_dataloader=_fixedDataset(num_batches=1)
        )
        assert "train/loss/total" in history[0]
        assert "val/loss/total" in history[0]


class TestErrorPaths:
    def test_invalid_log_every_n_steps_raises(self) -> None:
        model = _buildSingleModalityModel()
        with pytest.raises(ValueError, match="log_every_n_steps"):
            Trainer(model, log_every_n_steps=0)

    def test_empty_dataloader_raises_in_fit_epoch(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        with pytest.raises(ValueError, match="empty dataloader"):
            trainer.fitEpoch([], epoch=0)

    def test_empty_dataloader_raises_in_evaluate(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        with pytest.raises(ValueError, match="empty dataloader"):
            trainer.evaluate([])

    def test_non_positive_num_epochs_raises(self) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        with pytest.raises(ValueError, match="num_epochs"):
            trainer.fit(_fixedDataset(num_batches=1), num_epochs=0)
