"""Integration tests for `evaluation.metrics` and `evaluation.evaluate`.

Uses its own trivial linear dummy encoder/decoder registered under an
`_evaluate_test` suffix (see `test_trainer.py`'s module docstring for
why test files do not import each other's dummy fixtures).
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.evaluation.evaluate import EvaluationResults, evaluate
from global_vae.evaluation.metrics import (
    DEFAULT_RECONSTRUCTION_METRICS,
    computeMae,
    computeMse,
    computePearsonR,
    computeR2,
    computeRmse,
)
from global_vae.models.global_vae import GlobalVae

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_evaluate_test")
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


@registerDecoder("dummy_signal_decoder_evaluate_test")
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


def _buildModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_evaluate_test",
                "decoder": "dummy_signal_decoder_evaluate_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestMetrics:
    def test_mse_matches_functional_mse_loss(self) -> None:
        a, b = torch.randn(5, 4), torch.randn(5, 4)
        assert computeMse(a, b) == pytest.approx(F.mse_loss(a, b).item())

    def test_rmse_is_sqrt_of_mse(self) -> None:
        a, b = torch.randn(5, 4), torch.randn(5, 4)
        assert computeRmse(a, b) == pytest.approx(math.sqrt(computeMse(a, b)))

    def test_mae_matches_functional_l1_loss(self) -> None:
        a, b = torch.randn(5, 4), torch.randn(5, 4)
        assert computeMae(a, b) == pytest.approx(F.l1_loss(a, b).item())

    def test_r2_is_one_for_a_perfect_reconstruction(self) -> None:
        target = torch.randn(20, 3)
        assert computeR2(target, target) == pytest.approx(1.0, abs=1e-5)

    def test_r2_is_zero_when_predicting_the_global_mean(self) -> None:
        """computeR2 pools every element into one scalar baseline (target.mean()), matching
        computeMse's own flat pooling, not a per-column mean (sklearn's r2_score convention)."""
        target = torch.randn(50, 3)
        mean_prediction = torch.full_like(target, target.mean().item())
        assert computeR2(mean_prediction, target) == pytest.approx(0.0, abs=1e-5)

    def test_r2_is_nan_for_a_constant_target(self) -> None:
        target = torch.ones(10, 2)
        assert math.isnan(computeR2(torch.randn(10, 2), target))

    def test_pearson_r_is_one_for_a_perfectly_correlated_pair(self) -> None:
        target = torch.randn(30)
        scaled = target * 2.0 + 5.0  # a linear transform: correlation is still 1
        assert computePearsonR(scaled, target) == pytest.approx(1.0, abs=1e-5)

    def test_pearson_r_is_nan_for_a_constant_input(self) -> None:
        assert math.isnan(computePearsonR(torch.ones(10), torch.randn(10)))

    def test_default_metrics_dict_contains_the_expected_keys(self) -> None:
        assert set(DEFAULT_RECONSTRUCTION_METRICS) == {"mse", "rmse", "mae", "r2", "pearson_r"}


class TestEvaluationResults:
    def test_to_dict_round_trips_the_fields(self) -> None:
        results = EvaluationResults(
            num_samples=10,
            total_reconstruction_loss=0.5,
            total_regularization_loss=0.1,
            reconstruction_metrics={"signal": {"mse": 0.5}},
            regularization_metrics={"z_fused": {"configured": 0.1, "kl_standard_normal": 0.1}},
        )
        as_dict = results.toDict()
        assert as_dict["num_samples"] == 10
        assert as_dict["reconstruction_metrics"]["signal"]["mse"] == 0.5

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        results = EvaluationResults(
            num_samples=5, total_reconstruction_loss=0.2, total_regularization_loss=0.3
        )
        path = tmp_path / "report.json"
        results.save(path)
        with path.open() as f:
            loaded = json.load(f)
        assert loaded["num_samples"] == 5

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "report.json"
        EvaluationResults(0, 0.0, 0.0).save(path)
        assert path.exists()

    def test_summary_is_a_readable_multiline_string(self) -> None:
        results = EvaluationResults(
            num_samples=10,
            total_reconstruction_loss=0.5,
            total_regularization_loss=0.1,
            reconstruction_metrics={"signal": {"mse": 0.5}},
        )
        summary = results.summary()
        assert "10 samples" in summary
        assert "mse=0.500000" in summary


class TestEvaluate:
    def test_returns_finite_aggregate_losses(self) -> None:
        model = _buildModel()
        results = evaluate(model, _fixedDataset(3), device="cpu")
        assert math.isfinite(results.total_reconstruction_loss)
        assert math.isfinite(results.total_regularization_loss)

    def test_num_samples_matches_the_dataset_size(self) -> None:
        model = _buildModel()
        results = evaluate(model, _fixedDataset(3), device="cpu")
        assert results.num_samples == 3 * BATCH_SIZE

    def test_reconstruction_metrics_include_every_default_metric(self) -> None:
        model = _buildModel()
        results = evaluate(model, _fixedDataset(2), device="cpu")
        assert set(results.reconstruction_metrics["signal"]) == set(DEFAULT_RECONSTRUCTION_METRICS)

    def test_regularization_metrics_include_configured_and_kl_standard_normal(self) -> None:
        model = _buildModel()
        results = evaluate(model, _fixedDataset(2), device="cpu")
        assert set(results.regularization_metrics["z_fused"]) == {
            "configured",
            "kl_standard_normal",
        }
        # the default regularizer *is* kl_standard_normal, so the two must match exactly
        entry = results.regularization_metrics["z_fused"]
        assert entry["configured"] == pytest.approx(entry["kl_standard_normal"])

    def test_use_mean_true_is_deterministic_across_calls(self) -> None:
        model = _buildModel()
        dataset = _fixedDataset(2)
        first = evaluate(model, dataset, device="cpu", use_mean=True)
        second = evaluate(model, dataset, device="cpu", use_mean=True)
        assert first.reconstruction_metrics == second.reconstruction_metrics

    def test_custom_reconstruction_metrics_replaces_the_default_set(self) -> None:
        model = _buildModel()
        results = evaluate(
            model, _fixedDataset(2), device="cpu", reconstruction_metrics={"mse": computeMse}
        )
        assert set(results.reconstruction_metrics["signal"]) == {"mse"}

    def test_beta_scales_the_total_regularization_loss(self) -> None:
        model = _buildModel()
        dataset = _fixedDataset(2)
        unweighted = evaluate(model, dataset, device="cpu", beta=1.0)
        halved = evaluate(model, dataset, device="cpu", beta=0.5)
        assert halved.total_regularization_loss == pytest.approx(
            0.5 * unweighted.total_regularization_loss, rel=1e-4
        )

    def test_max_samples_bounds_num_samples(self) -> None:
        model = _buildModel()
        results = evaluate(model, _fixedDataset(3), device="cpu", max_samples=5)
        assert results.num_samples == 5
        assert results.reconstruction_metrics["signal"]  # still computed, just over fewer rows

    def test_empty_dataloader_raises(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="empty dataloader"):
            evaluate(model, [], device="cpu")

    def test_calls_model_eval(self) -> None:
        model = _buildModel()
        model.train()
        evaluate(model, _fixedDataset(1), device="cpu")
        assert not model.training
