"""Integration tests for `evaluation.cross_modal` (spec §5,
`docs/adr/0016-cross-modal-reconstruction-reporting.md`).

Uses its own trivial linear dummy encoders/decoders and a dummy Product-of-Experts
fusion, registered under a `_cross_modal_eval_test` suffix (see `test_trainer.py`'s
module docstring for why sibling test files do not share dummy fixtures).
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.evaluation.cross_modal import (
    computeCrossModalReconstructionMetrics,
    exportCrossModalFigures,
)
from global_vae.evaluation.metrics import computeMse
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion
from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.reconstruction_plot import collectCrossModalReconstructions

INPUT_DIM = 12
LATENT_DIM = 4
BATCH_SIZE = 6


@registerEncoder("dummy_a_encoder_cross_modal_eval_test")
class _DummyEncoderA(AbstractEncoder):
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
        return "a"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerEncoder("dummy_b_encoder_cross_modal_eval_test")
class _DummyEncoderB(AbstractEncoder):
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
        return "b"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_a_decoder_cross_modal_eval_test")
class _DummyDecoderA(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "a"


@registerDecoder("dummy_b_decoder_cross_modal_eval_test")
class _DummyDecoderB(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "b"


@registerFusion("dummy_poe_cross_modal_eval_test")
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
            "a": {
                "encoder": "dummy_a_encoder_cross_modal_eval_test",
                "decoder": "dummy_a_decoder_cross_modal_eval_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _buildTwoModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "a": {
                "encoder": "dummy_a_encoder_cross_modal_eval_test",
                "decoder": "dummy_a_decoder_cross_modal_eval_test",
            },
            "b": {
                "encoder": "dummy_b_encoder_cross_modal_eval_test",
                "decoder": "dummy_b_decoder_cross_modal_eval_test",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="dummy_poe_cross_modal_eval_test",
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [
        {
            "a": torch.randn(BATCH_SIZE, INPUT_DIM),
            "b": torch.randn(BATCH_SIZE, INPUT_DIM),
        }
        for _ in range(num_batches)
    ]


class TestComputeCrossModalReconstructionMetrics:
    def test_matches_direct_metric_computation(self) -> None:
        original = torch.randn(4, 10)
        reconstruction = torch.randn(4, 10)
        collected = {frozenset({"a"}): {"a": (original, reconstruction)}}
        metrics = computeCrossModalReconstructionMetrics(collected, metrics={"mse": computeMse})
        assert metrics[frozenset({"a"})]["a"]["mse"] == pytest.approx(
            computeMse(reconstruction, original)
        )

    def test_default_metrics_match_default_reconstruction_metrics(self) -> None:
        collected = {frozenset({"a"}): {"a": (torch.randn(4, 6), torch.randn(4, 6))}}
        metrics = computeCrossModalReconstructionMetrics(collected)
        assert set(metrics[frozenset({"a"})]["a"]) == {"mse", "rmse", "mae", "r2", "pearson_r"}

    def test_end_to_end_with_a_real_model(self) -> None:
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(model, _fixedDataset(2), device="cpu")
        metrics = computeCrossModalReconstructionMetrics(collected)
        assert set(metrics) == {frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})}
        for per_decoder in metrics.values():
            for per_metric in per_decoder.values():
                assert math.isfinite(per_metric["mse"])

    def test_empty_collected_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            computeCrossModalReconstructionMetrics({})


class TestExportCrossModalFigures:
    def test_single_modality_model_writes_nothing(self, tmp_path: Path) -> None:
        model = _buildSingleModalityModel()
        paths = exportCrossModalFigures(model, _fixedDataset(1), tmp_path, device="cpu")
        assert paths == []
        assert not any(tmp_path.iterdir())

    def test_two_modality_model_writes_one_figure(self, tmp_path: Path) -> None:
        model = _buildTwoModalityModel()
        paths = exportCrossModalFigures(model, _fixedDataset(2), tmp_path, device="cpu")
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "cross_modal_reconstructions.png"

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        model = _buildTwoModalityModel()
        output_dir = tmp_path / "nested" / "results"
        exportCrossModalFigures(model, _fixedDataset(1), output_dir, device="cpu")
        assert output_dir.exists()

    def test_inverse_transforms_do_not_raise(self, tmp_path: Path) -> None:
        model = _buildTwoModalityModel()
        paths = exportCrossModalFigures(
            model,
            _fixedDataset(1),
            tmp_path,
            device="cpu",
            inverse_transforms={"a": lambda x: x * 2.0 + 1.0},
        )
        assert all(path.exists() for path in paths)

    def test_explicit_input_subsets_are_forwarded(self, tmp_path: Path) -> None:
        model = _buildTwoModalityModel()
        paths = exportCrossModalFigures(
            model, _fixedDataset(1), tmp_path, device="cpu", input_subsets=[frozenset({"a"})]
        )
        assert len(paths) == 1

    def test_unknown_modality_in_explicit_subset_still_raises(self, tmp_path: Path) -> None:
        """A genuinely misused call must not be silently swallowed by the
        single-modality no-op path."""
        model = _buildTwoModalityModel()
        with pytest.raises(ValueError, match="does_not_exist"):
            exportCrossModalFigures(
                model,
                _fixedDataset(1),
                tmp_path,
                device="cpu",
                input_subsets=[frozenset({"does_not_exist"})],
            )
