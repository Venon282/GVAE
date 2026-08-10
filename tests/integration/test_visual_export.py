"""Integration tests for `evaluation.visual_export`.

Uses its own trivial linear dummy encoder/decoder registered under a
`_visual_export_test` suffix.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.models.global_vae import GlobalVae

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_visual_export_test")
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


@registerDecoder("dummy_signal_decoder_visual_export_test")
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
                "encoder": "dummy_signal_encoder_visual_export_test",
                "decoder": "dummy_signal_decoder_visual_export_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestExportEvaluationFigures:
    def test_writes_one_reconstruction_grid_and_two_latent_figures(self, tmp_path: Path) -> None:
        model = _buildModel()
        paths = exportEvaluationFigures(model, _fixedDataset(2), tmp_path, device="cpu")

        assert len(paths) == 3  # 1 reconstruction grid + 1 latent scatter + 1 KL bar chart
        for path in paths:
            assert path.exists()
            assert path.suffix == ".png"

    def test_filenames_reference_the_modality_and_latent_space(self, tmp_path: Path) -> None:
        model = _buildModel()
        paths = exportEvaluationFigures(model, _fixedDataset(2), tmp_path, device="cpu")
        names = {path.name for path in paths}
        assert names == {
            "reconstructions_signal.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        }

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        model = _buildModel()
        output_dir = tmp_path / "nested" / "results"
        exportEvaluationFigures(model, _fixedDataset(1), output_dir, device="cpu")
        assert output_dir.exists()

    def test_inverse_transform_does_not_raise(self, tmp_path: Path) -> None:
        model = _buildModel()
        paths = exportEvaluationFigures(
            model,
            _fixedDataset(1),
            tmp_path,
            device="cpu",
            inverse_transforms={"signal": lambda x: x * 2.0 + 1.0},
        )
        assert all(path.exists() for path in paths)

    def test_max_examples_and_latent_projection_method_are_forwarded(self, tmp_path: Path) -> None:
        model = _buildModel()
        paths = exportEvaluationFigures(
            model,
            _fixedDataset(1),
            tmp_path,
            device="cpu",
            max_examples=2,
            latent_projection_method="pca",
        )
        assert all(path.exists() for path in paths)

    def test_unknown_modality_in_inverse_transforms_is_simply_unused(self, tmp_path: Path) -> None:
        """A key not matching any real modality should not raise: it is simply never looked up."""
        model = _buildModel()
        paths = exportEvaluationFigures(
            model,
            _fixedDataset(1),
            tmp_path,
            device="cpu",
            inverse_transforms={"does_not_exist": lambda x: x},
        )
        assert all(path.exists() for path in paths)

    def test_raises_on_empty_dataloader(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="never observed"):
            exportEvaluationFigures(model, [], "/tmp/unused_output_dir", device="cpu")
