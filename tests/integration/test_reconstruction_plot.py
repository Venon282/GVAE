"""Integration tests for `visualization.reconstruction_plot`.

Uses its own trivial linear dummy encoder/decoder registered under a
`_reconstruction_plot_test` suffix (see `test_trainer.py`'s module
docstring for why test files do not import each other's fixtures).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.reconstruction_plot import (
    collectReconstructions,
    plotReconstruction,
    plotReconstructionGrid,
)

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 6


@registerEncoder("dummy_signal_encoder_reconstruction_plot_test")
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


@registerDecoder("dummy_signal_decoder_reconstruction_plot_test")
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
                "encoder": "dummy_signal_encoder_reconstruction_plot_test",
                "decoder": "dummy_signal_decoder_reconstruction_plot_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestPlotReconstruction:
    def test_returns_a_figure_with_matching_line_data(self) -> None:
        original = torch.arange(10, dtype=torch.float32)
        reconstruction = original + 0.5
        fig = plotReconstruction(original, reconstruction)
        lines = fig.axes[0].lines
        assert len(lines) == 2
        assert torch.allclose(torch.tensor(lines[0].get_ydata()), original)
        assert torch.allclose(torch.tensor(lines[1].get_ydata()), reconstruction)
        plt.close(fig)

    def test_inverse_transform_is_applied_to_both_series(self) -> None:
        original = torch.tensor([0.0, 1.0, 2.0])
        reconstruction = torch.tensor([0.5, 1.5, 2.5])
        fig = plotReconstruction(original, reconstruction, inverse_transform=lambda x: x * 2.0)
        lines = fig.axes[0].lines
        assert torch.allclose(torch.tensor(lines[0].get_ydata()), original * 2.0)
        assert torch.allclose(torch.tensor(lines[1].get_ydata()), reconstruction * 2.0)
        plt.close(fig)

    def test_custom_x_values_are_used(self) -> None:
        original = torch.tensor([0.0, 1.0, 2.0])
        reconstruction = torch.tensor([0.1, 1.1, 2.1])
        x_values = torch.tensor([10.0, 20.0, 30.0])
        fig = plotReconstruction(original, reconstruction, x_values=x_values)
        assert torch.allclose(torch.tensor(fig.axes[0].lines[0].get_xdata()), x_values)
        plt.close(fig)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length"):
            plotReconstruction(torch.zeros(5), torch.zeros(6))

    def test_non_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            plotReconstruction(torch.zeros(2, 5), torch.zeros(2, 5))


class TestPlotReconstructionGrid:
    def test_grid_shape_matches_ncols_and_example_count(self) -> None:
        originals = torch.randn(5, 10)
        reconstructions = torch.randn(5, 10)
        fig = plotReconstructionGrid(originals, reconstructions, ncols=2)
        # 5 examples, 2 columns -> 3 rows of axes = 6 axes total (one turned off)
        assert len(fig.axes) == 6
        plt.close(fig)

    def test_max_examples_truncates(self) -> None:
        originals = torch.randn(10, 8)
        reconstructions = torch.randn(10, 8)
        fig = plotReconstructionGrid(originals, reconstructions, max_examples=3, ncols=3)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_custom_titles_are_used(self) -> None:
        originals = torch.randn(2, 8)
        reconstructions = torch.randn(2, 8)
        fig = plotReconstructionGrid(
            originals, reconstructions, ncols=2, titles=["first", "second"]
        )
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == ["first", "second"]
        plt.close(fig)

    def test_wrong_titles_length_raises(self) -> None:
        originals = torch.randn(3, 8)
        reconstructions = torch.randn(3, 8)
        with pytest.raises(ValueError, match="titles"):
            plotReconstructionGrid(originals, reconstructions, titles=["only one"])

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            plotReconstructionGrid(torch.randn(3, 8), torch.randn(3, 9))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotReconstructionGrid(torch.empty(0, 8), torch.empty(0, 8))

    def test_non_positive_ncols_raises(self) -> None:
        with pytest.raises(ValueError, match="ncols"):
            plotReconstructionGrid(torch.randn(2, 8), torch.randn(2, 8), ncols=0)


class TestCollectReconstructions:
    def test_originals_exactly_match_the_input_batches(self) -> None:
        model = _buildModel()
        dataset = _fixedDataset(2)
        originals, reconstructions = collectReconstructions(model, dataset, "signal", device="cpu")
        expected_originals = torch.cat([batch["signal"] for batch in dataset], dim=0)
        assert torch.equal(originals, expected_originals)
        assert reconstructions.shape == expected_originals.shape

    def test_max_samples_truncates(self) -> None:
        model = _buildModel()
        originals, reconstructions = collectReconstructions(
            model, _fixedDataset(3), "signal", device="cpu", max_samples=4
        )
        assert originals.shape[0] == 4
        assert reconstructions.shape[0] == 4

    def test_unknown_modality_name_raises(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="never observed"):
            collectReconstructions(model, _fixedDataset(2), "does_not_exist", device="cpu")

    def test_empty_dataloader_raises(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="never observed"):
            collectReconstructions(model, [], "signal", device="cpu")

    def test_end_to_end_with_plot_reconstruction(self) -> None:
        model = _buildModel()
        originals, reconstructions = collectReconstructions(
            model, _fixedDataset(1), "signal", device="cpu"
        )
        fig = plotReconstruction(originals[0], reconstructions[0])
        assert fig is not None
        plt.close(fig)
