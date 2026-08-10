"""Integration tests for `visualization.latent_plot`.

Uses its own trivial linear dummy encoder/decoder registered under a
`_latent_plot_test` suffix (see `test_trainer.py`'s module docstring
for why test files do not import each other's dummy fixtures).
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
from global_vae.visualization.latent_plot import (
    collectLatentParams,
    collectLatentSamples,
    plotLatentSpace,
    plotPerDimensionKl,
    projectLatentSamples,
)

INPUT_DIM = 16
LATENT_DIM = 5
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_latent_plot_test")
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


@registerDecoder("dummy_signal_decoder_latent_plot_test")
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


def _buildModel(latent_dim: int = LATENT_DIM) -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_latent_plot_test",
                "decoder": "dummy_signal_decoder_latent_plot_test",
            },
        },
        latent_dim=latent_dim,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestProjectLatentSamples:
    def test_none_method_with_matching_dims_returns_unchanged_values(self) -> None:
        z = torch.randn(10, 2)
        projected = projectLatentSamples(z, method="none", n_components=2)
        assert torch.allclose(projected, z)

    def test_none_method_with_mismatched_dims_raises(self) -> None:
        z = torch.randn(10, 5)
        with pytest.raises(ValueError, match="none"):
            projectLatentSamples(z, method="none", n_components=2)

    def test_auto_is_identity_when_already_the_right_size(self) -> None:
        z = torch.randn(10, 2)
        projected = projectLatentSamples(z, method="auto", n_components=2)
        assert torch.allclose(projected, z)

    def test_auto_uses_pca_when_dimensions_differ(self) -> None:
        z = torch.randn(20, 5)
        projected = projectLatentSamples(z, method="auto", n_components=2)
        assert projected.shape == (20, 2)

    def test_pca_recovers_the_dominant_direction_of_variance(self) -> None:
        """A synthetic dataset that varies almost entirely along one axis: PCA's first
        component should capture nearly all of it."""
        torch.manual_seed(0)
        n = 200
        dominant = torch.randn(n, 1) * 10.0
        noise = torch.randn(n, 4) * 0.01
        z = torch.cat([dominant, noise], dim=1)
        projected = projectLatentSamples(z, method="pca", n_components=1)
        # the first PCA component should correlate almost perfectly (up to sign) with
        # the dominant original axis
        correlation = torch.corrcoef(torch.stack([projected[:, 0], dominant[:, 0]]))[0, 1]
        assert correlation.abs() > 0.99

    def test_tsne_projects_to_the_requested_dimensionality(self) -> None:
        z = torch.randn(30, 5)
        projected = projectLatentSamples(z, method="tsne", n_components=2, seed=0, perplexity=5)
        assert projected.shape == (30, 2)

    def test_umap_projects_to_the_requested_dimensionality(self) -> None:
        z = torch.randn(30, 5)
        projected = projectLatentSamples(z, method="umap", n_components=2, seed=0, n_neighbors=5)
        assert projected.shape == (30, 2)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            projectLatentSamples(torch.randn(5, 3), method="does_not_exist")

    def test_non_positive_n_components_raises(self) -> None:
        with pytest.raises(ValueError, match="n_components"):
            projectLatentSamples(torch.randn(5, 3), n_components=0)

    def test_empty_z_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            projectLatentSamples(torch.empty(0, 3))


class TestPlotLatentSpace:
    def test_returns_a_figure_for_each_n_components(self) -> None:
        z = torch.randn(20, 5)
        for n_components in (1, 2, 3):
            fig = plotLatentSpace(z, n_components=n_components)
            assert fig is not None
            plt.close(fig)

    def test_continuous_labels_add_a_colorbar(self) -> None:
        z = torch.randn(20, 5)
        labels = torch.rand(20)
        fig = plotLatentSpace(z, labels=labels)
        assert len(fig.axes) >= 2  # the main axes plus the colorbar's own axes
        plt.close(fig)

    def test_categorical_labels_add_one_legend_entry_per_unique_value(self) -> None:
        z = torch.randn(20, 5)
        labels = ["a"] * 10 + ["b"] * 5 + ["c"] * 5
        fig = plotLatentSpace(z, labels=labels)
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 3
        plt.close(fig)

    def test_no_labels_does_not_raise(self) -> None:
        fig = plotLatentSpace(torch.randn(20, 5))
        plt.close(fig)

    def test_invalid_n_components_raises(self) -> None:
        with pytest.raises(ValueError, match="n_components"):
            plotLatentSpace(torch.randn(5, 3), n_components=4)

    def test_mismatched_labels_length_raises(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            plotLatentSpace(torch.randn(5, 3), labels=torch.rand(4))


class TestCollectLatentParamsAndSamples:
    def test_shapes_match_the_whole_dataset(self) -> None:
        model = _buildModel()
        mu, logvar = collectLatentParams(model, _fixedDataset(3), "z_fused", device="cpu")
        assert mu.shape == (3 * BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (3 * BATCH_SIZE, LATENT_DIM)

    def test_max_samples_truncates(self) -> None:
        model = _buildModel()
        mu, _ = collectLatentParams(model, _fixedDataset(3), "z_fused", device="cpu", max_samples=5)
        assert mu.shape[0] == 5

    def test_use_mean_true_matches_mu_exactly(self) -> None:
        model = _buildModel()
        dataset = _fixedDataset(2)
        mu, _ = collectLatentParams(model, dataset, "z_fused", device="cpu")
        samples = collectLatentSamples(model, dataset, "z_fused", device="cpu", use_mean=True)
        assert torch.equal(mu, samples)

    def test_use_mean_false_differs_from_mu(self) -> None:
        model = _buildModel()
        dataset = _fixedDataset(2)
        mu, _ = collectLatentParams(model, dataset, "z_fused", device="cpu")
        samples = collectLatentSamples(model, dataset, "z_fused", device="cpu", use_mean=False)
        assert samples.shape == mu.shape
        assert not torch.equal(mu, samples)

    def test_unknown_latent_name_raises(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="never observed"):
            collectLatentParams(model, _fixedDataset(2), "does_not_exist", device="cpu")

    def test_empty_dataloader_raises(self) -> None:
        model = _buildModel()
        with pytest.raises(ValueError, match="never observed"):
            collectLatentParams(model, [], "z_fused", device="cpu")


class TestPlotPerDimensionKl:
    def test_bar_heights_match_the_kl_formula(self) -> None:
        mu = torch.randn(50, LATENT_DIM)
        logvar = torch.randn(50, LATENT_DIM)
        expected = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)

        fig = plotPerDimensionKl(mu, logvar)
        bar_heights = [patch.get_height() for patch in fig.axes[0].patches]
        assert torch.allclose(torch.tensor(bar_heights, dtype=expected.dtype), expected, atol=1e-5)
        plt.close(fig)

    def test_number_of_bars_matches_latent_dim(self) -> None:
        mu = torch.zeros(10, LATENT_DIM)
        logvar = torch.zeros(10, LATENT_DIM)
        fig = plotPerDimensionKl(mu, logvar)
        assert len(fig.axes[0].patches) == LATENT_DIM
        plt.close(fig)

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            plotPerDimensionKl(torch.zeros(5, 3), torch.zeros(5, 4))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotPerDimensionKl(torch.empty(0, 3), torch.empty(0, 3))
