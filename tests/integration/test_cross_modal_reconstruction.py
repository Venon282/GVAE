"""Integration tests for cross-modal reconstruction reporting
(`visualization.reconstruction_plot`'s `resolveDefaultInputSubsets`,
`collectCrossModalReconstructions`, `plotCrossModalReconstructionMatrix`; spec §5,
`docs/adr/0016-cross-modal-reconstruction-reporting.md`).

Uses its own trivial linear dummy encoders/decoders and a dummy Product-of-Experts
fusion, registered under a `_cross_modal_test` suffix (see `test_trainer.py`'s
module docstring for why sibling test files do not import each other's dummy
fixtures). Three dummy modalities ("a", "b", "c") are defined: most tests only need
two, but a few explicitly confirm nothing here is hardcoded to exactly two.
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
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion
from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.reconstruction_plot import (
    collectCrossModalReconstructions,
    plotCrossModalReconstructionMatrix,
    resolveDefaultInputSubsets,
)

INPUT_DIM = 12
LATENT_DIM = 4
BATCH_SIZE = 6


@registerEncoder("dummy_a_encoder_cross_modal_test")
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


@registerEncoder("dummy_b_encoder_cross_modal_test")
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


@registerEncoder("dummy_c_encoder_cross_modal_test")
class _DummyEncoderC(AbstractEncoder):
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
        return "c"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_a_decoder_cross_modal_test")
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


@registerDecoder("dummy_b_decoder_cross_modal_test")
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


@registerDecoder("dummy_c_decoder_cross_modal_test")
class _DummyDecoderC(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "c"


@registerFusion("dummy_poe_cross_modal_test")
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
                "encoder": "dummy_a_encoder_cross_modal_test",
                "decoder": "dummy_a_decoder_cross_modal_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _buildTwoModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "a": {
                "encoder": "dummy_a_encoder_cross_modal_test",
                "decoder": "dummy_a_decoder_cross_modal_test",
            },
            "b": {
                "encoder": "dummy_b_encoder_cross_modal_test",
                "decoder": "dummy_b_decoder_cross_modal_test",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="dummy_poe_cross_modal_test",
    )


def _buildThreeModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "a": {
                "encoder": "dummy_a_encoder_cross_modal_test",
                "decoder": "dummy_a_decoder_cross_modal_test",
            },
            "b": {
                "encoder": "dummy_b_encoder_cross_modal_test",
                "decoder": "dummy_b_decoder_cross_modal_test",
            },
            "c": {
                "encoder": "dummy_c_encoder_cross_modal_test",
                "decoder": "dummy_c_decoder_cross_modal_test",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="dummy_poe_cross_modal_test",
    )


def _fixedBatch(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "a": torch.randn(BATCH_SIZE, INPUT_DIM),
        "b": torch.randn(BATCH_SIZE, INPUT_DIM),
        "c": torch.randn(BATCH_SIZE, INPUT_DIM),
    }


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [
        {
            "a": torch.randn(BATCH_SIZE, INPUT_DIM),
            "b": torch.randn(BATCH_SIZE, INPUT_DIM),
            "c": torch.randn(BATCH_SIZE, INPUT_DIM),
        }
        for _ in range(num_batches)
    ]


class TestResolveDefaultInputSubsets:
    def test_two_modality_model_gives_singles_plus_full_set(self) -> None:
        model = _buildTwoModalityModel()
        subsets = resolveDefaultInputSubsets(model)
        assert set(subsets) == {frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})}

    def test_three_modality_model_gives_three_singles_plus_full_set(self) -> None:
        model = _buildThreeModalityModel()
        subsets = resolveDefaultInputSubsets(model)
        assert set(subsets) == {
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"c"}),
            frozenset({"a", "b", "c"}),
        }

    def test_single_modality_model_gives_only_its_own_singleton(self) -> None:
        """No duplicate "full set" entry when it would be identical to the one
        singleton already present."""
        model = _buildSingleModalityModel()
        subsets = resolveDefaultInputSubsets(model)
        assert subsets == [frozenset({"a"})]


class TestCollectCrossModalReconstructions:
    def test_default_subsets_cover_every_singleton_and_the_full_set(self) -> None:
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(model, [_fixedBatch()], device="cpu")
        assert set(collected) == {frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})}

    def test_every_decoder_reconstructs_from_a_single_modality_subset(self) -> None:
        """The actual point (spec §5): feeding only 'a' in must still produce a
        reconstruction for decoder 'b', not only for 'a'."""
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(
            model, [_fixedBatch()], input_subsets=[{"a"}], device="cpu"
        )
        assert set(collected[frozenset({"a"})]) == {"a", "b"}

    def test_ground_truth_always_comes_from_the_full_batch(self) -> None:
        """The original paired against decoder 'b' under the {'a'}-only subset must
        equal batch['b'] itself, not some restricted/absent value."""
        model = _buildTwoModalityModel()
        batch = _fixedBatch()
        collected = collectCrossModalReconstructions(
            model, [batch], input_subsets=[{"a"}], device="cpu"
        )
        original_b, _ = collected[frozenset({"a"})]["b"]
        assert torch.equal(original_b, batch["b"])

    def test_use_mean_true_is_deterministic_across_calls(self) -> None:
        model = _buildTwoModalityModel()
        dataset = _fixedDataset(2)
        first = collectCrossModalReconstructions(model, dataset, device="cpu", use_mean=True)
        second = collectCrossModalReconstructions(model, dataset, device="cpu", use_mean=True)
        for subset in first:
            for decoder_name in first[subset]:
                assert torch.equal(first[subset][decoder_name][1], second[subset][decoder_name][1])

    def test_use_mean_false_is_stochastic(self) -> None:
        model = _buildTwoModalityModel()
        dataset = _fixedDataset(2)
        first = collectCrossModalReconstructions(model, dataset, device="cpu", use_mean=False)
        second = collectCrossModalReconstructions(model, dataset, device="cpu", use_mean=False)
        _, first_recon = first[frozenset({"a", "b"})]["a"]
        _, second_recon = second[frozenset({"a", "b"})]["a"]
        assert not torch.equal(first_recon, second_recon)

    def test_max_samples_truncates_every_subset(self) -> None:
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(
            model, _fixedDataset(3), device="cpu", max_samples=5
        )
        for per_decoder in collected.values():
            for original, reconstruction in per_decoder.values():
                assert original.shape[0] == 5
                assert reconstruction.shape[0] == 5

    def test_explicit_subsets_override_the_default(self) -> None:
        model = _buildThreeModalityModel()
        collected = collectCrossModalReconstructions(
            model, [_fixedBatch()], input_subsets=[{"a", "b"}], device="cpu"
        )
        assert set(collected) == {frozenset({"a", "b"})}

    def test_empty_subset_raises(self) -> None:
        model = _buildTwoModalityModel()
        with pytest.raises(ValueError, match="empty subset"):
            collectCrossModalReconstructions(
                model, [_fixedBatch()], input_subsets=[set()], device="cpu"
            )

    def test_unknown_modality_name_in_subset_raises(self) -> None:
        model = _buildTwoModalityModel()
        with pytest.raises(ValueError, match="does_not_exist"):
            collectCrossModalReconstructions(
                model, [_fixedBatch()], input_subsets=[{"does_not_exist"}], device="cpu"
            )

    def test_empty_dataloader_raises(self) -> None:
        model = _buildTwoModalityModel()
        with pytest.raises(ValueError, match="empty dataloader"):
            collectCrossModalReconstructions(model, [], device="cpu")

    def test_no_input_subsets_at_all_raises(self) -> None:
        model = _buildTwoModalityModel()
        with pytest.raises(ValueError, match="input_subsets"):
            collectCrossModalReconstructions(model, [_fixedBatch()], input_subsets=[], device="cpu")


class TestPlotCrossModalReconstructionMatrix:
    def test_returns_a_figure_with_expected_grid_shape(self) -> None:
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(model, [_fixedBatch()], device="cpu")
        fig = plotCrossModalReconstructionMatrix(collected)
        assert len(fig.axes) == 3 * 2  # 3 rows ({a}, {b}, {a, b}) x 2 columns (a, b)
        plt.close(fig)

    def test_missing_pair_is_left_blank(self) -> None:
        collected = {
            frozenset({"a"}): {"a": (torch.randn(4, 8), torch.randn(4, 8))},
            frozenset({"a", "b"}): {
                "a": (torch.randn(4, 8), torch.randn(4, 8)),
                "b": (torch.randn(4, 8), torch.randn(4, 8)),
            },
        }
        fig = plotCrossModalReconstructionMatrix(collected)
        # 2 rows ({"a"} sorts first) x 2 columns; the {"a"} row has no "b" entry.
        assert len(fig.axes) == 4
        blank_axis = fig.axes[1]
        assert len(blank_axis.get_lines()) == 0
        plt.close(fig)

    def test_row_and_column_order_can_be_overridden(self) -> None:
        collected = {
            frozenset({"a"}): {"a": (torch.randn(4, 8), torch.randn(4, 8))},
            frozenset({"b"}): {"b": (torch.randn(4, 8), torch.randn(4, 8))},
        }
        fig = plotCrossModalReconstructionMatrix(
            collected,
            row_order=[frozenset({"b"}), frozenset({"a"})],
            column_order=["b", "a"],
        )
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_per_decoder_inverse_transform_is_applied(self) -> None:
        original = torch.tensor([[0.0, 1.0, 2.0]])
        reconstruction = torch.tensor([[0.5, 1.5, 2.5]])
        collected = {frozenset({"a"}): {"a": (original, reconstruction)}}
        fig = plotCrossModalReconstructionMatrix(
            collected, inverse_transform={"a": lambda x: x * 2.0}
        )
        line = fig.axes[0].lines[0]
        assert torch.allclose(torch.tensor(line.get_ydata()), original[0] * 2.0)
        plt.close(fig)

    def test_example_index_selects_the_requested_sample(self) -> None:
        originals = torch.stack([torch.zeros(5), torch.ones(5)])
        reconstructions = torch.stack([torch.full((5,), 2.0), torch.full((5,), 3.0)])
        collected = {frozenset({"a"}): {"a": (originals, reconstructions)}}
        fig = plotCrossModalReconstructionMatrix(collected, example_index=1)
        line = fig.axes[0].lines[0]
        assert torch.allclose(torch.tensor(line.get_ydata()), originals[1])
        plt.close(fig)

    def test_out_of_range_example_index_raises(self) -> None:
        collected = {frozenset({"a"}): {"a": (torch.randn(3, 5), torch.randn(3, 5))}}
        with pytest.raises(ValueError, match="example_index"):
            plotCrossModalReconstructionMatrix(collected, example_index=5)

    def test_empty_collected_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotCrossModalReconstructionMatrix({})

    def test_end_to_end_with_a_real_two_modality_model(self) -> None:
        model = _buildTwoModalityModel()
        collected = collectCrossModalReconstructions(model, [_fixedBatch()], device="cpu")
        fig = plotCrossModalReconstructionMatrix(collected, title="test matrix")
        assert fig is not None
        plt.close(fig)

    def test_end_to_end_with_a_real_three_modality_model(self) -> None:
        """Nothing here is hardcoded to exactly two modalities."""
        model = _buildThreeModalityModel()
        collected = collectCrossModalReconstructions(model, [_fixedBatch()], device="cpu")
        fig = plotCrossModalReconstructionMatrix(collected)
        assert len(fig.axes) == 4 * 3  # 4 rows ({a},{b},{c},{a,b,c}) x 3 columns
        plt.close(fig)
