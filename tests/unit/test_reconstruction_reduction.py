"""Unit tests for the `reduction` of `losses.reconstruction.computeTotalReconstructionLoss`.

Roadmap P2-1, `docs/adr/0029-loss-scale-convention.md`. `"mean"` is the default and must leave
the loss exactly as it was before the argument existed; `"sum"` is the per-sample sum over the
values of a modality, averaged over the batch. Every expected value below is written out by hand
from that definition, never from the function under test.
"""

from types import MappingProxyType

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction import (
    DEFAULT_REDUCTION,
    SUPPORTED_REDUCTIONS,
    computeTotalReconstructionLoss,
    listSupportedReductions,
    resolveReduction,
    validateReduction,
)
from global_vae.utils.autograd import backward


def _perSampleSum(per_element: torch.Tensor) -> torch.Tensor:
    """Sum every axis after the batch axis, then average over the batch."""
    return per_element.flatten(start_dim=1).sum(dim=1).mean()


class TestDefaultIsUnchanged:
    def test_default_reduction_is_mean(self) -> None:
        assert DEFAULT_REDUCTION == "mean"

    def test_no_argument_equals_the_loss_function_bit_for_bit(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss({"signal": reconstruction}, {"signal": target})
        assert torch.equal(loss, F.mse_loss(reconstruction, target))

    def test_explicit_mean_equals_no_argument_bit_for_bit(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        implicit = computeTotalReconstructionLoss({"signal": reconstruction}, {"signal": target})
        explicit = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, reduction="mean"
        )
        assert torch.equal(implicit, explicit)

    def test_weighted_default_equals_weight_times_the_loss_function(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, weights=2.5
        )
        assert torch.equal(loss, 2.5 * F.mse_loss(reconstruction, target))


class TestSum:
    def test_sum_is_the_per_sample_sum_of_squares_averaged_over_the_batch(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, reduction="sum"
        )
        expected = _perSampleSum((reconstruction - target) ** 2)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_sum_is_n_times_the_mean(self) -> None:
        reconstruction, target = torch.randn(5, 7), torch.randn(5, 7)
        mean = computeTotalReconstructionLoss({"signal": reconstruction}, {"signal": target})
        total = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, reduction="sum"
        )
        assert torch.allclose(total, 7 * mean, atol=1e-5)

    def test_n_is_the_product_of_every_axis_after_the_batch_axis(self) -> None:
        """A `(B, C, H, W)` image reconstructs `C * H * W` values per sample."""
        reconstruction, target = torch.randn(3, 2, 4, 5), torch.randn(3, 2, 4, 5)
        loss = computeTotalReconstructionLoss(
            {"image": reconstruction}, {"image": target}, reduction="sum"
        )
        expected = _perSampleSum((reconstruction - target) ** 2)
        assert torch.allclose(loss, expected, atol=1e-4)
        mean = F.mse_loss(reconstruction, target)
        assert torch.allclose(loss, 2 * 4 * 5 * mean, atol=1e-4)

    def test_a_vector_per_sample_has_one_value_so_sum_equals_mean(self) -> None:
        reconstruction, target = torch.randn(6), torch.randn(6)
        mean = computeTotalReconstructionLoss({"score": reconstruction}, {"score": target})
        total = computeTotalReconstructionLoss(
            {"score": reconstruction}, {"score": target}, reduction="sum"
        )
        assert torch.allclose(total, mean, atol=1e-6)

    def test_sum_with_l1_is_the_per_sample_sum_of_absolute_errors(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, loss_fn=F.l1_loss, reduction="sum"
        )
        assert torch.allclose(loss, _perSampleSum((reconstruction - target).abs()), atol=1e-5)

    def test_sum_with_bce_is_the_bernoulli_negative_log_likelihood_per_sample(self) -> None:
        probabilities = torch.rand(4, 10).clamp(0.05, 0.95)
        target = torch.randint(0, 2, (4, 10)).float()
        loss = computeTotalReconstructionLoss(
            {"mask": probabilities},
            {"mask": target},
            loss_fn=F.binary_cross_entropy,
            reduction="sum",
        )
        negative_log_likelihood = -(
            target * probabilities.log() + (1 - target) * (1 - probabilities).log()
        )
        assert torch.allclose(loss, _perSampleSum(negative_log_likelihood), atol=1e-5)

    def test_sum_with_smooth_l1(self) -> None:
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction},
            {"signal": target},
            loss_fn=F.smooth_l1_loss,
            reduction="sum",
        )
        per_element = F.smooth_l1_loss(reconstruction, target, reduction="none")
        assert torch.allclose(loss, _perSampleSum(per_element), atol=1e-5)

    def test_a_custom_element_wise_mean_loss_is_scaled_too(self) -> None:
        def meanCubedError(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (reconstruction - target).abs().pow(3).mean()

        reconstruction, target = torch.randn(4, 6), torch.randn(4, 6)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction},
            {"signal": target},
            loss_fn=meanCubedError,
            reduction="sum",
        )
        assert torch.allclose(
            loss, _perSampleSum((reconstruction - target).abs().pow(3)), atol=1e-4
        )

    def test_the_weight_is_applied_after_the_reduction(self) -> None:
        """`sum` with a weight of 0.5 is the unit-variance Gaussian negative log-likelihood."""
        reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
        loss = computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, weights=0.5, reduction="sum"
        )
        expected = 0.5 * _perSampleSum((reconstruction - target) ** 2)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_the_gradient_is_n_times_the_gradient_of_the_mean(self) -> None:
        target = torch.randn(4, 10)
        base = torch.randn(4, 10)

        mean_input = base.clone().requires_grad_(True)
        backward(computeTotalReconstructionLoss({"signal": mean_input}, {"signal": target}))
        sum_input = base.clone().requires_grad_(True)
        backward(
            computeTotalReconstructionLoss(
                {"signal": sum_input}, {"signal": target}, reduction="sum"
            )
        )

        assert mean_input.grad is not None and sum_input.grad is not None
        assert torch.allclose(sum_input.grad, 10 * mean_input.grad, atol=1e-6)


class TestPerModality:
    def setup_method(self) -> None:
        self.signal_reconstruction, self.signal_target = torch.randn(4, 10), torch.randn(4, 10)
        self.image_reconstruction, self.image_target = torch.randn(4, 3, 8), torch.randn(4, 3, 8)
        self.reconstructions = {
            "signal": self.signal_reconstruction,
            "image": self.image_reconstruction,
        }
        self.targets = {"signal": self.signal_target, "image": self.image_target}

    def test_one_name_applies_to_every_modality(self) -> None:
        loss = computeTotalReconstructionLoss(self.reconstructions, self.targets, reduction="sum")
        expected = _perSampleSum((self.signal_reconstruction - self.signal_target) ** 2) + (
            _perSampleSum((self.image_reconstruction - self.image_target) ** 2)
        )
        assert torch.allclose(loss, expected, atol=1e-4)

    def test_a_mapping_mixes_reductions(self) -> None:
        loss = computeTotalReconstructionLoss(
            self.reconstructions, self.targets, reduction={"signal": "sum", "image": "mean"}
        )
        expected = _perSampleSum(
            (self.signal_reconstruction - self.signal_target) ** 2
        ) + F.mse_loss(self.image_reconstruction, self.image_target)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_a_modality_left_out_of_the_mapping_keeps_mean(self) -> None:
        loss = computeTotalReconstructionLoss(
            self.reconstructions, self.targets, reduction={"signal": "sum"}
        )
        expected = _perSampleSum(
            (self.signal_reconstruction - self.signal_target) ** 2
        ) + F.mse_loss(self.image_reconstruction, self.image_target)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_an_empty_mapping_is_the_default_for_every_modality(self) -> None:
        loss = computeTotalReconstructionLoss(self.reconstructions, self.targets, reduction={})
        default = computeTotalReconstructionLoss(self.reconstructions, self.targets)
        assert torch.equal(loss, default)

    def test_a_mapping_may_name_a_modality_absent_from_this_pass(self) -> None:
        """Only a subset of the decoders reconstructs when a modality is dropped (spec §5)."""
        loss = computeTotalReconstructionLoss(
            {"signal": self.signal_reconstruction},
            self.targets,
            reduction={"signal": "sum", "image": "sum"},
        )
        expected = _perSampleSum((self.signal_reconstruction - self.signal_target) ** 2)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_any_read_only_mapping_is_accepted(self) -> None:
        loss = computeTotalReconstructionLoss(
            self.reconstructions,
            self.targets,
            reduction=MappingProxyType({"signal": "sum"}),
        )
        mixed = computeTotalReconstructionLoss(
            self.reconstructions, self.targets, reduction={"signal": "sum"}
        )
        assert torch.equal(loss, mixed)

    def test_it_combines_with_a_per_modality_loss_function_and_weight(self) -> None:
        mask_probabilities = torch.rand(4, 5).clamp(0.05, 0.95)
        mask_target = torch.randint(0, 2, (4, 5)).float()
        loss = computeTotalReconstructionLoss(
            {"signal": self.signal_reconstruction, "mask": mask_probabilities},
            {"signal": self.signal_target, "mask": mask_target},
            weights={"signal": 0.5, "mask": 2.0},
            loss_fn={"signal": F.mse_loss, "mask": F.binary_cross_entropy},
            reduction={"signal": "sum"},
        )
        expected = 0.5 * _perSampleSum(
            (self.signal_reconstruction - self.signal_target) ** 2
        ) + 2.0 * F.binary_cross_entropy(mask_probabilities, mask_target)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_the_shape_guard_still_applies_under_sum(self) -> None:
        with pytest.raises(ValueError, match="'image'"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(4, 8, 8)},
                {"image": torch.randn(4, 1, 8, 8)},
                reduction="sum",
            )


class TestValidation:
    def test_unknown_name_is_rejected_and_the_available_ones_are_listed(self) -> None:
        with pytest.raises(ValueError, match="'average'.*Available: mean, sum"):
            computeTotalReconstructionLoss(
                {"signal": torch.randn(2, 3)}, {"signal": torch.randn(2, 3)}, reduction="average"
            )

    def test_unknown_name_inside_a_mapping_names_the_modality(self) -> None:
        with pytest.raises(ValueError, match=r"reduction\['image'\]: 'avg'"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(2, 3)},
                {"image": torch.randn(2, 3)},
                reduction={"signal": "sum", "image": "avg"},
            )

    def test_a_non_string_value_inside_a_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"reduction\['signal'\]: 3"):
            validateReduction({"signal": 3})

    def test_a_non_string_modality_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="modality name must be a string"):
            validateReduction({1: "sum"})

    @pytest.mark.parametrize("value", [None, 3, 1.5, True, ["sum"], ("mean", "sum")])
    def test_a_value_that_is_neither_a_name_nor_a_mapping_is_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match="expected one of mean, sum"):
            validateReduction(value)

    def test_the_name_is_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="'Sum'"):
            validateReduction("Sum")

    def test_the_message_starts_with_the_field_name_the_caller_gave(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validateReduction("nope", "training.reconstruction_reduction")
        assert str(excinfo.value).startswith("training.reconstruction_reduction:")

    def test_the_default_field_name_is_reduction(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validateReduction("nope")
        assert str(excinfo.value).startswith("reduction:")

    @pytest.mark.parametrize("value", ["mean", "sum", {}, {"a": "mean"}, {"a": "sum", "b": "mean"}])
    def test_valid_values_pass(self, value: object) -> None:
        validateReduction(value)

    def test_an_invalid_reduction_is_rejected_before_any_loss_is_computed(self) -> None:
        calls: list[int] = []

        def recordingLoss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return F.mse_loss(reconstruction, target)

        with pytest.raises(ValueError, match="Available"):
            computeTotalReconstructionLoss(
                {"signal": torch.randn(2, 3)},
                {"signal": torch.randn(2, 3)},
                loss_fn=recordingLoss,
                reduction="nope",
            )
        assert calls == []


class TestHelpers:
    def test_supported_reductions(self) -> None:
        assert listSupportedReductions() == ["mean", "sum"]
        assert SUPPORTED_REDUCTIONS == ("mean", "sum")

    def test_the_returned_list_is_a_copy(self) -> None:
        listSupportedReductions().append("other")
        assert listSupportedReductions() == ["mean", "sum"]

    def test_resolve_a_single_name_returns_it_for_any_modality(self) -> None:
        assert resolveReduction("sum", "signal") == "sum"
        assert resolveReduction("mean", "image") == "mean"

    def test_resolve_a_mapping_returns_the_entry_or_the_default(self) -> None:
        mapping = {"signal": "sum"}
        assert resolveReduction(mapping, "signal") == "sum"
        assert resolveReduction(mapping, "image") == DEFAULT_REDUCTION
