"""Unit tests for the reconstruction loss registry and its losses (roadmap P5-3, ADR 0048)."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from global_vae.losses.reconstruction import (
    AbstractReconstructionLoss,
    collectReconstructionLossModules,
    computeTotalReconstructionLoss,
    getReconstructionLossClass,
    listRegisteredReconstructionLosses,
)
from global_vae.losses.reconstruction.bce import BceLoss
from global_vae.losses.reconstruction.bce_with_logits import BceWithLogitsLoss
from global_vae.losses.reconstruction.chamfer import ChamferLoss
from global_vae.losses.reconstruction.l1 import L1Loss
from global_vae.losses.reconstruction.masked_mse import MaskedMseLoss
from global_vae.losses.reconstruction.mse import MseLoss
from global_vae.losses.reconstruction.smooth_l1 import SmoothL1Loss
from global_vae.losses.reconstruction.tabular_mixed import TabularMixedLoss
from global_vae.losses.reconstruction.token_cross_entropy import TokenCrossEntropyLoss
from global_vae.utils.autograd import backward
from global_vae.utils.point_cloud_metrics import computeChamferDistance
from global_vae.utils.series_blocks import computeMaskedSquaredError
from global_vae.utils.text_blocks import computeTokenCrossEntropy

NAN = float("nan")
BATCH = 5

# The four losses of 2.0.0 and the function each one was.
CLASSIC: dict[str, tuple[Callable[[], Any], Callable[..., torch.Tensor]]] = {
    "mse": (MseLoss, functional.mse_loss),
    "l1": (L1Loss, functional.l1_loss),
    "bce": (BceLoss, functional.binary_cross_entropy),
    "smooth_l1": (SmoothL1Loss, functional.smooth_l1_loss),
}


def _pair(*shape: int, unit: bool = False, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A reconstruction and a target, in `[0.05, 0.95]` when `unit`, else Gaussian."""
    generator = torch.Generator().manual_seed(seed)
    if unit:
        low, high = 0.05, 0.95
        return (
            torch.rand(*shape, generator=generator) * (high - low) + low,
            torch.rand(*shape, generator=generator),
        )
    return torch.randn(*shape, generator=generator), torch.randn(*shape, generator=generator)


BUILT_IN = (
    "bce",
    "bce_with_logits",
    "chamfer",
    "l1",
    "masked_mse",
    "mse",
    "smooth_l1",
    "tabular_mixed",
    "token_cross_entropy",
)


class TestTheBaseClass:
    def test_the_abstract_forward_says_it_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            AbstractReconstructionLoss.forward(MseLoss(), torch.zeros(2, 3), torch.zeros(2, 3))


class TestTheRegistry:
    def test_the_built_in_losses_are_registered(self) -> None:
        """A subset, since a test module of a later phase may register a loss of its own."""
        listed = listRegisteredReconstructionLosses()
        assert listed == sorted(listed)
        assert set(BUILT_IN) <= set(listed)

    @pytest.mark.parametrize("name", BUILT_IN)
    def test_every_loss_is_a_module_of_the_abstract_class(self, name: str) -> None:
        assert issubclass(getReconstructionLossClass(name), AbstractReconstructionLoss)
        assert issubclass(getReconstructionLossClass(name), torch.nn.Module)

    def test_the_declarations_are_read_from_the_class(self) -> None:
        """No loss has to be built to ask what it assumes."""
        flags = {
            name: (
                getReconstructionLossClass(name).requires_matching_shapes,
                getReconstructionLossClass(name).reads_nan_as_unobserved,
            )
            for name in BUILT_IN
        }
        assert flags == {
            "mse": (True, False),
            "l1": (True, False),
            "bce": (True, False),
            "smooth_l1": (True, False),
            "bce_with_logits": (True, False),
            "masked_mse": (True, True),
            "token_cross_entropy": (False, False),
            "chamfer": (False, True),
            "tabular_mixed": (True, False),
        }

    def test_the_default_metrics_a_loss_declares(self) -> None:
        assert ChamferLoss.default_metrics == ("chamfer", "precision", "recall", "f_score")
        assert TokenCrossEntropyLoss.default_metrics == ("token_accuracy",)
        assert MseLoss.default_metrics is None


class TestTheFourClassicLossesAreUnchanged:
    """The numbers of 2.0.0: each name is its `torch.nn.functional` function, bit for bit."""

    @pytest.mark.parametrize("name", CLASSIC)
    def test_the_value_is_the_functions(self, name: str) -> None:
        loss_class, function = CLASSIC[name]
        reconstruction, target = _pair(BATCH, 4, 3, unit=True)
        assert torch.equal(loss_class()(reconstruction, target), function(reconstruction, target))

    @pytest.mark.parametrize("name", CLASSIC)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_the_aggregate_is_what_2_0_0_computed(self, name: str, reduction: str) -> None:
        """`computeTotalReconstructionLoss` with the class equals the function with the old rule."""
        loss_class, function = CLASSIC[name]
        reconstruction, target = _pair(BATCH, 4, 3, unit=True)
        by_class = computeTotalReconstructionLoss(
            {"x": reconstruction}, {"x": target}, loss_fn=loss_class(), reduction=reduction
        )
        by_function = computeTotalReconstructionLoss(
            {"x": reconstruction}, {"x": target}, loss_fn=function, reduction=reduction
        )
        scale = math.prod(reconstruction.shape[1:]) if reduction == "sum" else 1
        assert torch.equal(by_class, by_function)
        assert torch.equal(by_class, function(reconstruction, target) * scale)

    @pytest.mark.parametrize("name", CLASSIC)
    def test_the_gradient_is_the_functions(self, name: str) -> None:
        loss_class, function = CLASSIC[name]
        reconstruction, target = _pair(BATCH, 6, unit=True)
        first = reconstruction.clone().requires_grad_()
        second = reconstruction.clone().requires_grad_()
        backward(loss_class()(first, target))
        backward(function(second, target))
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, second.grad)

    def test_smooth_l1_takes_beta_and_the_default_is_the_functions(self) -> None:
        reconstruction, target = _pair(BATCH, 6)
        assert torch.equal(
            SmoothL1Loss(beta=0.25)(reconstruction, target),
            functional.smooth_l1_loss(reconstruction, target, beta=0.25),
        )
        assert not torch.equal(
            SmoothL1Loss(beta=0.25)(reconstruction, target), SmoothL1Loss()(reconstruction, target)
        )

    def test_a_negative_beta_is_refused(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            SmoothL1Loss(beta=-1.0)

    def test_every_classic_loss_counts_the_values_of_a_sample(self) -> None:
        reconstruction, target = _pair(BATCH, 4, 3)
        for loss_class, _ in CLASSIC.values():
            assert loss_class().countTerms(reconstruction, target) == 12


class TestBceWithLogits:
    def test_it_is_the_function_on_raw_scores(self) -> None:
        scores, _ = _pair(BATCH, 7)
        target = (torch.rand(BATCH, 7) > 0.5).float()
        assert torch.equal(
            BceWithLogitsLoss()(scores, target),
            functional.binary_cross_entropy_with_logits(scores, target),
        )

    def test_it_equals_bce_on_the_sigmoid_of_the_scores(self) -> None:
        scores, _ = _pair(BATCH, 7)
        target = (torch.rand(BATCH, 7) > 0.5).float()
        assert torch.allclose(
            BceWithLogitsLoss()(scores, target),
            BceLoss()(torch.sigmoid(scores), target),
            atol=1e-6,
        )

    def test_it_stays_finite_where_a_sigmoid_then_bce_saturates(self) -> None:
        scores = torch.tensor([[200.0, -200.0]])
        target = torch.tensor([[0.0, 1.0]])
        value = BceWithLogitsLoss()(scores, target)
        assert torch.isfinite(value) and value.item() == pytest.approx(200.0)


class TestMaskedMse:
    def _series(self) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction, target = _pair(BATCH, 3, 10)
        target = target.clone()
        target[0, :, 2:5] = NAN
        target[3, 1, :] = NAN
        return reconstruction, target

    def test_it_equals_mse_when_nothing_is_missing(self) -> None:
        reconstruction, target = _pair(BATCH, 3, 10)
        assert torch.allclose(
            MaskedMseLoss()(reconstruction, target),
            functional.mse_loss(reconstruction, target),
            rtol=1e-6,
        )

    def test_with_sum_it_is_the_masked_squared_error_of_the_helper(self) -> None:
        reconstruction, target = self._series()
        total = computeTotalReconstructionLoss(
            {"s": reconstruction}, {"s": target}, loss_fn=MaskedMseLoss(), reduction="sum"
        )
        assert torch.allclose(total, computeMaskedSquaredError(reconstruction, target), rtol=1e-6)

    def test_the_value_is_finite_with_gaps_and_mse_is_not(self) -> None:
        reconstruction, target = self._series()
        assert torch.isfinite(MaskedMseLoss()(reconstruction, target))
        assert torch.isnan(functional.mse_loss(reconstruction, target))

    def test_an_unobserved_value_adds_nothing_and_gets_no_gradient(self) -> None:
        reconstruction, target = self._series()
        reconstruction = reconstruction.clone().requires_grad_()
        backward(MaskedMseLoss()(reconstruction, target))
        assert reconstruction.grad is not None
        assert torch.isfinite(reconstruction.grad).all()
        assert (reconstruction.grad[torch.isnan(target)] == 0).all()
        assert (reconstruction.grad[~torch.isnan(target)] != 0).any()

    def test_it_counts_the_values_of_a_sample_so_that_sum_is_exact(self) -> None:
        reconstruction, target = self._series()
        assert MaskedMseLoss().countTerms(reconstruction, target) == 30

    def test_it_reads_nan_as_unobserved(self) -> None:
        assert MaskedMseLoss.reads_nan_as_unobserved is True


class TestTokenCrossEntropy:
    VOCAB = 11
    LENGTH = 6

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(3)
        scores = torch.randn(BATCH, self.LENGTH, self.VOCAB, generator=generator)
        ids = torch.randint(0, self.VOCAB, (BATCH, self.LENGTH), generator=generator)
        return scores, ids

    def test_ids_and_the_one_hot_tensor_give_the_same_value(self) -> None:
        scores, ids = self._batch()
        one_hot = functional.one_hot(ids, self.VOCAB).float()
        loss = TokenCrossEntropyLoss()
        assert torch.equal(loss(scores, ids), loss(scores, one_hot))

    def test_it_is_the_helper_divided_by_the_tokens(self) -> None:
        scores, ids = self._batch()
        one_hot = functional.one_hot(ids, self.VOCAB).float()
        expected = computeTokenCrossEntropy(scores, one_hot) / self.LENGTH
        assert torch.allclose(TokenCrossEntropyLoss()(scores, ids), expected, rtol=1e-6)

    def test_it_is_torchs_cross_entropy_per_token(self) -> None:
        scores, ids = self._batch()
        expected = functional.cross_entropy(scores.transpose(1, 2), ids)
        assert torch.allclose(TokenCrossEntropyLoss()(scores, ids), expected, rtol=1e-5)

    def test_with_sum_it_is_the_cross_entropy_of_the_text(self) -> None:
        scores, ids = self._batch()
        one_hot = functional.one_hot(ids, self.VOCAB).float()
        total = computeTotalReconstructionLoss(
            {"t": scores}, {"t": ids}, loss_fn=TokenCrossEntropyLoss(), reduction="sum"
        )
        assert torch.allclose(total, computeTokenCrossEntropy(scores, one_hot), rtol=1e-5)

    def test_the_terms_are_the_tokens_and_not_the_tokens_times_the_vocabulary(self) -> None:
        scores, ids = self._batch()
        assert TokenCrossEntropyLoss().countTerms(scores, ids) == self.LENGTH

    def test_ids_are_accepted_by_the_aggregate_with_no_shape_check(self) -> None:
        scores, ids = self._batch()
        assert scores.shape != ids.shape
        computeTotalReconstructionLoss({"t": scores}, {"t": ids}, loss_fn=TokenCrossEntropyLoss())

    def test_a_perfect_decoder_scores_nearly_zero(self) -> None:
        _, ids = self._batch()
        scores = functional.one_hot(ids, self.VOCAB).float() * 50.0
        assert TokenCrossEntropyLoss()(scores, ids).item() == pytest.approx(0.0, abs=1e-4)

    def test_the_gradient_reaches_the_scores(self) -> None:
        scores, ids = self._batch()
        scores = scores.requires_grad_()
        backward(TokenCrossEntropyLoss()(scores, ids))
        assert scores.grad is not None and scores.grad.abs().sum() > 0

    def test_an_id_outside_the_vocabulary_is_refused(self) -> None:
        scores, ids = self._batch()
        ids[0, 0] = self.VOCAB
        with pytest.raises(ValueError, match=rf"\[0, {self.VOCAB}\)"):
            TokenCrossEntropyLoss()(scores, ids)

    def test_ids_of_the_wrong_shape_or_dtype_are_refused(self) -> None:
        scores, ids = self._batch()
        with pytest.raises(ValueError, match="int32 or int64"):
            TokenCrossEntropyLoss()(scores, ids[:, :-1])
        with pytest.raises(ValueError, match="int32 or int64"):
            TokenCrossEntropyLoss()(scores, ids.to(torch.int16))

    def test_a_one_hot_target_of_another_shape_is_refused_and_says_what_to_pass(self) -> None:
        scores, _ = self._batch()
        with pytest.raises(ValueError, match="token ids"):
            TokenCrossEntropyLoss()(scores, torch.zeros(BATCH, self.LENGTH, self.VOCAB + 1))

    def test_scores_that_are_not_three_dimensional_are_refused(self) -> None:
        with pytest.raises(ValueError, match="batch, length, vocabulary"):
            TokenCrossEntropyLoss()(
                torch.zeros(BATCH, self.VOCAB), torch.zeros(BATCH, dtype=torch.long)
            )


class TestChamfer:
    def _clouds(self, reconstruction_points: int, target_points: int) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(5)
        return (
            torch.randn(BATCH, reconstruction_points, 3, generator=generator),
            torch.randn(BATCH, target_points, 3, generator=generator),
        )

    def test_it_is_the_chamfer_distance_unchanged(self) -> None:
        reconstruction, target = self._clouds(12, 12)
        assert torch.equal(
            ChamferLoss()(reconstruction, target), computeChamferDistance(reconstruction, target)
        )

    def test_the_two_clouds_may_have_different_numbers_of_points(self) -> None:
        reconstruction, target = self._clouds(12, 9)
        assert torch.isfinite(ChamferLoss()(reconstruction, target))
        computeTotalReconstructionLoss({"c": reconstruction}, {"c": target}, loss_fn=ChamferLoss())

    def test_a_plain_callable_still_needs_equal_shapes(self) -> None:
        reconstruction, target = self._clouds(12, 9)
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"c": reconstruction}, {"c": target}, loss_fn=computeChamferDistance
            )

    def test_a_row_of_nan_is_a_point_that_does_not_exist(self) -> None:
        reconstruction, target = self._clouds(12, 9)
        padded = torch.cat([target, torch.full((BATCH, 3, 3), NAN)], dim=1)
        assert torch.allclose(
            ChamferLoss()(reconstruction, padded), ChamferLoss()(reconstruction, target)
        )

    def test_the_order_of_the_points_does_not_matter(self) -> None:
        _, cloud = self._clouds(10, 10)
        shuffled = cloud[:, torch.randperm(10, generator=torch.Generator().manual_seed(1))]
        assert ChamferLoss()(shuffled, cloud).item() == pytest.approx(0.0, abs=1e-5)

    def test_mean_and_sum_give_the_same_number_since_it_is_one_term(self) -> None:
        reconstruction, target = self._clouds(12, 9)
        values = [
            computeTotalReconstructionLoss(
                {"c": reconstruction}, {"c": target}, loss_fn=ChamferLoss(), reduction=reduction
            )
            for reduction in ("mean", "sum")
        ]
        assert torch.equal(values[0], values[1])
        assert ChamferLoss().countTerms(reconstruction, target) == 1


class TestTabularMixed:
    NUMERIC = 4
    SIZES = (3, 2)

    def _rows(self, probabilities: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(7)
        features = self.NUMERIC + sum(self.SIZES)
        reconstruction = torch.randn(BATCH, features, generator=generator)
        target = torch.randn(BATCH, features, generator=generator)
        start = self.NUMERIC
        for size in self.SIZES:
            block = reconstruction[:, start : start + size]
            if probabilities:
                reconstruction[:, start : start + size] = torch.softmax(block, dim=1)
            choice = torch.randint(0, size, (BATCH,), generator=generator)
            target[:, start : start + size] = functional.one_hot(choice, size).float()
            start += size
        return reconstruction, target

    def _byHand(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The definition, written column by column."""
        squared = ((reconstruction[:, : self.NUMERIC] - target[:, : self.NUMERIC]) ** 2).sum(dim=1)
        entropy = torch.zeros(BATCH)
        start = self.NUMERIC
        for size in self.SIZES:
            for row in range(BATCH):
                category = int(target[row, start : start + size].argmax())
                entropy[row] -= reconstruction[row, start + category].log()
            start += size
        return ((squared + entropy) / (self.NUMERIC + len(self.SIZES))).mean()

    def test_it_matches_the_definition_on_probabilities(self) -> None:
        reconstruction, target = self._rows()
        loss = TabularMixedLoss(categorical_sizes=self.SIZES)
        assert torch.allclose(loss(reconstruction, target), self._byHand(reconstruction, target))

    def test_with_sum_it_is_the_squared_error_plus_the_negative_log_likelihood(self) -> None:
        reconstruction, target = self._rows()
        total = computeTotalReconstructionLoss(
            {"t": reconstruction},
            {"t": target},
            loss_fn=TabularMixedLoss(categorical_sizes=self.SIZES),
            reduction="sum",
        )
        terms = self.NUMERIC + len(self.SIZES)
        assert torch.allclose(total, self._byHand(reconstruction, target) * terms)

    def test_the_terms_are_the_numeric_columns_plus_the_blocks(self) -> None:
        reconstruction, target = self._rows()
        assert TabularMixedLoss(categorical_sizes=self.SIZES).countTerms(
            reconstruction, target
        ) == (self.NUMERIC + 2)

    def test_raw_scores_are_softmaxed_inside_the_loss(self) -> None:
        scores, target = self._rows(probabilities=False)
        from_scores = TabularMixedLoss(categorical_sizes=self.SIZES, probabilities=False)(
            scores, target
        )
        probabilities = scores.clone()
        start = self.NUMERIC
        for size in self.SIZES:
            probabilities[:, start : start + size] = torch.softmax(
                scores[:, start : start + size], dim=1
            )
            start += size
        assert torch.allclose(
            from_scores,
            TabularMixedLoss(categorical_sizes=self.SIZES)(probabilities, target),
            atol=1e-6,
        )

    def test_with_no_block_it_is_the_mse_of_the_table(self) -> None:
        reconstruction, target = _pair(BATCH, 9)
        assert torch.allclose(
            TabularMixedLoss()(reconstruction, target),
            functional.mse_loss(reconstruction, target),
            rtol=1e-6,
        )

    def test_a_block_that_puts_no_mass_on_the_category_is_large_and_finite(self) -> None:
        reconstruction, target = self._rows()
        reconstruction[:, self.NUMERIC : self.NUMERIC + 3] = torch.tensor([1.0, 0.0, 0.0])
        target[:, self.NUMERIC : self.NUMERIC + 3] = torch.tensor([0.0, 1.0, 0.0])
        value = TabularMixedLoss(categorical_sizes=self.SIZES)(reconstruction, target)
        assert torch.isfinite(value) and value.item() > 3.0

    def test_the_gradient_reaches_every_column(self) -> None:
        reconstruction, target = self._rows()
        reconstruction = reconstruction.requires_grad_()
        backward(TabularMixedLoss(categorical_sizes=self.SIZES)(reconstruction, target))
        assert reconstruction.grad is not None
        assert (reconstruction.grad.abs().sum(dim=0) > 0).all()

    def test_blocks_wider_than_the_row_are_refused(self) -> None:
        reconstruction, target = _pair(BATCH, 4)
        with pytest.raises(ValueError, match="more than the 4 columns"):
            TabularMixedLoss(categorical_sizes=(3, 2))(reconstruction, target)

    def test_a_block_of_one_category_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            TabularMixedLoss(categorical_sizes=(1,))

    def test_it_reads_categorical_sizes_from_the_decoder(self) -> None:
        assert TabularMixedLoss.decoder_arguments == ("categorical_sizes",)

    def test_shapes_that_are_not_rows_are_refused(self) -> None:
        with pytest.raises(ValueError, match="num_features"):
            TabularMixedLoss()(torch.zeros(BATCH, 3, 2), torch.zeros(BATCH, 3, 2))


class TestTheAggregateReadsTheDeclarations:
    def test_a_loss_that_requires_matching_shapes_is_checked_with_the_old_message(self) -> None:
        reconstruction, _ = _pair(BATCH, 4)
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"x": reconstruction}, {"x": torch.zeros(BATCH, 5)}, loss_fn=MseLoss()
            )

    def test_a_plain_callable_keeps_the_old_term_count(self) -> None:
        reconstruction, target = _pair(BATCH, 4, 3)
        total = computeTotalReconstructionLoss(
            {"x": reconstruction}, {"x": target}, loss_fn=functional.mse_loss, reduction="sum"
        )
        assert torch.equal(total, functional.mse_loss(reconstruction, target) * 12)

    def test_a_mapping_can_mix_registered_losses_and_plain_functions(self) -> None:
        x, tx = _pair(BATCH, 4)
        y, ty = _pair(BATCH, 6, seed=2)
        total = computeTotalReconstructionLoss(
            {"x": x, "y": y},
            {"x": tx, "y": ty},
            loss_fn={"x": L1Loss(), "y": functional.mse_loss},
        )
        assert torch.allclose(total, functional.l1_loss(x, tx) + functional.mse_loss(y, ty))

    def test_a_term_count_is_applied_per_modality(self) -> None:
        scores = torch.randn(BATCH, 6, 11)
        ids = torch.randint(0, 11, (BATCH, 6))
        x, tx = _pair(BATCH, 4)
        total = computeTotalReconstructionLoss(
            {"t": scores, "x": x},
            {"t": ids, "x": tx},
            loss_fn={"t": TokenCrossEntropyLoss(), "x": MseLoss()},
            reduction="sum",
        )
        expected = (
            computeTokenCrossEntropy(scores, functional.one_hot(ids, 11).float())
            + functional.mse_loss(x, tx) * 4
        )
        assert torch.allclose(total, expected, rtol=1e-5)


class TestCollectingModules:
    def test_modules_are_collected_once_each(self) -> None:
        shared = MseLoss()
        modules = collectReconstructionLossModules({"a": shared, "b": shared, "c": L1Loss()})
        assert list(modules) == ["a", "c"]
        assert modules["a"] is shared

    def test_a_single_loss_is_collected_as_default(self) -> None:
        loss = MseLoss()
        assert collectReconstructionLossModules(loss) == {"default": loss}

    def test_plain_functions_are_left_out(self) -> None:
        assert collectReconstructionLossModules(functional.mse_loss) == {}
        assert collectReconstructionLossModules({"a": functional.mse_loss}) == {}

    def test_a_dot_in_a_name_becomes_an_underscore(self) -> None:
        assert list(collectReconstructionLossModules({"a.b": MseLoss()})) == ["a_b"]

    def test_two_losses_that_would_share_a_module_name_are_refused(self) -> None:
        """A silent overwrite would leave one loss's parameters out of the optimizer."""
        with pytest.raises(ValueError, match="both become 'a_b'"):
            collectReconstructionLossModules({"a.b": MseLoss(), "a_b": L1Loss()})

    def test_one_module_under_two_colliding_names_is_fine(self) -> None:
        shared = MseLoss()
        assert collectReconstructionLossModules({"a.b": shared, "a_b": shared}) == {"a_b": shared}

    def test_any_module_is_collected(self) -> None:
        loss = torch.nn.MSELoss()
        assert collectReconstructionLossModules(loss) == {"default": loss}
