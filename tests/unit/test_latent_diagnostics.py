"""Unit tests for `evaluation.latent_diagnostics` and `computeKlPerDimension` (roadmap P2-1).

Every expected value is worked out by hand from the definitions: the KL of `N(mu, s^2)` to
`N(0, 1)` is `0.5 * (mu^2 + s^2 - 1 - ln s^2)`, and a dimension is active when the variance of its
posterior mean over the samples is strictly above the threshold.
"""

import json
import math
from dataclasses import asdict

import pytest
import torch

from global_vae.evaluation.latent_diagnostics import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    LatentDiagnostics,
    computeLatentDiagnostics,
    validateActiveUnitThreshold,
)
from global_vae.losses.regularizers.kl_standard_normal import (
    KlStandardNormalRegularizer,
    computeKlPerDimension,
)


class TestComputeKlPerDimension:
    def test_the_prior_has_zero_kl(self) -> None:
        kl = computeKlPerDimension(torch.zeros(3, 4), torch.zeros(3, 4))
        assert torch.equal(kl, torch.zeros(3, 4))

    def test_a_shifted_mean_costs_half_its_square(self) -> None:
        mu = torch.tensor([[2.0, 0.0]])
        kl = computeKlPerDimension(mu, torch.zeros(1, 2))
        assert kl[0, 0].item() == pytest.approx(2.0)
        assert kl[0, 1].item() == pytest.approx(0.0)

    def test_a_wider_posterior(self) -> None:
        """`s^2 = 4`: `0.5 * (4 - 1 - ln 4)`."""
        kl = computeKlPerDimension(torch.zeros(1, 1), torch.full((1, 1), math.log(4.0)))
        assert kl.item() == pytest.approx(0.5 * (3.0 - math.log(4.0)))

    def test_a_narrower_posterior(self) -> None:
        """`s^2 = 0.25`: `0.5 * (0.25 - 1 + ln 4)`."""
        kl = computeKlPerDimension(torch.zeros(1, 1), torch.full((1, 1), math.log(0.25)))
        assert kl.item() == pytest.approx(0.5 * (-0.75 + math.log(4.0)))

    def test_it_is_never_negative(self) -> None:
        torch.manual_seed(0)
        kl = computeKlPerDimension(torch.randn(50, 6), torch.randn(50, 6))
        assert bool((kl >= -1e-6).all())

    def test_the_output_has_the_shape_of_the_input(self) -> None:
        assert computeKlPerDimension(torch.zeros(5, 7), torch.zeros(5, 7)).shape == (5, 7)

    def test_the_regularizer_is_its_sum_over_the_dimensions(self) -> None:
        torch.manual_seed(1)
        mu, logvar = torch.randn(8, 5), torch.randn(8, 5)
        per_sample = KlStandardNormalRegularizer()(mu, logvar)
        assert torch.allclose(per_sample, computeKlPerDimension(mu, logvar).sum(dim=-1), atol=1e-6)

    def test_the_regularizer_still_matches_the_closed_form_it_had_before(self) -> None:
        torch.manual_seed(2)
        mu, logvar = torch.randn(8, 5), torch.randn(8, 5)
        previous_formula = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        assert torch.allclose(
            KlStandardNormalRegularizer()(mu, logvar), previous_formula, atol=1e-6
        )


class TestComputeLatentDiagnostics:
    def test_a_collapsed_space_has_no_active_unit_and_no_kl(self) -> None:
        diagnostics = computeLatentDiagnostics(torch.zeros(10, 3), torch.zeros(10, 3))
        assert diagnostics.num_dimensions == 3
        assert diagnostics.active_units == 0
        assert diagnostics.kl_per_dimension == [0.0, 0.0, 0.0]
        assert diagnostics.mu_variance_per_dimension == [0.0, 0.0, 0.0]

    def test_only_the_dimension_whose_mean_moves_is_active(self) -> None:
        mu = torch.zeros(4, 3)
        mu[:, 1] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        diagnostics = computeLatentDiagnostics(mu, torch.zeros(4, 3))
        assert diagnostics.active_units == 1
        assert diagnostics.mu_variance_per_dimension == pytest.approx([0.0, 1.0, 0.0])

    def test_the_kl_is_the_mean_over_samples_per_dimension(self) -> None:
        """Dimension 0 has `mu = +-1` (KL 0.5 each), dimension 1 sits on the prior."""
        mu = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        diagnostics = computeLatentDiagnostics(mu, torch.zeros(2, 2))
        assert diagnostics.kl_per_dimension == pytest.approx([0.5, 0.0])

    def test_a_flat_mean_with_a_wide_posterior_has_kl_but_is_not_active(self) -> None:
        """The two views disagree on purpose: KL above zero, posterior mean not moving."""
        logvar = torch.full((6, 1), math.log(4.0))
        diagnostics = computeLatentDiagnostics(torch.zeros(6, 1), logvar)
        assert diagnostics.kl_per_dimension[0] == pytest.approx(0.5 * (3.0 - math.log(4.0)))
        assert diagnostics.active_units == 0

    def test_the_variance_is_the_population_variance(self) -> None:
        """Values 0 and 2: mean 1, population variance 1 (the sample variance would be 2)."""
        mu = torch.tensor([[0.0], [2.0]])
        diagnostics = computeLatentDiagnostics(mu, torch.zeros(2, 1))
        assert diagnostics.mu_variance_per_dimension == pytest.approx([1.0])

    def test_a_single_sample_gives_zero_variance_and_no_nan(self) -> None:
        diagnostics = computeLatentDiagnostics(torch.tensor([[3.0, -2.0]]), torch.zeros(1, 2))
        assert diagnostics.mu_variance_per_dimension == [0.0, 0.0]
        assert diagnostics.active_units == 0
        assert not any(math.isnan(value) for value in diagnostics.kl_per_dimension)

    def test_the_comparison_with_the_threshold_is_strict(self) -> None:
        mu = torch.tensor([[0.0], [2.0]])  # population variance exactly 1
        at_threshold = computeLatentDiagnostics(mu, torch.zeros(2, 1), active_threshold=1.0)
        below_threshold = computeLatentDiagnostics(mu, torch.zeros(2, 1), active_threshold=0.99)
        assert at_threshold.active_units == 0
        assert below_threshold.active_units == 1

    def test_the_default_threshold_is_the_one_of_burda_et_al(self) -> None:
        assert DEFAULT_ACTIVE_UNIT_THRESHOLD == 0.01
        diagnostics = computeLatentDiagnostics(torch.zeros(3, 2), torch.zeros(3, 2))
        assert diagnostics.active_threshold == 0.01

    def test_a_zero_threshold_counts_any_dimension_that_moves(self) -> None:
        mu = torch.tensor([[0.0, 0.0], [1e-3, 0.0]])
        diagnostics = computeLatentDiagnostics(mu, torch.zeros(2, 2), active_threshold=0.0)
        assert diagnostics.active_units == 1

    def test_the_threshold_is_kept_in_the_result(self) -> None:
        diagnostics = computeLatentDiagnostics(
            torch.randn(5, 2), torch.zeros(5, 2), active_threshold=0.25
        )
        assert diagnostics.active_threshold == 0.25

    def test_one_value_per_dimension(self) -> None:
        torch.manual_seed(3)
        diagnostics = computeLatentDiagnostics(torch.randn(20, 6), torch.randn(20, 6))
        assert diagnostics.num_dimensions == 6
        assert len(diagnostics.kl_per_dimension) == 6
        assert len(diagnostics.mu_variance_per_dimension) == 6
        assert 0 <= diagnostics.active_units <= 6

    def test_the_total_kl_is_the_value_the_regularizer_reports(self) -> None:
        torch.manual_seed(4)
        mu, logvar = torch.randn(30, 5), torch.randn(30, 5)
        diagnostics = computeLatentDiagnostics(mu, logvar)
        regularizer_mean = KlStandardNormalRegularizer()(mu, logvar).mean().item()
        assert sum(diagnostics.kl_per_dimension) == pytest.approx(regularizer_mean, rel=1e-5)

    def test_the_values_are_plain_python_floats_and_ints(self) -> None:
        diagnostics = computeLatentDiagnostics(torch.randn(5, 2), torch.randn(5, 2))
        assert all(type(value) is float for value in diagnostics.kl_per_dimension)
        assert all(type(value) is float for value in diagnostics.mu_variance_per_dimension)
        assert type(diagnostics.active_units) is int
        assert type(diagnostics.num_dimensions) is int

    def test_the_result_is_json_serializable(self) -> None:
        diagnostics = computeLatentDiagnostics(torch.randn(5, 2), torch.randn(5, 2))
        round_trip = json.loads(json.dumps(asdict(diagnostics)))
        assert round_trip["active_units"] == diagnostics.active_units
        assert round_trip["kl_per_dimension"] == pytest.approx(diagnostics.kl_per_dimension)

    def test_a_double_precision_input_works(self) -> None:
        diagnostics = computeLatentDiagnostics(
            torch.randn(5, 2, dtype=torch.float64), torch.zeros(5, 2, dtype=torch.float64)
        )
        assert diagnostics.num_dimensions == 2


class TestValidation:
    def test_mu_must_be_two_dimensional(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(samples, dim\), got \(3,\)"):
            computeLatentDiagnostics(torch.zeros(3), torch.zeros(3))

    def test_mu_and_logvar_must_have_the_same_shape(self) -> None:
        with pytest.raises(ValueError, match=r"same shape.*\(4, 3\).*\(4, 2\)"):
            computeLatentDiagnostics(torch.zeros(4, 3), torch.zeros(4, 2))

    def test_no_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            computeLatentDiagnostics(torch.zeros(0, 3), torch.zeros(0, 3))

    def test_no_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            computeLatentDiagnostics(torch.zeros(3, 0), torch.zeros(3, 0))

    @pytest.mark.parametrize("threshold", [-0.1, math.inf, -math.inf, math.nan])
    def test_a_threshold_that_is_negative_or_not_finite_is_rejected(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="active_unit_threshold"):
            computeLatentDiagnostics(torch.zeros(3, 2), torch.zeros(3, 2), threshold)
        with pytest.raises(ValueError, match="active_unit_threshold"):
            validateActiveUnitThreshold(threshold)

    @pytest.mark.parametrize("threshold", [0.0, 0.01, 1.0, 100.0])
    def test_a_finite_non_negative_threshold_is_accepted(self, threshold: float) -> None:
        validateActiveUnitThreshold(threshold)


def test_the_dataclass_can_be_built_by_hand() -> None:
    diagnostics = LatentDiagnostics(num_dimensions=2, active_threshold=0.01, active_units=1)
    assert diagnostics.kl_per_dimension == []
    assert diagnostics.mu_variance_per_dimension == []
