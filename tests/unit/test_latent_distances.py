"""The latent distances of roadmap P6-3 (ADR 0053).

The registry itself (registration, lookup, duplicates, unknown names) is covered for all thirteen
registries by `test_registry_contract.py`. This file checks what is specific to the distances:

- what the six classes declare (`is_symmetric`, `is_zero_on_identical`, `is_stochastic`,
  `max_spaces`, `min_batch_size`), and that each declaration is true of the class;
- the properties the roadmap asks for, over all six: non-negative, zero on identical inputs where
  declared, symmetric where declared, one value per sample, finite gradients at the origin,
  `float32` under mixed precision;
- the closed forms, on one-dimensional cases worked out by hand (the 2-Wasserstein distance between
  `N(0, 1^2)` and `N(1, 2^2)` is the square root of 2) and against independent formulas;
- the two distances that are not closed forms, `mixture_jsd` against a quadrature and its exact
  limits, and `info_nce` against the cross-entropy worked out by hand;
- the refusals: a malformed input, a pair of the wrong shape, too many spaces, too small a batch,
  a bad option.
"""

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest
import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import (
    getLatentDistanceClass,
    listRegisteredLatentDistances,
)
from global_vae.losses.distances.wasserstein2_diag import Wasserstein2DiagDistance

NAMES = listRegisteredLatentDistances()
EXACT_NAMES = [name for name in NAMES if not getLatentDistanceClass(name).is_stochastic]
ZERO_ON_IDENTICAL = [name for name in NAMES if getLatentDistanceClass(name).is_zero_on_identical]

# Every name with the options that change what is computed, so that the properties are checked on
# each branch (the root of the 2-Wasserstein distance, the mean over dimensions, K-space sums).
VARIANTS: list[tuple[str, dict[str, Any]]] = [
    *[(name, {}) for name in NAMES],
    ("mean_mse", {"dimension_reduction": "mean"}),
    ("wasserstein2_diag", {"squared": False}),
    ("wasserstein2_diag", {"dimension_reduction": "mean", "squared": False}),
    ("symmetric_kl", {"dimension_reduction": "mean"}),
    ("mixture_jsd", {"num_samples": 3, "sum_over_spaces": True}),
    ("mixture_jsd", {"dimension_reduction": "mean"}),
    ("info_nce", {"temperature": 0.5}),
]
VARIANT_IDS = [f"{name}{kwargs}" if kwargs else name for name, kwargs in VARIANTS]

# These compare the means only: the log-variance is not in their graph, so it gets no gradient.
IGNORES_VARIANCES = {"mean_mse", "cosine", "info_nce"}


@pytest.fixture(autouse=True)
def seedTorch() -> Iterator[None]:
    """Start every test from the same random state, and put the previous one back afterwards.

    `mixture_jsd` draws from the global generator, so without this a test that reads a Monte Carlo
    value would depend on which tests ran before it.
    """
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


def _make(name: str, **kwargs: Any) -> AbstractLatentDistance:
    return getLatentDistanceClass(name)(**kwargs)


def _posterior(
    batch: int = 6,
    dim: int = 5,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
) -> LatentParams:
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch, dim, generator=generator) * scale
    logvar = torch.randn(batch, dim, generator=generator) * 0.5
    return mu.to(dtype), logvar.to(dtype)


def _row(mu: float, logvar: float, dtype: torch.dtype = torch.float64) -> LatentParams:
    """A one-sample, one-dimensional posterior."""
    return (
        torch.tensor([[mu]], dtype=dtype),
        torch.tensor([[logvar]], dtype=dtype),
    )


def _assertFiniteGradients(name: str, posteriors: list[LatentParams]) -> None:
    """Every mean has a finite gradient; so has every log-variance the distance reads."""
    for mu, logvar in posteriors:
        assert mu.grad is not None, (name, "mu")
        assert torch.isfinite(mu.grad).all(), (name, mu.grad)
        if name in IGNORES_VARIANCES:
            assert logvar.grad is None, (name, "logvar is not read, so it has no gradient")
        else:
            assert logvar.grad is not None, (name, "logvar")
            assert torch.isfinite(logvar.grad).all(), (name, logvar.grad)


class TestDeclarations:
    def test_the_six_built_ins_are_registered(self) -> None:
        assert NAMES == [
            "cosine",
            "info_nce",
            "mean_mse",
            "mixture_jsd",
            "symmetric_kl",
            "wasserstein2_diag",
        ]

    @pytest.mark.parametrize("name", NAMES)
    def test_every_entry_is_a_latent_distance_module_with_no_state(self, name: str) -> None:
        distance = _make(name)
        assert isinstance(distance, AbstractLatentDistance)
        assert isinstance(distance, torch.nn.Module)
        assert list(distance.parameters()) == []
        assert distance.state_dict() == {}

    def test_what_each_class_declares(self) -> None:
        declared = {
            name: (
                cls.is_symmetric,
                cls.is_zero_on_identical,
                cls.is_stochastic,
                cls.max_spaces,
                cls.min_batch_size,
            )
            for name in NAMES
            for cls in [getLatentDistanceClass(name)]
        }
        assert declared == {
            "cosine": (True, True, False, 2, 1),
            "info_nce": (True, False, False, 2, 2),
            "mean_mse": (True, True, False, 2, 1),
            "mixture_jsd": (True, True, True, None, 1),
            "symmetric_kl": (True, True, False, 2, 1),
            "wasserstein2_diag": (True, True, False, 2, 1),
        }

    def test_the_defaults_of_the_base_class_are_the_common_case(self) -> None:
        assert AbstractLatentDistance.is_symmetric is True
        assert AbstractLatentDistance.is_zero_on_identical is True
        assert AbstractLatentDistance.is_stochastic is False
        assert AbstractLatentDistance.max_spaces == 2
        assert AbstractLatentDistance.min_batch_size == 1

    def test_the_abstract_method_cannot_be_skipped(self) -> None:
        with pytest.raises(TypeError, match="computeDistance"):
            AbstractLatentDistance()  # type: ignore[abstract]


class TestProperties:
    @pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=VARIANT_IDS)
    def test_one_value_per_sample_in_float32(self, name: str, kwargs: dict[str, Any]) -> None:
        value = _make(name, **kwargs)(_posterior(seed=1), _posterior(seed=2))
        assert value.shape == (6,)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()

    @pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=VARIANT_IDS)
    def test_it_is_non_negative(self, name: str, kwargs: dict[str, Any]) -> None:
        stochastic = getLatentDistanceClass(name).is_stochastic
        if stochastic:
            # An unbiased, unclamped estimate (ADR 0053): with one draw the mean of a small batch
            # can fall below 0 when the posteriors nearly coincide (about 1 case in 400 here). With
            # many draws the noise is far below the divergence, so the mean cannot.
            kwargs = {**kwargs, "num_samples": 256}
        distance = _make(name, **kwargs)
        for seed in range(5):
            a = _posterior(batch=8, dim=4, seed=seed, dtype=torch.float64)
            b = _posterior(batch=8, dim=4, seed=seed + 100, dtype=torch.float64)
            value = distance(a, b)
            if stochastic:
                assert value.mean() >= 0.0
            else:
                assert (value >= -1e-12).all(), value

    @pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=VARIANT_IDS)
    def test_it_is_zero_on_identical_inputs_when_it_says_so(
        self, name: str, kwargs: dict[str, Any]
    ) -> None:
        distance = _make(name, **kwargs)
        posterior = _posterior(batch=5, dim=4, seed=3)
        value = distance(posterior, posterior)
        if distance.is_zero_on_identical:
            assert torch.allclose(value, torch.zeros_like(value), atol=1e-6), value
        else:
            # A contrastive loss is a cross-entropy: positive even for a perfect match, and it
            # only approaches 0 when the samples of the batch are far apart.
            assert (value > 0.0).all(), value

    @pytest.mark.parametrize("name", [n for n in EXACT_NAMES if n != "info_nce"])
    def test_it_is_symmetric(self, name: str) -> None:
        distance = _make(name)
        a = _posterior(seed=4)
        b = _posterior(seed=5)
        assert distance.is_symmetric
        assert torch.allclose(distance(a, b), distance(b, a), rtol=1e-5, atol=1e-6)

    def test_info_nce_is_symmetric_because_it_averages_the_two_directions(self) -> None:
        distance = _make("info_nce")
        a = _posterior(seed=4)
        b = _posterior(seed=5)
        assert torch.allclose(distance(a, b), distance(b, a), rtol=1e-5, atol=1e-6)

    def test_mixture_jsd_is_symmetric_in_expectation(self) -> None:
        distance = _make("mixture_jsd", num_samples=4000)
        a = _row(0.0, 0.0)
        b = _row(1.0, math.log(4.0))
        torch.manual_seed(0)
        forward = distance(a, b)
        torch.manual_seed(1)
        backward = distance(b, a)
        assert abs(forward.item() - backward.item()) < 0.01

    @pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=VARIANT_IDS)
    def test_gradients_are_finite_at_the_origin(self, name: str, kwargs: dict[str, Any]) -> None:
        distance = _make(name, **kwargs)
        mu_a = torch.zeros(4, 3, requires_grad=True)
        logvar_a = torch.zeros(4, 3, requires_grad=True)
        mu_b = torch.zeros(4, 3, requires_grad=True)
        logvar_b = torch.zeros(4, 3, requires_grad=True)
        value = distance((mu_a, logvar_a), (mu_b, logvar_b))
        assert value.requires_grad
        value.mean().backward()
        _assertFiniteGradients(name, [(mu_a, logvar_a), (mu_b, logvar_b)])

    @pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=VARIANT_IDS)
    def test_gradients_are_finite_on_identical_inputs_away_from_the_origin(
        self, name: str, kwargs: dict[str, Any]
    ) -> None:
        distance = _make(name, **kwargs)
        mu, logvar = _posterior(batch=4, dim=3, seed=7)
        pair = [(mu.clone().requires_grad_(), logvar.clone().requires_grad_()) for _ in range(2)]
        distance(*pair).mean().backward()
        _assertFiniteGradients(name, pair)

    @pytest.mark.parametrize("name", EXACT_NAMES)
    def test_the_gradient_does_not_nan_on_a_zero_distance_even_without_the_origin(
        self, name: str
    ) -> None:
        # The root of the 2-Wasserstein distance is the case that needs care, and it is below.
        distance = _make(name)
        mu = torch.full((3, 2), 0.7, requires_grad=True)
        logvar = torch.full((3, 2), -0.3, requires_grad=True)
        distance((mu, logvar), (mu.detach().clone(), logvar.detach().clone())).sum().backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()


class TestMixedPrecision:
    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_inputs_give_a_float32_value(
        self, name: str, dtype: torch.dtype
    ) -> None:
        a = _posterior(seed=1, dtype=dtype)
        b = _posterior(seed=2, dtype=dtype)
        value = _make(name)(a, b)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()

    @pytest.mark.parametrize("name", NAMES)
    def test_half_precision_gradients_reach_the_inputs_in_their_own_dtype(self, name: str) -> None:
        mu_a, logvar_a = (t.requires_grad_() for t in _posterior(seed=1, dtype=torch.bfloat16))
        b = _posterior(seed=2, dtype=torch.bfloat16)
        _make(name)((mu_a, logvar_a), b).mean().backward()
        assert mu_a.grad is not None and mu_a.grad.dtype == torch.bfloat16
        assert torch.isfinite(mu_a.grad.float()).all()

    @pytest.mark.parametrize("name", NAMES)
    def test_the_value_is_the_same_inside_and_outside_autocast(self, name: str) -> None:
        distance = _make(name)
        a = _posterior(batch=8, dim=16, seed=1)
        b = _posterior(batch=8, dim=16, seed=2)
        torch.manual_seed(5)
        plain = distance(a, b)
        torch.manual_seed(5)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            inside = distance(a, b)
        assert inside.dtype == torch.float32
        assert torch.equal(plain, inside)

    def test_a_matrix_product_would_not_survive_autocast_without_the_guard(self) -> None:
        # What the guard is for: info_nce is a matrix product, which autocast runs in bfloat16.
        a = torch.randn(8, 16)
        b = torch.randn(8, 16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert (a @ b.T).dtype == torch.bfloat16

    @pytest.mark.parametrize("name", NAMES)
    def test_float64_is_kept(self, name: str) -> None:
        a = _posterior(seed=1, dtype=torch.float64)
        b = _posterior(seed=2, dtype=torch.float64)
        assert _make(name)(a, b).dtype == torch.float64


class TestMeanMse:
    def test_one_dimensional_case(self) -> None:
        value = _make("mean_mse")(_row(0.0, 0.0), _row(1.0, 0.0))
        assert value.item() == pytest.approx(1.0)

    def test_sum_and_mean_over_the_dimensions(self) -> None:
        a = (torch.zeros(1, 2), torch.zeros(1, 2))
        b = (torch.tensor([[1.0, 3.0]]), torch.zeros(1, 2))
        assert _make("mean_mse")(a, b).item() == pytest.approx(10.0)
        assert _make("mean_mse", dimension_reduction="mean")(a, b).item() == pytest.approx(5.0)

    def test_it_ignores_the_variances(self) -> None:
        mu_a, mu_b = _posterior(seed=1)[0], _posterior(seed=2)[0]
        distance = _make("mean_mse")
        small = distance((mu_a, torch.full_like(mu_a, -5.0)), (mu_b, torch.full_like(mu_b, 3.0)))
        large = distance((mu_a, torch.full_like(mu_a, 4.0)), (mu_b, torch.full_like(mu_b, -2.0)))
        assert torch.equal(small, large)

    def test_it_is_the_squared_euclidean_distance(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        expected = torch.linalg.vector_norm(a[0] - b[0], dim=-1).pow(2)
        assert torch.allclose(_make("mean_mse")(a, b), expected, rtol=1e-5)


class TestWasserstein2Diag:
    def test_the_distance_between_n01_and_n1_2sq_is_the_root_of_two(self) -> None:
        a = _row(0.0, 0.0)  # N(0, 1^2)
        b = _row(1.0, 2.0 * math.log(2.0))  # N(1, 2^2)
        assert _make("wasserstein2_diag", squared=False)(a, b).item() == pytest.approx(
            math.sqrt(2.0)
        )
        assert _make("wasserstein2_diag")(a, b).item() == pytest.approx(2.0)

    def test_the_default_is_the_squared_form(self) -> None:
        assert Wasserstein2DiagDistance().squared is True

    def test_it_matches_the_general_formula_for_diagonal_gaussians(self) -> None:
        # trace(S_a + S_b - 2 (S_a^(1/2) S_b S_a^(1/2))^(1/2)) with diagonal covariances, plus the
        # squared gap of the means: the formula for any two Gaussians, applied to diagonals.
        a = _posterior(batch=4, dim=3, seed=1, dtype=torch.float64)
        b = _posterior(batch=4, dim=3, seed=2, dtype=torch.float64)
        got = _make("wasserstein2_diag")(a, b).numpy()
        for row in range(4):
            cov_a = np.diag(np.exp(a[1][row].numpy()))
            cov_b = np.diag(np.exp(b[1][row].numpy()))
            root_a = np.sqrt(cov_a)
            middle = np.sqrt(root_a @ cov_b @ root_a)
            expected = ((a[0][row] - b[0][row]).numpy() ** 2).sum() + np.trace(
                cov_a + cov_b - 2.0 * middle
            )
            assert got[row] == pytest.approx(expected, rel=1e-9)

    def test_the_root_is_the_root_of_the_squared_value(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        squared = _make("wasserstein2_diag")(a, b)
        root = _make("wasserstein2_diag", squared=False)(a, b)
        assert torch.allclose(root, squared.sqrt(), rtol=1e-6)

    def test_the_root_is_taken_after_the_dimensions_are_reduced(self) -> None:
        a = (torch.zeros(1, 4), torch.zeros(1, 4))
        b = (torch.ones(1, 4), torch.zeros(1, 4))
        root_of_sum = _make("wasserstein2_diag", squared=False)(a, b)
        root_of_mean = _make("wasserstein2_diag", squared=False, dimension_reduction="mean")(a, b)
        assert root_of_sum.item() == pytest.approx(2.0)
        assert root_of_mean.item() == pytest.approx(1.0)

    def test_the_gradient_of_the_root_is_zero_and_finite_where_the_distance_is_zero(self) -> None:
        mu = torch.zeros(1, 2, requires_grad=True)
        logvar = torch.zeros(1, 2, requires_grad=True)
        value = _make("wasserstein2_diag", squared=False)(
            (mu, logvar), (torch.zeros(1, 2), torch.zeros(1, 2))
        )
        assert value.item() == 0.0
        value.sum().backward()
        assert mu.grad is not None and torch.equal(mu.grad, torch.zeros(1, 2))
        assert logvar.grad is not None and torch.equal(logvar.grad, torch.zeros(1, 2))

    def test_the_gradient_pulls_the_means_together(self) -> None:
        mu_a = torch.tensor([[0.0]], requires_grad=True)
        _make("wasserstein2_diag")(
            (mu_a, torch.zeros(1, 1)), _row(2.0, 0.0, torch.float32)
        ).sum().backward()
        assert mu_a.grad is not None and mu_a.grad.item() < 0.0

    def test_it_uses_the_variances(self) -> None:
        distance = _make("wasserstein2_diag")
        same_mean_a = _row(1.0, 0.0)
        assert distance(same_mean_a, _row(1.0, 0.0)).item() == pytest.approx(0.0)
        assert distance(same_mean_a, _row(1.0, 2.0)).item() > 0.1


class TestSymmetricKl:
    def test_one_dimensional_case(self) -> None:
        # N(0, 1) and N(1, 4): KL(a||b) = 0.5 (ln 4 + 2/4 - 1), KL(b||a) = 0.5 (-ln 4 + 5 - 1),
        # and the logarithms cancel in the sum: 0.5 (0.5 + 5 - 2) = 1.75.
        value = _make("symmetric_kl")(_row(0.0, 0.0), _row(1.0, math.log(4.0)))
        assert value.item() == pytest.approx(1.75)

    def test_it_is_the_sum_of_the_two_directed_divergences(self) -> None:
        def directed(a: LatentParams, b: LatentParams) -> torch.Tensor:
            (mu_a, logvar_a), (mu_b, logvar_b) = a, b
            per_dimension = 0.5 * (
                logvar_b - logvar_a + (logvar_a.exp() + (mu_a - mu_b).pow(2)) / logvar_b.exp() - 1.0
            )
            return per_dimension.sum(dim=-1)

        a = _posterior(seed=1, dtype=torch.float64)
        b = _posterior(seed=2, dtype=torch.float64)
        got = _make("symmetric_kl")(a, b)
        assert torch.allclose(got, directed(a, b) + directed(b, a), rtol=1e-9)

    def test_a_tiny_variance_is_clamped_instead_of_exploding(self) -> None:
        distance = _make("symmetric_kl")
        tiny = distance(_row(0.0, -100.0), _row(0.5, 0.0))
        at_the_clamp = distance(_row(0.0, -10.0), _row(0.5, 0.0))
        assert torch.isfinite(tiny).all()
        assert tiny.item() == pytest.approx(at_the_clamp.item())

    def test_the_clamp_is_configurable(self) -> None:
        loose = _make("symmetric_kl", min_logvar=-20.0)(_row(0.0, -15.0), _row(0.5, 0.0))
        tight = _make("symmetric_kl", min_logvar=-5.0)(_row(0.0, -15.0), _row(0.5, 0.0))
        assert loose.item() > 100.0 * tight.item()

    def test_a_clamped_log_variance_has_no_gradient(self) -> None:
        logvar = torch.tensor([[-50.0]], requires_grad=True)
        _make("symmetric_kl")(
            (torch.zeros(1, 1), logvar), _row(0.5, 0.0, torch.float32)
        ).sum().backward()
        assert logvar.grad is not None and logvar.grad.item() == 0.0

    def test_it_explodes_without_the_clamp_which_is_why_it_is_there(self) -> None:
        wide = _make("symmetric_kl", min_logvar=-60.0)(_row(0.0, -50.0), _row(0.5, 0.0))
        assert wide.item() > 1e20

    def test_sum_and_mean_over_the_dimensions(self) -> None:
        a = (torch.zeros(1, 4), torch.zeros(1, 4))
        b = (torch.ones(1, 4), torch.zeros(1, 4))
        total = _make("symmetric_kl")(a, b)
        mean = _make("symmetric_kl", dimension_reduction="mean")(a, b)
        assert total.item() == pytest.approx(4.0)  # per dimension 0.5 * (1 + 1) = 1
        assert mean.item() == pytest.approx(1.0)


class TestCosine:
    @pytest.mark.parametrize(
        ("mu_a", "mu_b", "expected"),
        [
            ([1.0, 0.0], [0.0, 1.0], 1.0),
            ([1.0, 0.0], [2.0, 0.0], 0.0),
            ([1.0, 0.0], [-1.0, 0.0], 2.0),
            ([1.0, 1.0], [1.0, 0.0], 1.0 - 1.0 / math.sqrt(2.0)),
        ],
    )
    def test_hand_computed_cases(
        self, mu_a: list[float], mu_b: list[float], expected: float
    ) -> None:
        zero = torch.zeros(1, 2, dtype=torch.float64)
        a = (torch.tensor([mu_a], dtype=torch.float64), zero)
        b = (torch.tensor([mu_b], dtype=torch.float64), zero)
        assert _make("cosine")(a, b).item() == pytest.approx(expected)

    def test_it_is_blind_to_the_scale_of_the_means(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        scaled = ((a[0] * 7.0), a[1])
        distance = _make("cosine")
        assert torch.allclose(distance(a, b), distance(scaled, b), atol=1e-6)

    def test_it_ignores_the_variances(self) -> None:
        mu_a, mu_b = _posterior(seed=1)[0], _posterior(seed=2)[0]
        distance = _make("cosine")
        assert torch.equal(
            distance((mu_a, torch.zeros_like(mu_a)), (mu_b, torch.zeros_like(mu_b))),
            distance((mu_a, torch.full_like(mu_a, 3.0)), (mu_b, torch.full_like(mu_b, -3.0))),
        )

    def test_a_zero_mean_has_no_direction_and_counts_as_orthogonal(self) -> None:
        zero = (torch.zeros(1, 3), torch.zeros(1, 3))
        other = (torch.ones(1, 3), torch.zeros(1, 3))
        distance = _make("cosine")
        assert distance(zero, other).item() == pytest.approx(1.0)
        assert distance(zero, zero).item() == pytest.approx(1.0)

    def test_the_range_is_zero_to_two(self) -> None:
        value = _make("cosine")(_posterior(batch=64, seed=1), _posterior(batch=64, seed=2))
        assert (value >= -1e-6).all() and (value <= 2.0 + 1e-6).all()

    def test_it_takes_no_dimension_reduction(self) -> None:
        with pytest.raises(TypeError, match="dimension_reduction"):
            _make("cosine", dimension_reduction="mean")


class TestInfoNce:
    def test_identical_orthogonal_directions_at_temperature_one(self) -> None:
        eye = torch.eye(2, dtype=torch.float64)
        zero = torch.zeros(2, 2, dtype=torch.float64)
        # logits [[1, 0], [0, 1]]: the loss of each row is -log(e / (e + 1)) = log(1 + e^-1).
        value = _make("info_nce", temperature=1.0)((eye, zero), (eye, zero))
        expected = math.log(1.0 + math.exp(-1.0))
        assert value.tolist() == pytest.approx([expected, expected])

    def test_the_temperature_scales_the_logits(self) -> None:
        eye = torch.eye(2, dtype=torch.float64)
        zero = torch.zeros(2, 2, dtype=torch.float64)
        value = _make("info_nce", temperature=0.5)((eye, zero), (eye, zero))
        assert value[0].item() == pytest.approx(math.log(1.0 + math.exp(-2.0)))

    def test_with_more_negatives_the_floor_is_higher(self) -> None:
        eye = torch.eye(4, dtype=torch.float64)
        zero = torch.zeros(4, 4, dtype=torch.float64)
        value = _make("info_nce", temperature=1.0)((eye, zero), (eye, zero))
        assert value[0].item() == pytest.approx(math.log(1.0 + 3.0 * math.exp(-1.0)))

    def test_when_nothing_tells_the_samples_apart_the_loss_is_the_chance_level(self) -> None:
        same = (torch.ones(5, 3), torch.zeros(5, 3))
        value = _make("info_nce")(same, same)
        assert value.tolist() == pytest.approx([math.log(5.0)] * 5, rel=1e-5)

    def test_a_matched_batch_scores_lower_than_a_shuffled_one(self) -> None:
        mu = torch.randn(16, 8, generator=torch.Generator().manual_seed(0))
        zero = torch.zeros(16, 8)
        distance = _make("info_nce")
        matched = distance((mu, zero), (mu + 0.01, zero)).mean()
        shuffled = distance((mu, zero), (mu.roll(1, dims=0), zero)).mean()
        assert matched < shuffled

    def test_the_value_of_a_sample_depends_on_the_others_of_the_batch(self) -> None:
        mu = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
        zero = torch.zeros(4, 8)
        distance = _make("info_nce")
        full = distance((mu, zero), (mu, zero))
        alone = distance((mu[:2], zero[:2]), (mu[:2], zero[:2]))
        assert not torch.allclose(full[:2], alone)

    def test_it_ignores_the_variances(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        distance = _make("info_nce")
        other = (torch.full_like(a[1], 2.0), torch.full_like(b[1], -2.0))
        assert torch.equal(distance(a, b), distance((a[0], other[0]), (b[0], other[1])))

    def test_a_gradient_step_makes_a_pair_closer_than_the_others(self) -> None:
        generator = torch.Generator().manual_seed(2)
        mu_a = torch.randn(8, 6, generator=generator, requires_grad=True)
        mu_b = torch.randn(8, 6, generator=generator, requires_grad=True)
        zero = torch.zeros(8, 6)
        distance = _make("info_nce")
        before = distance((mu_a, zero), (mu_b, zero)).mean()
        before.backward()
        assert mu_a.grad is not None and mu_b.grad is not None
        with torch.no_grad():
            stepped_a = mu_a - 0.5 * mu_a.grad
            stepped_b = mu_b - 0.5 * mu_b.grad
        after = distance((stepped_a, zero), (stepped_b, zero)).mean()
        assert after < before

    def test_a_batch_of_one_has_no_negatives(self) -> None:
        one = _posterior(batch=1)
        with pytest.raises(ValueError, match=r"InfoNceDistance needs at least 2 samples"):
            _make("info_nce")(one, one)

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_the_temperature_must_be_positive(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            _make("info_nce", temperature=temperature)

    def test_it_takes_no_dimension_reduction(self) -> None:
        with pytest.raises(TypeError, match="dimension_reduction"):
            _make("info_nce", dimension_reduction="mean")


def _jsd_by_quadrature(posteriors: list[tuple[float, float]]) -> float:
    """The Jensen-Shannon divergence of 1-D Gaussians `(mu, sigma)`, by a fine grid."""
    grid = np.linspace(-40.0, 40.0, 400001)
    step = grid[1] - grid[0]
    densities = [
        np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
        for mu, sigma in posteriors
    ]
    mixture = np.mean(densities, axis=0)
    total = 0.0
    for density in densities:
        mask = density > 1e-300
        total += float(np.sum(density[mask] * np.log(density[mask] / mixture[mask])) * step)
    return total / len(densities)


class TestMixtureJsd:
    @staticmethod
    def _batch_of(rows: int, mu: float, logvar: float) -> LatentParams:
        return (
            torch.full((rows, 1), mu, dtype=torch.float64),
            torch.full((rows, 1), logvar, dtype=torch.float64),
        )

    def test_it_estimates_the_divergence_computed_by_quadrature(self) -> None:
        expected = _jsd_by_quadrature([(0.0, 1.0), (1.0, 2.0)])
        distance = _make("mixture_jsd")
        torch.manual_seed(0)
        estimate = distance(
            self._batch_of(40000, 0.0, 0.0), self._batch_of(40000, 1.0, math.log(4.0))
        ).mean()
        assert estimate.item() == pytest.approx(expected, abs=0.01)
        assert 0.0 < expected < math.log(2.0)

    def test_it_estimates_a_three_space_divergence(self) -> None:
        expected = _jsd_by_quadrature([(0.0, 1.0), (1.0, 1.0), (-1.5, 0.5)])
        torch.manual_seed(1)
        estimate = _make("mixture_jsd")(
            self._batch_of(40000, 0.0, 0.0),
            self._batch_of(40000, 1.0, 0.0),
            self._batch_of(40000, -1.5, 2.0 * math.log(0.5)),
        ).mean()
        assert estimate.item() == pytest.approx(expected, abs=0.01)

    def test_far_apart_posteriors_give_the_log_of_the_number_of_spaces(self) -> None:
        distance = _make("mixture_jsd")
        far_two = distance(_row(-30.0, 0.0), _row(30.0, 0.0))
        far_three = distance(_row(-30.0, 0.0), _row(0.0, 0.0), _row(30.0, 0.0))
        assert far_two.item() == pytest.approx(math.log(2.0), abs=1e-9)
        assert far_three.item() == pytest.approx(math.log(3.0), abs=1e-9)

    def test_the_mmvm_form_is_the_number_of_spaces_times_the_divergence(self) -> None:
        plain = _make("mixture_jsd")
        mmvm = _make("mixture_jsd", sum_over_spaces=True)
        spaces = (_row(-30.0, 0.0), _row(0.0, 0.0), _row(30.0, 0.0))
        assert plain(*spaces).item() == pytest.approx(math.log(3.0), abs=1e-9)
        assert mmvm(*spaces).item() == pytest.approx(3.0 * math.log(3.0), abs=1e-9)

    @pytest.mark.parametrize("count", [2, 3, 4])
    def test_identical_posteriors_give_exactly_zero_whatever_the_draws(self, count: int) -> None:
        posterior = _posterior(batch=7, dim=5, seed=3, dtype=torch.float64)
        for seed in range(3):
            torch.manual_seed(seed)
            value = _make("mixture_jsd", num_samples=5)(*[posterior] * count)
            assert value.abs().max().item() < 1e-12

    def test_one_draw_never_exceeds_the_log_of_the_number_of_spaces(self) -> None:
        a = _posterior(batch=512, dim=3, seed=1, dtype=torch.float64, scale=3.0)
        b = _posterior(batch=512, dim=3, seed=2, dtype=torch.float64, scale=3.0)
        torch.manual_seed(0)
        value = _make("mixture_jsd")(a, b)
        assert (value <= math.log(2.0) + 1e-9).all()

    def test_it_draws_from_the_global_generator(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        distance = _make("mixture_jsd")
        torch.manual_seed(3)
        first = distance(a, b)
        torch.manual_seed(3)
        again = distance(a, b)
        torch.manual_seed(4)
        other = distance(a, b)
        assert torch.equal(first, again)
        assert not torch.equal(first, other)

    def test_more_draws_lower_the_variance_of_the_estimate(self) -> None:
        a = self._batch_of(2000, 0.0, 0.0)
        b = self._batch_of(2000, 1.0, math.log(4.0))
        torch.manual_seed(0)
        one = _make("mixture_jsd", num_samples=1)(a, b)
        torch.manual_seed(0)
        sixteen = _make("mixture_jsd", num_samples=16)(a, b)
        assert sixteen.std() < 0.5 * one.std()

    def test_the_mean_over_dimensions_is_the_sum_divided_by_the_width(self) -> None:
        a = _posterior(batch=6, dim=4, seed=1)
        b = _posterior(batch=6, dim=4, seed=2)
        torch.manual_seed(0)
        total = _make("mixture_jsd")(a, b)
        torch.manual_seed(0)
        mean = _make("mixture_jsd", dimension_reduction="mean")(a, b)
        assert torch.allclose(mean, total / 4.0, rtol=1e-6)

    def test_the_gradient_pulls_the_means_together(self) -> None:
        mu_a = torch.zeros(4000, 1, dtype=torch.float64, requires_grad=True)
        logvar = torch.zeros(4000, 1, dtype=torch.float64)
        torch.manual_seed(0)
        _make("mixture_jsd")((mu_a, logvar), self._batch_of(4000, 2.0, 0.0)).mean().backward()
        assert mu_a.grad is not None
        assert mu_a.grad.sum().item() < 0.0

    def test_the_gradient_reaches_both_the_means_and_the_log_variances(self) -> None:
        a = tuple(t.requires_grad_() for t in _posterior(batch=8, dim=3, seed=1))
        b = tuple(t.requires_grad_() for t in _posterior(batch=8, dim=3, seed=2))
        _make("mixture_jsd", num_samples=4)(a, b).mean().backward()
        for tensor in (*a, *b):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
            assert tensor.grad.abs().sum() > 0.0

    def test_a_log_variance_out_of_range_stays_finite(self) -> None:
        value = _make("mixture_jsd")(_row(0.0, -300.0), _row(1.0, 300.0))
        assert torch.isfinite(value).all()

    @pytest.mark.parametrize("num_samples", [0, -1, 1.5, True, "2"])
    def test_the_number_of_draws_must_be_a_positive_integer(self, num_samples: Any) -> None:
        with pytest.raises(ValueError, match="num_samples must be an integer of at least 1"):
            _make("mixture_jsd", num_samples=num_samples)


class TestOptions:
    @pytest.mark.parametrize(
        "name", ["mean_mse", "wasserstein2_diag", "symmetric_kl", "mixture_jsd"]
    )
    def test_a_bad_dimension_reduction_is_refused(self, name: str) -> None:
        with pytest.raises(
            ValueError, match=r"dimension_reduction must be one of \['sum', 'mean'\]"
        ):
            _make(name, dimension_reduction="max")

    @pytest.mark.parametrize("name", ["symmetric_kl", "mixture_jsd"])
    def test_the_log_variance_range_must_be_ordered(self, name: str) -> None:
        with pytest.raises(ValueError, match="min_logvar must be below max_logvar"):
            _make(name, min_logvar=2.0, max_logvar=2.0)

    @pytest.mark.parametrize("name", ["cosine", "info_nce"])
    def test_eps_must_be_positive(self, name: str) -> None:
        with pytest.raises(ValueError, match="eps must be positive"):
            _make(name, eps=0.0)

    def test_the_default_scale_is_the_sum_over_the_dimensions(self) -> None:
        for name in ("mean_mse", "wasserstein2_diag", "symmetric_kl", "mixture_jsd"):
            assert _make(name).dimension_reduction == "sum"


class TestRefusals:
    @pytest.mark.parametrize("name", NAMES)
    def test_a_posterior_must_be_a_pair(self, name: str) -> None:
        mu = torch.zeros(3, 2)
        with pytest.raises(ValueError, match=r"latent space 0 must be a \(mu, logvar\) pair"):
            _make(name)(mu, (mu, mu))

    @pytest.mark.parametrize("name", NAMES)
    def test_the_tensors_must_be_floating_point(self, name: str) -> None:
        ints = torch.zeros(3, 2, dtype=torch.long)
        good = _posterior(batch=3, dim=2)
        with pytest.raises(ValueError, match="mu of latent space 0 must be a floating point"):
            _make(name)((ints, good[1]), good)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_tensors_must_be_two_dimensional(self, name: str) -> None:
        flat = torch.zeros(3)
        good = _posterior(batch=3, dim=2)
        with pytest.raises(ValueError, match=r"must have shape \(batch, dim\), got \(3,\)"):
            _make(name)(good, (flat, flat))

    @pytest.mark.parametrize("name", NAMES)
    def test_logvar_must_have_the_shape_of_mu(self, name: str) -> None:
        good = _posterior(batch=3, dim=2)
        with pytest.raises(ValueError, match="logvar of latent space 1 has shape"):
            _make(name)(good, (good[0], torch.zeros(3, 3)))

    @pytest.mark.parametrize("name", NAMES)
    def test_two_spaces_of_different_widths_are_refused_by_name(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"latent space 1 has shape \(4, 3\).*latent space 0"):
            _make(name)(_posterior(batch=4, dim=2), _posterior(batch=4, dim=3))

    @pytest.mark.parametrize("name", NAMES)
    def test_two_spaces_of_different_batch_sizes_are_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="same number of samples and the same width"):
            _make(name)(_posterior(batch=4, dim=2), _posterior(batch=5, dim=2))

    @pytest.mark.parametrize("name", [n for n in NAMES if n != "mixture_jsd"])
    def test_a_distance_of_two_spaces_refuses_a_third(self, name: str) -> None:
        post = _posterior()
        with pytest.raises(ValueError, match="compares 2 latent spaces, got 3"):
            _make(name)(post, post, post)

    def test_a_wrong_result_shape_is_caught_in_the_base_class(self) -> None:
        class Scalar(AbstractLatentDistance):
            def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
                return posteriors[0][0].sum()

        post = _posterior(batch=3, dim=2)
        with pytest.raises(ValueError, match=r"returned shape \(\), but a distance returns one"):
            Scalar()(post, post)

    def test_a_subclass_inherits_the_default_declarations(self) -> None:
        class Plain(AbstractLatentDistance):
            def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
                return posteriors[0][0].sum(dim=-1)

        post = _posterior(batch=3, dim=2)
        assert Plain()(post, post).shape == (3,)
        with pytest.raises(ValueError, match="compares 2 latent spaces"):
            Plain()(post, post, post)

    def test_the_abstract_method_raises_if_a_subclass_delegates_to_it(self) -> None:
        class Delegating(AbstractLatentDistance):
            def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
                # Deliberate: this is the only way to reach the abstract body.
                return super().computeDistance(posteriors)  # type: ignore[safe-super]

        post = _posterior(batch=3, dim=2)
        with pytest.raises(NotImplementedError):
            Delegating()(post, post)

    def test_a_device_without_autocast_is_scored_like_any_other(self) -> None:
        # The meta device is how shapes are inferred without memory; it has no autocast context.
        mu = torch.zeros(3, 2, device="meta")
        value = _make("mean_mse")((mu, mu), (mu, mu))
        assert value.shape == (3,)
        assert value.device.type == "meta"

    def test_the_inputs_are_not_modified(self) -> None:
        a, b = _posterior(seed=1), _posterior(seed=2)
        copies = [t.clone() for t in (*a, *b)]
        for name in NAMES:
            _make(name)(a, b)
        for original, copy in zip((*a, *b), copies, strict=True):
            assert torch.equal(original, copy)
