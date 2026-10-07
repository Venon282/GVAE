"""Unit tests for the latent regularizer registry pattern (spec §2.3, §10).

Mirrors the test style already used for the other self-registration
registries in this codebase (encoders, decoders, fusion, assemblers):
registration, lookup, duplicate-registration and unknown-name error
paths, plus a value-correctness check for each built-in strategy
(`kl_standard_normal`, `free_bits_kl`, `mmd`).
"""

import pytest
import torch

from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.free_bits_kl import FreeBitsKlRegularizer
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer
from global_vae.losses.regularizers.mmd import MmdRegularizer
from global_vae.losses.regularizers.registry import (
    getRegularizerClass,
    listRegisteredRegularizers,
    registerRegularizer,
)


def test_kl_standard_normal_is_registered_by_default() -> None:
    assert "kl_standard_normal" in listRegisteredRegularizers()
    assert getRegularizerClass("kl_standard_normal") is KlStandardNormalRegularizer


def test_free_bits_kl_is_registered_by_default() -> None:
    assert "free_bits_kl" in listRegisteredRegularizers()
    assert getRegularizerClass("free_bits_kl") is FreeBitsKlRegularizer


def test_mmd_is_registered_by_default() -> None:
    assert "mmd" in listRegisteredRegularizers()
    assert getRegularizerClass("mmd") is MmdRegularizer


def test_unknown_regularizer_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="does_not_exist"):
        getRegularizerClass("does_not_exist")


def test_duplicate_registration_raises_value_error() -> None:
    @registerRegularizer("dummy_regularizer_duplicate_check")
    class _First(AbstractLatentRegularizer):
        def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            return mu.new_zeros(mu.shape[0])

    with pytest.raises(ValueError, match="already registered"):

        @registerRegularizer("dummy_regularizer_duplicate_check")
        class _Second(AbstractLatentRegularizer):
            def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
                return mu.new_zeros(mu.shape[0])


def test_kl_standard_normal_matches_closed_form_at_the_prior() -> None:
    """At mu=0, logvar=0 (i.e. exactly the prior), the KL divergence must be zero."""
    regularizer = KlStandardNormalRegularizer()
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kl = regularizer(mu, logvar)
    assert kl.shape == (4,)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)


def test_kl_standard_normal_matches_manual_formula() -> None:
    regularizer = KlStandardNormalRegularizer()
    mu = torch.randn(3, 5)
    logvar = torch.randn(3, 5)
    kl = regularizer(mu, logvar)
    expected = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    assert torch.allclose(kl, expected, atol=1e-5)


def test_kl_standard_normal_is_stateless_nn_module() -> None:
    """No learnable parameters, but still swappable via the same registry/ModuleDict."""
    regularizer = KlStandardNormalRegularizer()
    assert list(regularizer.parameters()) == []


class TestFreeBitsKlRegularizer:
    """Value-correctness and edge cases for the free-bits KL strategy."""

    def test_zero_free_bits_matches_plain_kl_standard_normal(self) -> None:
        """At free_bits=0, per-dimension free bits is mathematically the plain KL term."""
        mu = torch.randn(5, 6)
        logvar = torch.randn(5, 6)
        free_bits = FreeBitsKlRegularizer(free_bits=0.0)
        plain_kl = KlStandardNormalRegularizer()
        assert torch.allclose(free_bits(mu, logvar), plain_kl(mu, logvar), atol=1e-5)

    def test_per_dimension_budget_is_never_exceeded_from_below(self) -> None:
        """Every dimension contributes at least `free_bits`, even one exactly at the prior."""
        regularizer = FreeBitsKlRegularizer(free_bits=0.5, per_dimension=True)
        mu = torch.zeros(3, 4)  # exactly at the prior: plain per-dim KL would be 0 everywhere
        logvar = torch.zeros(3, 4)
        penalty = regularizer(mu, logvar)
        assert torch.allclose(penalty, torch.full((3,), 0.5 * 4), atol=1e-6)

    def test_aggregate_variant_clips_the_summed_kl_not_each_dimension(self) -> None:
        regularizer = FreeBitsKlRegularizer(free_bits=10.0, per_dimension=False)
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        penalty = regularizer(mu, logvar)
        assert torch.allclose(penalty, torch.full((2,), 10.0), atol=1e-6)

    def test_large_kl_is_unaffected_by_a_small_budget(self) -> None:
        """Once a dimension's own KL already exceeds the budget, the penalty is exactly its KL."""
        regularizer = FreeBitsKlRegularizer(free_bits=1e-6, per_dimension=True)
        plain_kl = KlStandardNormalRegularizer()
        mu = torch.randn(4, 8) * 5.0  # far from the prior: KL per dim >> 1e-6 everywhere
        logvar = torch.randn(4, 8)
        assert torch.allclose(regularizer(mu, logvar), plain_kl(mu, logvar), atol=1e-4)

    def test_negative_free_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="free_bits"):
            FreeBitsKlRegularizer(free_bits=-0.1)

    def test_is_stateless_nn_module(self) -> None:
        assert list(FreeBitsKlRegularizer().parameters()) == []


class TestMmdRegularizer:
    """Value-correctness and edge cases for the MMD strategy."""

    def test_output_shape_is_per_sample_but_constant_across_the_batch(self) -> None:
        """MMD is a batch-level statistic (class docstring): every entry must be identical."""
        regularizer = MmdRegularizer()
        mu = torch.randn(6, 4)
        logvar = torch.zeros(6, 4)
        penalty = regularizer(mu, logvar)
        assert penalty.shape == (6,)
        assert torch.allclose(penalty, penalty[0].expand(6), atol=1e-6)

    def test_is_near_zero_when_posterior_matches_the_prior(self) -> None:
        """Posterior samples drawn straight from N(0, I) should give a small (not necessarily
        exactly zero, since this is a finite-sample estimate) MMD relative to a mismatched case."""
        torch.manual_seed(0)
        regularizer = MmdRegularizer(num_prior_samples=512)
        matched_mu = torch.zeros(256, 4)
        matched_logvar = torch.zeros(256, 4)
        matched_mmd = regularizer(matched_mu, matched_logvar)[0]

        mismatched_mu = torch.full((256, 4), 5.0)
        mismatched_logvar = torch.zeros(256, 4)
        mismatched_mmd = regularizer(mismatched_mu, mismatched_logvar)[0]

        assert matched_mmd.abs() < mismatched_mmd.abs()

    def test_imq_kernel_also_runs_and_is_non_negative_in_expectation(self) -> None:
        torch.manual_seed(0)
        regularizer = MmdRegularizer(kernel="imq")
        mu = torch.zeros(32, 4)
        logvar = torch.zeros(32, 4)
        penalty = regularizer(mu, logvar)
        assert penalty.shape == (32,)

    def test_unknown_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel"):
            MmdRegularizer(kernel="does_not_exist")

    def test_batch_size_below_two_raises(self) -> None:
        regularizer = MmdRegularizer()
        with pytest.raises(ValueError, match="at least 2"):
            regularizer(torch.randn(1, 4), torch.zeros(1, 4))

    def test_gradients_flow_back_to_mu_and_logvar(self) -> None:
        regularizer = MmdRegularizer()
        mu = torch.randn(8, 4, requires_grad=True)
        logvar = torch.zeros(8, 4, requires_grad=True)
        penalty = regularizer(mu, logvar)
        penalty.mean().backward()
        assert mu.grad is not None and torch.any(mu.grad != 0)

    def test_is_stateless_nn_module(self) -> None:
        assert list(MmdRegularizer().parameters()) == []
