"""Unit tests for the latent regularizer registry pattern (spec §2.3, §10).

Mirrors the test style already used for the other self-registration
registries in this codebase (encoders, decoders, fusion, assemblers):
registration, lookup, duplicate-registration and unknown-name error
paths, plus a value-correctness check for the one built-in strategy,
`kl_standard_normal`.
"""

import pytest
import torch

from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer
from global_vae.losses.regularizers.registry import (
    getRegularizerClass,
    listRegisteredRegularizers,
    registerRegularizer,
)


def test_kl_standard_normal_is_registered_by_default() -> None:
    assert "kl_standard_normal" in listRegisteredRegularizers()
    assert getRegularizerClass("kl_standard_normal") is KlStandardNormalRegularizer


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
