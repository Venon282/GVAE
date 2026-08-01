"""Unit tests for `losses.regularization.computeTotalRegularizationLoss`."""

import pytest
import torch
from torch import nn

from global_vae.losses.regularization import computeTotalRegularizationLoss
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)).mean()


def test_single_latent_space_defaults_to_unweighted_kl() -> None:
    regularizers = nn.ModuleDict({"z": KlStandardNormalRegularizer()})
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8)
    loss = computeTotalRegularizationLoss(regularizers, {"z": (mu, logvar)})
    assert torch.allclose(loss, _kl(mu, logvar), atol=1e-5)


def test_sums_across_latent_spaces() -> None:
    regularizers = nn.ModuleDict(
        {"z_a": KlStandardNormalRegularizer(), "z_b": KlStandardNormalRegularizer()}
    )
    mu_a, logvar_a = torch.randn(4, 8), torch.randn(4, 8)
    mu_b, logvar_b = torch.randn(4, 5), torch.randn(4, 5)
    loss = computeTotalRegularizationLoss(
        regularizers, {"z_a": (mu_a, logvar_a), "z_b": (mu_b, logvar_b)}
    )
    expected = _kl(mu_a, logvar_a) + _kl(mu_b, logvar_b)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_uniform_float_beta_applies_to_every_latent_space() -> None:
    regularizers = nn.ModuleDict(
        {"z_a": KlStandardNormalRegularizer(), "z_b": KlStandardNormalRegularizer()}
    )
    mu_a, logvar_a = torch.randn(4, 8), torch.randn(4, 8)
    mu_b, logvar_b = torch.randn(4, 5), torch.randn(4, 5)
    loss = computeTotalRegularizationLoss(
        regularizers, {"z_a": (mu_a, logvar_a), "z_b": (mu_b, logvar_b)}, beta=0.5
    )
    expected = 0.5 * (_kl(mu_a, logvar_a) + _kl(mu_b, logvar_b))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_per_latent_space_beta() -> None:
    regularizers = nn.ModuleDict(
        {"z_shared": KlStandardNormalRegularizer(), "z_private": KlStandardNormalRegularizer()}
    )
    mu_shared, logvar_shared = torch.randn(4, 8), torch.randn(4, 8)
    mu_private, logvar_private = torch.randn(4, 8), torch.randn(4, 8)
    loss = computeTotalRegularizationLoss(
        regularizers,
        {"z_shared": (mu_shared, logvar_shared), "z_private": (mu_private, logvar_private)},
        beta={"z_shared": 1.0, "z_private": 0.1},
    )
    expected = _kl(mu_shared, logvar_shared) + 0.1 * _kl(mu_private, logvar_private)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_subset_of_latent_spaces_is_accepted() -> None:
    """A latent space absent from `latent_params` this pass (spec §5) is simply skipped."""
    regularizers = nn.ModuleDict(
        {"z_a": KlStandardNormalRegularizer(), "z_b": KlStandardNormalRegularizer()}
    )
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8)
    loss = computeTotalRegularizationLoss(regularizers, {"z_a": (mu, logvar)})
    assert torch.allclose(loss, _kl(mu, logvar), atol=1e-5)


def test_rejects_empty_latent_params() -> None:
    regularizers = nn.ModuleDict({"z": KlStandardNormalRegularizer()})
    with pytest.raises(ValueError):
        computeTotalRegularizationLoss(regularizers, {})


def test_missing_regularizer_for_a_latent_space_raises_key_error() -> None:
    regularizers = nn.ModuleDict({"z_a": KlStandardNormalRegularizer()})
    with pytest.raises(KeyError, match="z_b"):
        computeTotalRegularizationLoss(
            regularizers, {"z_b": (torch.randn(2, 4), torch.randn(2, 4))}
        )
