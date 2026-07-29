"""Unit tests for `losses.reconstruction.computeTotalReconstructionLoss`."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction import computeTotalReconstructionLoss


def test_single_modality_defaults_to_mse() -> None:
    reconstruction = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss = computeTotalReconstructionLoss({"signal": reconstruction}, {"signal": target})
    expected = F.mse_loss(reconstruction, target)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_sums_across_modalities_of_different_shapes() -> None:
    signal_recon, signal_target = torch.randn(4, 10), torch.randn(4, 10)
    image_recon, image_target = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    loss = computeTotalReconstructionLoss(
        {"signal": signal_recon, "image": image_recon},
        {"signal": signal_target, "image": image_target},
    )
    expected = F.mse_loss(signal_recon, signal_target) + F.mse_loss(image_recon, image_target)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_per_modality_weight() -> None:
    reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
    loss = computeTotalReconstructionLoss(
        {"signal": reconstruction}, {"signal": target}, weights={"signal": 2.0}
    )
    expected = 2.0 * F.mse_loss(reconstruction, target)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_uniform_float_weight_applies_to_every_modality() -> None:
    signal_recon, signal_target = torch.randn(4, 10), torch.randn(4, 10)
    image_recon, image_target = torch.randn(4, 5), torch.randn(4, 5)
    loss = computeTotalReconstructionLoss(
        {"signal": signal_recon, "image": image_recon},
        {"signal": signal_target, "image": image_target},
        weights=0.5,
    )
    expected = 0.5 * (
        F.mse_loss(signal_recon, signal_target) + F.mse_loss(image_recon, image_target)
    )
    assert torch.allclose(loss, expected, atol=1e-5)


def test_shared_custom_loss_fn() -> None:
    reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
    loss = computeTotalReconstructionLoss(
        {"signal": reconstruction}, {"signal": target}, loss_fn=F.l1_loss
    )
    expected = F.l1_loss(reconstruction, target)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_per_modality_loss_fn() -> None:
    """A binary/segmentation-style target can use a different loss than a continuous one."""
    signal_recon, signal_target = torch.randn(4, 10), torch.randn(4, 10)
    mask_recon = torch.rand(4, 5)
    mask_target = torch.randint(0, 2, (4, 5)).float()
    loss = computeTotalReconstructionLoss(
        {"signal": signal_recon, "mask": mask_recon},
        {"signal": signal_target, "mask": mask_target},
        loss_fn={"signal": F.mse_loss, "mask": F.binary_cross_entropy},
    )
    expected = F.mse_loss(signal_recon, signal_target) + F.binary_cross_entropy(
        mask_recon, mask_target
    )
    assert torch.allclose(loss, expected, atol=1e-5)


def test_missing_loss_fn_for_a_modality_raises_key_error() -> None:
    reconstruction, target = torch.randn(4, 10), torch.randn(4, 10)
    with pytest.raises(KeyError):
        computeTotalReconstructionLoss(
            {"signal": reconstruction}, {"signal": target}, loss_fn={"other": F.mse_loss}
        )


def test_rejects_empty_reconstructions() -> None:
    with pytest.raises(ValueError):
        computeTotalReconstructionLoss({}, {})


def test_missing_target_raises_key_error() -> None:
    with pytest.raises(KeyError):
        computeTotalReconstructionLoss({"signal": torch.randn(2, 4)}, {})
