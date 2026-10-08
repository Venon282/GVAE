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


class TestShapeGuard:
    """A reconstruction and its target must have the same shape (roadmap P0-4(g)).

    `F.mse_loss` and its siblings only warn when the shapes differ, then broadcast. For a
    `(B, H, W)` reconstruction against a `(B, 1, H, W)` target that means a `(B, B, H, W)`
    comparison: every reconstruction scored against every target in the batch, and a finite,
    plausible-looking loss that trains the wrong thing. The guard turns that into an error.
    """

    def test_channel_axis_mismatch_from_the_audit_raises(self) -> None:
        """The 2D decoder's `(B, H, W)` output against a `(B, 1, H, W)` target is rejected."""
        batch_size, height, width = 4, 8, 8
        with pytest.raises(ValueError, match=r"\(4, 8, 8\).*\(4, 1, 8, 8\)"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(batch_size, height, width)},
                {"image": torch.randn(batch_size, 1, height, width)},
            )

    def test_message_names_the_modality_and_both_shapes(self) -> None:
        """The message says which reconstruction is wrong and what each shape is."""
        with pytest.raises(ValueError) as excinfo:
            computeTotalReconstructionLoss(
                {"spectrum": torch.randn(4, 16)}, {"spectrum": torch.randn(4, 1, 16)}
            )
        message = str(excinfo.value)
        assert "'spectrum'" in message
        assert "(4, 16)" in message
        assert "(4, 1, 16)" in message

    def test_vector_against_column_raises(self) -> None:
        """`(B,)` against `(B, 1)` would broadcast to `(B, B)`, so it is rejected."""
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss({"score": torch.randn(5)}, {"score": torch.randn(5, 1)})

    def test_same_number_of_elements_in_another_layout_raises(self) -> None:
        """Equal element counts are not enough: `(4, 6)` is not `(4, 2, 3)`."""
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"signal": torch.randn(4, 6)}, {"signal": torch.randn(4, 2, 3)}
            )

    def test_batch_size_mismatch_raises(self) -> None:
        """A target from a different number of samples is rejected."""
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"signal": torch.randn(4, 10)}, {"signal": torch.randn(3, 10)}
            )

    def test_one_bad_modality_among_several_is_named(self) -> None:
        """The modality with the mismatch is the one in the message."""
        with pytest.raises(ValueError, match="'image'") as excinfo:
            computeTotalReconstructionLoss(
                {"signal": torch.randn(4, 10), "image": torch.randn(4, 8, 8)},
                {"signal": torch.randn(4, 10), "image": torch.randn(4, 1, 8, 8)},
            )
        assert "'signal'" not in str(excinfo.value)

    def test_matching_multi_dimensional_shapes_are_accepted(self) -> None:
        """The same `(B, H, W)` on both sides is a normal call."""
        reconstruction, target = torch.randn(4, 8, 8), torch.randn(4, 8, 8)
        loss = computeTotalReconstructionLoss({"image": reconstruction}, {"image": target})
        assert torch.allclose(loss, F.mse_loss(reconstruction, target), atol=1e-5)

    def test_squeezing_the_target_makes_the_audit_case_valid(self) -> None:
        """The fix the message asks for works: drop the channel axis from the target."""
        reconstruction = torch.randn(4, 8, 8)
        target = torch.randn(4, 1, 8, 8)
        loss = computeTotalReconstructionLoss(
            {"image": reconstruction}, {"image": target.squeeze(1)}
        )
        assert torch.allclose(loss, F.mse_loss(reconstruction, target.squeeze(1)), atol=1e-5)

    def test_guard_applies_to_a_shared_custom_loss_fn(self) -> None:
        """It is not specific to `mse_loss`: `l1_loss` is guarded as well."""
        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(4, 8, 8)},
                {"image": torch.randn(4, 1, 8, 8)},
                loss_fn=F.l1_loss,
            )

    def test_guard_applies_to_a_per_modality_loss_fn_dict(self) -> None:
        """A per-modality `loss_fn` dict does not bypass the check."""
        with pytest.raises(ValueError, match="'image'"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(4, 8, 8)},
                {"image": torch.randn(4, 1, 8, 8)},
                loss_fn={"image": F.mse_loss},
            )

    def test_guard_runs_before_the_loss_function(self) -> None:
        """The loss function is never called on tensors that do not line up."""
        calls: list[tuple[torch.Size, torch.Size]] = []

        def recordingLoss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            calls.append((reconstruction.shape, target.shape))
            return F.mse_loss(reconstruction, target)

        with pytest.raises(ValueError, match="shape"):
            computeTotalReconstructionLoss(
                {"image": torch.randn(4, 8, 8)},
                {"image": torch.randn(4, 1, 8, 8)},
                loss_fn=recordingLoss,
            )
        assert calls == []
