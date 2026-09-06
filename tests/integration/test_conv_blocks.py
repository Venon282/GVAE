"""Unit tests for `utils.conv_blocks` (spec §6, §7, §12).

`Residual1DBlock` (encoder side) and `Residual1DUpBlock` (decoder side)
are plain, unregistered building blocks (not modalities of their own), so
these tests exercise them directly: shapes across depth/stride
combinations, gradient flow, the identity-vs-projection shortcut choice,
both decoder upsample modes, the two decoder-only output-suppression
flags, and every documented error path (`depth < 1`, an even `kernel_size`
with `depth > 1`, and a shortcut that cannot reach the main path's output
length for every input length).
"""

import pytest
import torch

from global_vae.utils.conv_blocks import Residual1DBlock, Residual1DUpBlock
from global_vae.utils.conv_math import (
    computeConv1dOutputLength,
    computeConvTranspose1dOutputLength,
    computeUpsampleThenConv1dOutputLength,
)


class TestResidual1DBlock:
    def test_output_shape_with_no_projection_needed(self) -> None:
        """Same channel width, stride 1: the shortcut is a plain identity."""
        block = Residual1DBlock(in_channels=16, out_channels=16, depth=2, kernel_size=3, stride=1)
        assert isinstance(block.shortcut, torch.nn.Identity)
        y = block(torch.randn(4, 16, 50))
        assert y.shape == (4, 16, computeConv1dOutputLength(50, 3, 1, 1, 1))

    def test_output_shape_with_channel_change_projection(self) -> None:
        block = Residual1DBlock(in_channels=1, out_channels=8, depth=2, kernel_size=3, stride=1)
        assert not isinstance(block.shortcut, torch.nn.Identity)
        y = block(torch.randn(4, 1, 50))
        assert y.shape == (4, 8, computeConv1dOutputLength(50, 3, 1, 1, 1))

    def test_output_shape_with_stride_projection(self) -> None:
        block = Residual1DBlock(in_channels=8, out_channels=8, depth=2, kernel_size=3, stride=2)
        assert not isinstance(block.shortcut, torch.nn.Identity)
        y = block(torch.randn(4, 8, 50))
        assert y.shape == (4, 8, computeConv1dOutputLength(50, 3, 2, 1, 1))

    def test_flexible_block_depths_three_layers(self) -> None:
        """Spec's own example: "3 layers before the connection"."""
        block = Residual1DBlock(in_channels=8, out_channels=16, depth=3, kernel_size=3, stride=2)
        # every non-last layer contributes conv+norm+act (3 modules); the last layer
        # contributes conv+norm only (activation deferred until after the residual add).
        assert len(block.main) == 2 * 3 + 2
        y = block(torch.randn(2, 8, 40))
        assert y.shape == (2, 16, computeConv1dOutputLength(40, 3, 2, 1, 1))

    def test_flexible_block_depths_four_layers(self) -> None:
        """...then 4" for a different (here: deeper) stage."""
        block = Residual1DBlock(in_channels=16, out_channels=16, depth=4, kernel_size=3, stride=1)
        y = block(torch.randn(2, 16, 40))
        assert y.shape == (2, 16, 40)

    def test_depth_one_is_a_single_projection_layer_plus_shortcut(self) -> None:
        block = Residual1DBlock(in_channels=8, out_channels=16, depth=1, kernel_size=3, stride=2)
        y = block(torch.randn(2, 8, 40))
        assert y.shape == (2, 16, computeConv1dOutputLength(40, 3, 2, 1, 1))

    def test_depth_one_allows_an_even_kernel_size(self) -> None:
        """No internal layer exists at depth=1, so the odd-kernel constraint does not apply,
        as long as the (still-required) shortcut can match the main path's length."""
        block = Residual1DBlock(
            in_channels=8,
            out_channels=16,
            depth=1,
            kernel_size=4,
            stride=1,
            padding=2,
            shortcut_kernel_size=2,
        )
        y = block(torch.randn(2, 8, 40))
        assert y.shape[1] == 16

    def test_gradients_reach_every_parameter(self) -> None:
        block = Residual1DBlock(in_channels=4, out_channels=8, depth=3, kernel_size=3, stride=2)
        y = block(torch.randn(3, 4, 30))
        y.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_activation_and_normalization_can_be_disabled(self) -> None:
        block = Residual1DBlock(
            in_channels=8,
            out_channels=8,
            depth=2,
            kernel_size=3,
            activation=None,
            normalization=None,
        )
        assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in block.modules())
        assert isinstance(block.output_activation, torch.nn.Identity)

    def test_non_positive_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            Residual1DBlock(in_channels=8, out_channels=8, depth=0)

    def test_even_kernel_size_with_depth_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="odd kernel_size"):
            Residual1DBlock(in_channels=8, out_channels=8, depth=2, kernel_size=4)

    def test_mismatched_shortcut_offset_raises_with_actionable_message(self) -> None:
        """kernel_size=4 with depth=1 skips the odd-kernel check, but the default 1x1
        shortcut still cannot reach the same length as this main path for every input
        length once a projection is actually needed (channel change here)."""
        with pytest.raises(ValueError, match="length offset"):
            Residual1DBlock(in_channels=8, out_channels=16, depth=1, kernel_size=4, padding=1)


class TestResidual1DUpBlock:
    def test_conv_transpose_mode_output_shape(self) -> None:
        block = Residual1DUpBlock(
            in_channels=32,
            out_channels=16,
            depth=2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            upsample_mode="conv_transpose",
        )
        y = block(torch.randn(4, 32, 8))
        assert y.shape == (4, 16, computeConvTranspose1dOutputLength(8, 3, 2, 1, 1, 1))

    def test_interpolate_conv_mode_output_shape(self) -> None:
        block = Residual1DUpBlock(
            in_channels=32,
            out_channels=16,
            depth=3,
            kernel_size=3,
            stride=2,
            padding=1,
            upsample_mode="interpolate_conv",
        )
        y = block(torch.randn(4, 32, 8))
        assert y.shape == (4, 16, computeUpsampleThenConv1dOutputLength(8, 2, 3, 1, 1))

    def test_flexible_block_depths(self) -> None:
        shallow = Residual1DUpBlock(
            in_channels=16, out_channels=8, depth=1, kernel_size=3, stride=2, padding=1
        )
        deep = Residual1DUpBlock(
            in_channels=16, out_channels=8, depth=4, kernel_size=3, stride=2, padding=1
        )
        x = torch.randn(2, 16, 10)
        assert shallow(x).shape == deep(x).shape

    def test_no_projection_needed_uses_identity_shortcut(self) -> None:
        block = Residual1DUpBlock(
            in_channels=16,
            out_channels=16,
            depth=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        assert isinstance(block.shortcut, torch.nn.Identity)

    def test_gradients_reach_every_parameter_conv_transpose(self) -> None:
        block = Residual1DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            upsample_mode="conv_transpose",
        )
        y = block(torch.randn(2, 16, 6))
        y.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_gradients_reach_every_parameter_interpolate_conv(self) -> None:
        block = Residual1DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=3,
            kernel_size=3,
            stride=2,
            padding=1,
            upsample_mode="interpolate_conv",
        )
        y = block(torch.randn(2, 16, 6))
        y.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_output_flags_suppress_only_the_final_layer(self) -> None:
        """Internal layers of a deep block must stay normalized/activated even when the
        block's own final output is suppressed (the "last transition" convention)."""
        block = Residual1DUpBlock(
            in_channels=32,
            out_channels=1,
            depth=3,
            kernel_size=3,
            stride=2,
            padding=1,
            upsample_mode="interpolate_conv",
            apply_output_normalization=False,
            apply_output_activation=False,
        )
        assert isinstance(block.output_activation, torch.nn.Identity)
        assert any(isinstance(m, torch.nn.BatchNorm1d) for m in block.main.modules())
        y = block(torch.randn(4, 32, 8))
        assert bool((y < 0).any())  # unconstrained: negative values must survive

    def test_apply_output_flags_default_to_true(self) -> None:
        block = Residual1DUpBlock(
            in_channels=32,
            out_channels=1,
            depth=1,
            kernel_size=3,
            stride=2,
            padding=1,
            upsample_mode="interpolate_conv",
        )
        assert not isinstance(block.output_activation, torch.nn.Identity)

    def test_non_positive_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            Residual1DUpBlock(in_channels=8, out_channels=8, depth=0)

    def test_even_kernel_size_with_depth_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="odd kernel_size"):
            Residual1DUpBlock(
                in_channels=8, out_channels=8, depth=2, kernel_size=4, stride=1, padding=1
            )

    def test_unknown_upsample_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="upsample_mode"):
            Residual1DUpBlock(in_channels=8, out_channels=8, depth=1, upsample_mode="magic")

    def test_infeasible_conv_transpose_shortcut_output_padding_raises(self) -> None:
        with pytest.raises(ValueError, match="output_padding"):
            Residual1DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=6,
                stride=2,
                padding=1,
                output_padding=0,
                upsample_mode="conv_transpose",
            )

    def test_mismatched_interpolate_conv_shortcut_offset_raises(self) -> None:
        with pytest.raises(ValueError, match="length offset"):
            Residual1DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=4,
                stride=2,
                padding=1,
                upsample_mode="interpolate_conv",
            )
