"""Unit tests for `utils.conv_blocks` (spec §6, §7, §12).

`Residual1DBlock`/`Residual1DUpBlock` (1D, encoder/decoder side) and their 2D
generalizations `Residual2DBlock`/`Residual2DUpBlock` (spec §6's image modality,
`docs/adr/0018-2d-residual-encoder-decoder.md`) are plain, unregistered building blocks
(not modalities of their own), so these tests exercise them directly: shapes across
depth/stride combinations, gradient flow, the identity-vs-projection shortcut choice,
both decoder upsample modes, the two decoder-only output-suppression flags, and every
documented error path (`depth < 1`, an even `kernel_size` with `depth > 1`, and a
shortcut that cannot reach the main path's output length for every input length).

The 2D classes are covered by their own `TestResidual2DBlock`/`TestResidual2DUpBlock`
below. Beyond mirroring the 1D cases, they check what only exists once there are two
axes: non-square kernels/strides, a shortcut mismatch on a *single* axis (which a
symmetric test could not detect), and the "every input shape, not one example shape"
guarantee, exercised on odd and even sizes on each axis independently.
"""

import pytest
import torch

from global_vae.utils.conv_blocks import (
    Residual1DBlock,
    Residual1DUpBlock,
    Residual2DBlock,
    Residual2DUpBlock,
)
from global_vae.utils.conv_math import (
    computeConv1dOutputLength,
    computeConv2dOutputShape,
    computeConvTranspose1dOutputLength,
    computeConvTranspose2dOutputShape,
    computeUpsampleThenConv1dOutputLength,
    computeUpsampleThenConv2dOutputShape,
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


# --------------------------------------------------------------------------------------
# 2D counterparts (spec §6's image modality, docs/adr/0018-2d-residual-encoder-decoder.md)
# --------------------------------------------------------------------------------------

# Input shapes mixing odd and even sizes on each axis, so a shortcut that only lines up
# with its main path for *some* sizes (a floor-division coincidence) cannot go unnoticed.
_SWEEP_SHAPES = [(h, w) for h in (16, 17, 23, 32) for w in (15, 18, 24, 31)]


class TestResidual2DBlock:
    def test_output_shape_with_no_projection_needed(self) -> None:
        """Same channel width, stride (1, 1): the shortcut is a plain identity."""
        block = Residual2DBlock(in_channels=16, out_channels=16, depth=2, kernel_size=(3, 3))
        assert isinstance(block.shortcut, torch.nn.Identity)
        assert block.needs_projection is False
        y = block(torch.randn(4, 16, 20, 24))
        assert y.shape == (4, 16, 20, 24)

    def test_output_shape_with_channel_change_projection(self) -> None:
        block = Residual2DBlock(in_channels=1, out_channels=8, depth=2, kernel_size=(3, 3))
        assert not isinstance(block.shortcut, torch.nn.Identity)
        assert block.needs_projection is True
        y = block(torch.randn(4, 1, 20, 24))
        assert y.shape == (4, 8, 20, 24)

    def test_output_shape_with_non_square_stride_projection(self) -> None:
        block = Residual2DBlock(
            in_channels=8, out_channels=8, depth=2, kernel_size=(3, 3), stride=(2, 1)
        )
        assert not isinstance(block.shortcut, torch.nn.Identity)
        y = block(torch.randn(4, 8, 40, 30))
        assert y.shape == (
            4,
            8,
            *computeConv2dOutputShape((40, 30), (3, 3), (2, 1), (1, 1), (1, 1)),
        )
        assert y.shape[-2:] == (20, 30)

    def test_non_square_kernel_keeps_the_shape_with_default_padding(self) -> None:
        """Default padding is per axis: (3, 5) -> (1, 2), which preserves both axes."""
        block = Residual2DBlock(in_channels=4, out_channels=8, depth=3, kernel_size=(3, 5))
        y = block(torch.randn(2, 4, 21, 26))
        assert y.shape == (2, 8, 21, 26)

    def test_flexible_block_depths_three_layers(self) -> None:
        """Spec's own example: "3 layers before the connection"."""
        block = Residual2DBlock(
            in_channels=8, out_channels=16, depth=3, kernel_size=(3, 3), stride=(2, 2)
        )
        # every non-last layer contributes conv+norm+act (3 modules); the last layer
        # contributes conv+norm only (activation deferred until after the residual add).
        assert len(block.main) == 2 * 3 + 2
        y = block(torch.randn(2, 8, 40, 36))
        assert y.shape == (2, 16, 20, 18)

    def test_flexible_block_depths_four_layers(self) -> None:
        """...then 4" for a different (here: deeper) stage."""
        block = Residual2DBlock(in_channels=16, out_channels=16, depth=4, kernel_size=(3, 3))
        y = block(torch.randn(2, 16, 24, 20))
        assert y.shape == (2, 16, 24, 20)

    def test_depth_one_is_a_single_projection_layer_plus_shortcut(self) -> None:
        block = Residual2DBlock(
            in_channels=8, out_channels=16, depth=1, kernel_size=(3, 3), stride=(2, 2)
        )
        y = block(torch.randn(2, 8, 40, 36))
        assert y.shape == (2, 16, 20, 18)

    def test_depth_one_allows_an_even_kernel_size(self) -> None:
        """No internal layer exists at depth=1, so the odd-kernel constraint does not apply,
        as long as the (still-required) shortcut can match the main path's offsets."""
        block = Residual2DBlock(
            in_channels=8,
            out_channels=16,
            depth=1,
            kernel_size=(4, 4),
            padding=(2, 2),
            shortcut_kernel_size=(2, 2),
        )
        y = block(torch.randn(2, 8, 20, 22))
        assert y.shape[1] == 16
        assert y.shape[-2:] == computeConv2dOutputShape((20, 22), (4, 4), (1, 1), (2, 2), (1, 1))

    def test_shortcut_reaches_the_main_path_shape_for_every_input_shape(self) -> None:
        """The guarantee the class docstring makes: not one example shape, every shape.
        A mismatch would make the residual `+` raise, so a clean sweep over odd and even
        sizes on each axis is the direct check."""
        block = Residual2DBlock(
            in_channels=4, out_channels=8, depth=2, kernel_size=(3, 5), stride=(2, 3)
        )
        block.eval()
        for height, width in _SWEEP_SHAPES:
            y = block(torch.randn(1, 4, height, width))
            assert y.shape[-2:] == computeConv2dOutputShape(
                (height, width), (3, 5), (2, 3), (1, 2), (1, 1)
            )

    def test_dilation_is_honored_by_the_default_padding(self) -> None:
        block = Residual2DBlock(
            in_channels=4, out_channels=4, depth=2, kernel_size=(3, 3), dilation=(2, 1)
        )
        y = block(torch.randn(2, 4, 20, 20))
        assert y.shape == (2, 4, 20, 20)

    def test_gradients_reach_every_parameter(self) -> None:
        block = Residual2DBlock(
            in_channels=4, out_channels=8, depth=3, kernel_size=(3, 3), stride=(2, 2)
        )
        y = block(torch.randn(3, 4, 30, 30))
        y.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_activation_and_normalization_can_be_disabled(self) -> None:
        block = Residual2DBlock(
            in_channels=8,
            out_channels=8,
            depth=2,
            kernel_size=(3, 3),
            activation=None,
            normalization=None,
        )
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in block.modules())
        assert isinstance(block.output_activation, torch.nn.Identity)

    def test_uses_batchnorm2d_by_default(self) -> None:
        block = Residual2DBlock(in_channels=4, out_channels=8, depth=2)
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in block.main.modules())
        assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in block.modules())

    def test_non_positive_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            Residual2DBlock(in_channels=8, out_channels=8, depth=0)

    @pytest.mark.parametrize("kernel_size", [(4, 4), (3, 4), (4, 3)])
    def test_even_kernel_size_on_either_axis_with_depth_greater_than_one_raises(
        self, kernel_size: tuple[int, int]
    ) -> None:
        """Both axes must be odd independently: an even size on just one is enough."""
        with pytest.raises(ValueError, match="odd kernel_size"):
            Residual2DBlock(in_channels=8, out_channels=8, depth=2, kernel_size=kernel_size)

    def test_mismatched_shortcut_offset_raises_with_actionable_message(self) -> None:
        with pytest.raises(ValueError, match="length offset"):
            Residual2DBlock(
                in_channels=8, out_channels=16, depth=1, kernel_size=(4, 4), padding=(1, 1)
            )

    def test_shortcut_mismatch_on_a_single_axis_is_still_detected(self) -> None:
        """Height offset is 0 (kernel 3, padding 1), width offset is -1 (kernel 4, padding
        1): a check that only looked at one axis, or at their sum-symmetry, would pass."""
        with pytest.raises(ValueError, match="length offset"):
            Residual2DBlock(
                in_channels=8, out_channels=16, depth=1, kernel_size=(3, 4), padding=(1, 1)
            )

    def test_mismatch_is_not_raised_when_no_projection_is_needed(self) -> None:
        """An identity shortcut has nothing to reconcile, so an otherwise-mismatched
        padding is not this class's job to reject (the residual add would simply be
        applied to whatever shape the main path produces)."""
        block = Residual2DBlock(
            in_channels=8, out_channels=8, depth=1, kernel_size=(3, 3), padding=(1, 1)
        )
        assert isinstance(block.shortcut, torch.nn.Identity)


class TestResidual2DUpBlock:
    def test_conv_transpose_mode_output_shape(self) -> None:
        block = Residual2DUpBlock(
            in_channels=32,
            out_channels=16,
            depth=2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            upsample_mode="conv_transpose",
        )
        y = block(torch.randn(4, 32, 8, 6))
        assert y.shape == (
            4,
            16,
            *computeConvTranspose2dOutputShape((8, 6), (3, 3), (2, 2), (1, 1), (1, 1), (1, 1)),
        )

    def test_interpolate_conv_mode_output_shape(self) -> None:
        block = Residual2DUpBlock(
            in_channels=32,
            out_channels=16,
            depth=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            upsample_mode="interpolate_conv",
        )
        y = block(torch.randn(4, 32, 8, 6))
        assert y.shape == (
            4,
            16,
            *computeUpsampleThenConv2dOutputShape((8, 6), (2, 2), (3, 3), (1, 1), (1, 1)),
        )

    def test_non_square_stride_and_kernel_in_conv_transpose_mode(self) -> None:
        block = Residual2DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=2,
            kernel_size=(3, 5),
            stride=(2, 3),
            padding=(1, 2),
            output_padding=(1, 2),
            upsample_mode="conv_transpose",
        )
        y = block(torch.randn(2, 16, 7, 5))
        assert y.shape[-2:] == computeConvTranspose2dOutputShape(
            (7, 5), (3, 5), (2, 3), (1, 2), (1, 2), (1, 1)
        )

    def test_non_square_stride_in_interpolate_conv_mode(self) -> None:
        block = Residual2DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=2,
            kernel_size=(3, 3),
            stride=(3, 1),
            padding=(1, 1),
            upsample_mode="interpolate_conv",
        )
        y = block(torch.randn(2, 16, 5, 9))
        assert y.shape == (2, 8, 15, 9)

    def test_flexible_block_depths(self) -> None:
        shallow = Residual2DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        deep = Residual2DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=4,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        x = torch.randn(2, 16, 10, 12)
        assert shallow(x).shape == deep(x).shape

    def test_no_projection_needed_uses_identity_shortcut(self) -> None:
        block = Residual2DUpBlock(
            in_channels=16,
            out_channels=16,
            depth=2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        assert isinstance(block.shortcut, torch.nn.Identity)
        assert block.needs_projection is False

    @pytest.mark.parametrize("mode", ["conv_transpose", "interpolate_conv"])
    def test_shortcut_reaches_the_main_path_shape_for_every_input_shape(self, mode: str) -> None:
        block = Residual2DUpBlock(
            in_channels=4,
            out_channels=8,
            depth=2,
            kernel_size=(3, 5),
            stride=(2, 3),
            padding=(1, 2),
            output_padding=(1, 2) if mode == "conv_transpose" else (0, 0),
            upsample_mode=mode,
        )
        block.eval()
        for height, width in [(h, w) for h in (4, 5, 8) for w in (3, 6, 7)]:
            y = block(torch.randn(1, 4, height, width))
            if mode == "conv_transpose":
                expected = computeConvTranspose2dOutputShape(
                    (height, width), (3, 5), (2, 3), (1, 2), (1, 2), (1, 1)
                )
            else:
                expected = computeUpsampleThenConv2dOutputShape(
                    (height, width), (2, 3), (3, 5), (1, 2), (1, 1)
                )
            assert y.shape[-2:] == expected

    @pytest.mark.parametrize("mode", ["conv_transpose", "interpolate_conv"])
    def test_gradients_reach_every_parameter(self, mode: str) -> None:
        block = Residual2DUpBlock(
            in_channels=16,
            out_channels=8,
            depth=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1) if mode == "conv_transpose" else (0, 0),
            upsample_mode=mode,
        )
        y = block(torch.randn(2, 16, 6, 6))
        y.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_output_flags_suppress_only_the_final_layer(self) -> None:
        """Internal layers of a deep block must stay normalized/activated even when the
        block's own final output is suppressed (the "last transition" convention)."""
        block = Residual2DUpBlock(
            in_channels=32,
            out_channels=1,
            depth=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            upsample_mode="interpolate_conv",
            apply_output_normalization=False,
            apply_output_activation=False,
        )
        assert isinstance(block.output_activation, torch.nn.Identity)
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in block.main.modules())
        y = block(torch.randn(4, 32, 8, 8))
        assert bool((y < 0).any())  # unconstrained: negative values must survive

    def test_output_flags_also_skip_the_shortcuts_normalization(self) -> None:
        block = Residual2DUpBlock(
            in_channels=8,
            out_channels=4,
            depth=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            upsample_mode="interpolate_conv",
            apply_output_normalization=False,
        )
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in block.shortcut.modules())

    def test_apply_output_flags_default_to_true(self) -> None:
        block = Residual2DUpBlock(
            in_channels=32,
            out_channels=1,
            depth=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            upsample_mode="interpolate_conv",
        )
        assert not isinstance(block.output_activation, torch.nn.Identity)

    def test_non_positive_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            Residual2DUpBlock(in_channels=8, out_channels=8, depth=0)

    @pytest.mark.parametrize("kernel_size", [(4, 4), (3, 4), (4, 3)])
    def test_even_kernel_size_on_either_axis_with_depth_greater_than_one_raises(
        self, kernel_size: tuple[int, int]
    ) -> None:
        with pytest.raises(ValueError, match="odd kernel_size"):
            Residual2DUpBlock(
                in_channels=8,
                out_channels=8,
                depth=2,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
            )

    def test_unknown_upsample_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="upsample_mode"):
            Residual2DUpBlock(in_channels=8, out_channels=8, depth=1, upsample_mode="magic")

    def test_infeasible_conv_transpose_shortcut_output_padding_raises(self) -> None:
        with pytest.raises(ValueError, match="output_padding"):
            Residual2DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(0, 0),
                upsample_mode="conv_transpose",
            )

    def test_infeasible_conv_transpose_shortcut_on_a_single_axis_is_still_detected(self) -> None:
        """Axis 0 (kernel 3) is feasible; only axis 1 (kernel 6) is not."""
        with pytest.raises(ValueError, match="output_padding"):
            Residual2DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=(3, 6),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(0, 0),
                upsample_mode="conv_transpose",
            )

    def test_mismatched_interpolate_conv_shortcut_offset_raises(self) -> None:
        with pytest.raises(ValueError, match="length offset"):
            Residual2DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                upsample_mode="interpolate_conv",
            )

    def test_mismatched_interpolate_conv_shortcut_on_a_single_axis_is_still_detected(self) -> None:
        with pytest.raises(ValueError, match="length offset"):
            Residual2DUpBlock(
                in_channels=8,
                out_channels=16,
                depth=1,
                kernel_size=(3, 4),
                stride=(2, 2),
                padding=(1, 1),
                upsample_mode="interpolate_conv",
            )
