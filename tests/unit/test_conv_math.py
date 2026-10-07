"""Unit tests for `utils.conv_math` (spec §6, §12).

`conv_math` is the one place in this codebase where `Conv`/`ConvTranspose`/pooling shape
arithmetic is written down; every encoder, decoder, and residual block delegates to it
(`OneDCnnEncoder.computeMinimumInputLength`, `OneDCnnDecoder.computeOutputLength`,
`TwoDCnnDecoder.computeOutputShape`, `Residual1DBlock`/`Residual2DBlock`'s shortcut
verification, ...). A wrong formula here would not fail loudly: it would silently make
those classes accept a configuration that produces the wrong shape, or reject a valid
one. So these tests deliberately do not only compare the functions against
hand-computed values (which would merely re-state the same formula): wherever a real
PyTorch module exists for the operation, the function's result is checked against the
shape that module *actually produces* on a real tensor.

Structure:

- 1D functions, each cross-checked against `nn.Conv1d`/`nn.ConvTranspose1d`/
  `nn.MaxPool1d`/`nn.Upsample` + `nn.Conv1d`.
- The length-offset helpers, whose defining property (equal offsets imply equal output
  length for *every* input length, the guarantee a residual shortcut relies on) is
  checked over a sweep of input lengths, not a single one.
- The 2D counterparts, which are documented as "the 1D function applied once per axis"
  (`conv_math`'s own module-level note): checked against `nn.Conv2d` etc. on non-square
  configurations where the two axes disagree, so an accidental axis mix-up cannot pass.
"""

import itertools

import pytest
import torch
from torch import nn

from global_vae.utils.conv_math import (
    computeConv1dLengthOffset,
    computeConv1dOutputLength,
    computeConv2dLengthOffset,
    computeConv2dOutputShape,
    computeConvTranspose1dLengthOffset,
    computeConvTranspose1dOutputLength,
    computeConvTranspose2dLengthOffset,
    computeConvTranspose2dOutputShape,
    computeUpsampleStack2dOutputShape,
    computeUpsampleStackOutputLength,
    computeUpsampleThenConv1dOutputLength,
    computeUpsampleThenConv2dOutputShape,
    isLengthPreservingConv1d,
    solveConvTranspose1dOutputPadding,
    solveConvTranspose2dOutputPadding,
    solveMinimumInputLengthForConv1d,
    solveMinimumInputShapeForConv2d,
)

# (kernel_size, stride, padding, dilation) sweeps kept small enough to stay fast, large
# enough to include even kernels, strides above 1, and dilations above 1.
_CONV_1D_CASES = list(itertools.product((1, 2, 3, 4, 5, 7), (1, 2, 3), (0, 1, 2, 3), (1, 2, 3)))
_INPUT_LENGTHS = (12, 13, 20, 33)


def _actualConv1dLength(
    length: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int | None:
    """Output length of a real `nn.Conv1d`, or `None` if PyTorch itself rejects the input."""
    conv = nn.Conv1d(1, 1, kernel_size, stride=stride, padding=padding, dilation=dilation)
    try:
        return int(conv(torch.zeros(1, 1, length)).shape[-1])
    except RuntimeError:
        return None


class TestConv1dOutputLength:
    @pytest.mark.parametrize(("kernel_size", "stride", "padding", "dilation"), _CONV_1D_CASES)
    def test_matches_a_real_conv1d(
        self, kernel_size: int, stride: int, padding: int, dilation: int
    ) -> None:
        for length in _INPUT_LENGTHS:
            actual = _actualConv1dLength(length, kernel_size, stride, padding, dilation)
            if actual is None:
                # The formula would give a length <= 0: PyTorch refuses the input outright.
                assert (
                    computeConv1dOutputLength(length, kernel_size, stride, padding, dilation) <= 0
                )
                continue
            assert (
                computeConv1dOutputLength(length, kernel_size, stride, padding, dilation) == actual
            )

    def test_hand_computed_value(self) -> None:
        # (32 + 2 * 1 - 1 * (5 - 1) - 1) // 2 + 1 = 29 // 2 + 1 = 15
        assert computeConv1dOutputLength(32, 5, 2, 1, 1) == 15


class TestConvTranspose1dOutputLength:
    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation"),
        list(itertools.product((1, 2, 3, 4, 5), (1, 2, 3), (0, 1, 2), (1, 2))),
    )
    def test_matches_a_real_convtranspose1d(
        self, kernel_size: int, stride: int, padding: int, dilation: int
    ) -> None:
        for output_padding in range(max(stride, dilation)):
            module = nn.ConvTranspose1d(
                1,
                1,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
            )
            for length in (4, 9):
                try:
                    actual = int(module(torch.zeros(1, 1, length)).shape[-1])
                except RuntimeError:
                    continue  # padding too large for this kernel: not a valid module
                assert (
                    computeConvTranspose1dOutputLength(
                        length, kernel_size, stride, padding, output_padding, dilation
                    )
                    == actual
                )

    def test_default_doubling_configuration(self) -> None:
        """kernel 4 / stride 2 / padding 1 doubles the length, the documented default of
        `OneDCnnDecoder` under conv_transpose mode."""
        assert computeConvTranspose1dOutputLength(8, 4, 2, 1, 0, 1) == 16


class TestUpsampleThenConv1dOutputLength:
    @pytest.mark.parametrize(
        ("scale_factor", "kernel_size", "padding", "dilation"),
        list(itertools.product((1, 2, 3), (1, 3, 4, 5), (0, 1, 2), (1, 2))),
    )
    def test_matches_a_real_upsample_then_conv1d(
        self, scale_factor: int, kernel_size: int, padding: int, dilation: int
    ) -> None:
        module = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv1d(1, 1, kernel_size, padding=padding, dilation=dilation),
        )
        for length in (6, 11):
            try:
                actual = int(module(torch.zeros(1, 1, length)).shape[-1])
            except RuntimeError:
                continue
            assert (
                computeUpsampleThenConv1dOutputLength(
                    length, scale_factor, kernel_size, padding, dilation
                )
                == actual
            )

    def test_kernel_three_padding_one_doubles_exactly(self) -> None:
        """The "same"-padded pairing the decoders' docstrings name for interpolate_conv."""
        assert computeUpsampleThenConv1dOutputLength(8, 2, 3, 1, 1) == 16


class TestSolveConvTranspose1dOutputPadding:
    @pytest.mark.parametrize("output_padding", [0, 1, 2])
    def test_round_trips_the_output_padding_it_was_derived_from(self, output_padding: int) -> None:
        kernel_size, stride, padding, dilation = 3, 3, 1, 1
        target = computeConvTranspose1dOutputLength(
            7, kernel_size, stride, padding, output_padding, dilation
        )
        assert (
            solveConvTranspose1dOutputPadding(7, target, kernel_size, stride, padding, dilation)
            == output_padding
        )

    def test_can_return_a_value_pytorch_would_reject(self) -> None:
        """Documented: the solver only closes the arithmetic gap; validity against
        `0 <= output_padding < max(stride, dilation)` is the caller's check."""
        gap_of_five = computeConvTranspose1dOutputLength(8, 4, 2, 1, 0, 1) + 5
        assert solveConvTranspose1dOutputPadding(8, gap_of_five, 4, 2, 1, 1) == 5

    def test_negative_when_the_target_is_below_the_natural_length(self) -> None:
        natural = computeConvTranspose1dOutputLength(8, 4, 2, 1, 0, 1)
        assert solveConvTranspose1dOutputPadding(8, natural - 3, 4, 2, 1, 1) == -3


class TestSolveMinimumInputLengthForConv1d:
    @pytest.mark.parametrize(("kernel_size", "stride", "padding", "dilation"), _CONV_1D_CASES)
    def test_is_the_exact_minimum(
        self, kernel_size: int, stride: int, padding: int, dilation: int
    ) -> None:
        for target in (1, 2, 5):
            minimum = solveMinimumInputLengthForConv1d(
                target, kernel_size, stride, padding, dilation
            )
            assert minimum >= 1
            assert (
                computeConv1dOutputLength(minimum, kernel_size, stride, padding, dilation) >= target
            )
            if minimum > 1:
                # one shorter must fall short: this is what makes it the *minimum*
                assert (
                    computeConv1dOutputLength(minimum - 1, kernel_size, stride, padding, dilation)
                    < target
                )

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding"), [(2, 2, 0), (3, 2, 1), (4, 3, 0)]
    )
    def test_also_applies_to_max_pooling(self, kernel_size: int, stride: int, padding: int) -> None:
        """Documented: pooling layers share the convolution length formula (dilation=1)."""
        minimum = solveMinimumInputLengthForConv1d(1, kernel_size, stride, padding, 1)
        pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        assert pool(torch.zeros(1, 1, minimum)).shape[-1] >= 1
        if minimum > 1:
            with pytest.raises(RuntimeError):
                pool(torch.zeros(1, 1, minimum - 1))

    def test_never_below_one(self) -> None:
        """A 1x1 convolution with heavy padding would mathematically allow a length <= 0."""
        assert solveMinimumInputLengthForConv1d(1, 1, 1, 5, 1) == 1


class TestConv1dLengthOffset:
    @pytest.mark.parametrize(
        ("kernel_size", "padding", "dilation"),
        list(itertools.product((1, 2, 3, 4, 5, 7), (0, 1, 2, 3), (1, 2, 3))),
    )
    def test_stride_one_output_is_input_plus_offset_for_every_length(
        self, kernel_size: int, padding: int, dilation: int
    ) -> None:
        offset = computeConv1dLengthOffset(kernel_size, padding, dilation)
        for length in range(20, 40):
            assert (
                computeConv1dOutputLength(length, kernel_size, 1, padding, dilation)
                == length + offset
            )

    def test_equal_offsets_give_equal_lengths_for_every_input_length(self) -> None:
        """The guarantee a residual shortcut needs against its main path: kernel 3 /
        padding 1 and kernel 1 / padding 0 are both offset 0, so they agree at every
        length; kernel 3 / padding 0 (offset -2) never does."""
        assert computeConv1dLengthOffset(3, 1, 1) == computeConv1dLengthOffset(1, 0, 1) == 0
        for length in range(5, 40):
            assert computeConv1dOutputLength(length, 3, 1, 1, 1) == computeConv1dOutputLength(
                length, 1, 1, 0, 1
            )
            assert computeConv1dOutputLength(length, 3, 1, 0, 1) != computeConv1dOutputLength(
                length, 1, 1, 0, 1
            )

    def test_is_length_preserving_matches_a_real_conv(self) -> None:
        for kernel_size, padding, dilation in itertools.product((1, 3, 4, 5), (0, 1, 2), (1, 2)):
            preserving = isLengthPreservingConv1d(kernel_size, padding, dilation)
            actual = _actualConv1dLength(25, kernel_size, 1, padding, dilation)
            assert actual is not None
            assert preserving == (actual == 25)

    def test_even_kernel_with_odd_dilation_can_never_preserve_length(self) -> None:
        for padding in range(0, 8):
            assert not isLengthPreservingConv1d(4, padding, 1)


class TestConvTranspose1dLengthOffset:
    @pytest.mark.parametrize(
        ("kernel_size", "padding", "output_padding", "dilation"),
        list(itertools.product((1, 2, 3, 4, 5), (0, 1, 2), (0, 1), (1, 2))),
    )
    def test_output_is_scaled_input_plus_offset_for_every_length(
        self, kernel_size: int, padding: int, output_padding: int, dilation: int
    ) -> None:
        stride = 2
        offset = computeConvTranspose1dLengthOffset(kernel_size, padding, output_padding, dilation)
        for length in range(3, 30):
            assert (
                computeConvTranspose1dOutputLength(
                    length, kernel_size, stride, padding, output_padding, dilation
                )
                == (length - 1) * stride + offset + 1
            )


class TestUpsampleStackOutputLength:
    def test_matches_a_real_stack_in_both_modes(self) -> None:
        kernel_sizes = (4, 3, 4)
        strides = (2, 2, 2)
        paddings = (1, 1, 1)
        output_paddings = (0, 0, 1)
        dilations = (1, 1, 1)
        modes = ("conv_transpose", "interpolate_conv", "conv_transpose")

        layers: list[nn.Module] = []
        for stage in range(3):
            if modes[stage] == "conv_transpose":
                layers.append(
                    nn.ConvTranspose1d(
                        1,
                        1,
                        kernel_sizes[stage],
                        stride=strides[stage],
                        padding=paddings[stage],
                        output_padding=output_paddings[stage],
                        dilation=dilations[stage],
                    )
                )
            else:
                layers.append(nn.Upsample(scale_factor=strides[stage], mode="nearest"))
                layers.append(
                    nn.Conv1d(
                        1,
                        1,
                        kernel_sizes[stage],
                        padding=paddings[stage],
                        dilation=dilations[stage],
                    )
                )
        actual = int(nn.Sequential(*layers)(torch.zeros(1, 1, 8)).shape[-1])
        assert (
            computeUpsampleStackOutputLength(
                8, kernel_sizes, strides, paddings, output_paddings, dilations, modes
            )
            == actual
        )

    def test_default_decoder_configuration_doubles_at_every_transition(self) -> None:
        length = computeUpsampleStackOutputLength(
            8,
            (4, 4, 4),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            ("conv_transpose",) * 3,
        )
        assert length == 64

    def test_output_paddings_are_ignored_in_interpolate_conv_mode(self) -> None:
        with_padding = computeUpsampleStackOutputLength(
            8, (3,), (2,), (1,), (1,), (1,), ("interpolate_conv",)
        )
        without_padding = computeUpsampleStackOutputLength(
            8, (3,), (2,), (1,), (0,), (1,), ("interpolate_conv",)
        )
        assert with_padding == without_padding == 16

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="upsample_mode"):
            computeUpsampleStackOutputLength(8, (3,), (2,), (1,), (0,), (1,), ("magic",))


# --------------------------------------------------------------------------------------
# 2D counterparts. Every configuration below is deliberately non-square (different
# kernel/stride/padding/dilation on each axis), so a function that mixed the axes up, or
# silently used one axis's hyperparameters for both, cannot pass by symmetry.
# --------------------------------------------------------------------------------------

_NON_SQUARE_CONV_2D_CASES = [
    ((3, 5), (1, 2), (1, 2), (1, 1)),
    ((5, 3), (2, 1), (2, 1), (1, 2)),
    ((4, 2), (2, 3), (0, 1), (2, 1)),
    ((1, 7), (1, 1), (0, 3), (1, 1)),
    ((3, 3), (2, 2), (1, 1), (1, 1)),
]
_INPUT_SHAPES = ((16, 16), (17, 23), (31, 12), (40, 41))


class TestConv2dOutputShape:
    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation"), _NON_SQUARE_CONV_2D_CASES
    )
    def test_matches_a_real_conv2d(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        conv = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, dilation=dilation)
        for shape in _INPUT_SHAPES:
            actual = tuple(conv(torch.zeros(1, 1, *shape)).shape[-2:])
            assert computeConv2dOutputShape(shape, kernel_size, stride, padding, dilation) == actual

    def test_each_axis_only_depends_on_its_own_hyperparameters(self) -> None:
        shape = (30, 21)
        kernel_size, stride, padding, dilation = (3, 5), (2, 1), (1, 0), (1, 2)
        height, width = computeConv2dOutputShape(shape, kernel_size, stride, padding, dilation)
        assert height == computeConv1dOutputLength(
            shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        )
        assert width == computeConv1dOutputLength(
            shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        )

    def test_swapping_every_pair_swaps_the_result(self) -> None:
        """A non-square problem and its transpose must give transposed answers."""
        shape, kernel_size, stride, padding, dilation = (30, 21), (3, 5), (2, 1), (1, 0), (1, 2)
        height, width = computeConv2dOutputShape(shape, kernel_size, stride, padding, dilation)
        swapped = computeConv2dOutputShape(
            shape[::-1], kernel_size[::-1], stride[::-1], padding[::-1], dilation[::-1]
        )
        assert swapped == (width, height)


class TestConvTranspose2dOutputShape:
    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "output_padding", "dilation"),
        [
            ((4, 3), (2, 3), (1, 1), (0, 2), (1, 1)),
            ((3, 5), (2, 1), (1, 2), (1, 0), (1, 1)),
            ((4, 4), (2, 2), (1, 1), (0, 0), (1, 1)),
            ((3, 3), (3, 2), (0, 1), (2, 1), (1, 2)),
        ],
    )
    def test_matches_a_real_convtranspose2d(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        output_padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        module = nn.ConvTranspose2d(
            1,
            1,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )
        for shape in ((5, 7), (8, 8), (11, 4)):
            actual = tuple(module(torch.zeros(1, 1, *shape)).shape[-2:])
            assert (
                computeConvTranspose2dOutputShape(
                    shape, kernel_size, stride, padding, output_padding, dilation
                )
                == actual
            )


class TestUpsampleThenConv2dOutputShape:
    @pytest.mark.parametrize(
        ("scale_factor", "kernel_size", "padding", "dilation"),
        [
            ((2, 2), (3, 3), (1, 1), (1, 1)),
            ((2, 3), (3, 5), (1, 2), (1, 1)),
            ((3, 1), (5, 3), (2, 0), (1, 2)),
        ],
    )
    def test_matches_a_real_upsample_then_conv2d(
        self,
        scale_factor: tuple[int, int],
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        module = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(1, 1, kernel_size, padding=padding, dilation=dilation),
        )
        for shape in ((6, 9), (8, 8)):
            actual = tuple(module(torch.zeros(1, 1, *shape)).shape[-2:])
            assert (
                computeUpsampleThenConv2dOutputShape(
                    shape, scale_factor, kernel_size, padding, dilation
                )
                == actual
            )


class TestSolveConvTranspose2dOutputPadding:
    def test_round_trips_independently_per_axis(self) -> None:
        kernel_size, stride, padding, dilation = (3, 4), (3, 2), (1, 1), (1, 1)
        output_padding = (2, 1)
        target = computeConvTranspose2dOutputShape(
            (5, 6), kernel_size, stride, padding, output_padding, dilation
        )
        assert (
            solveConvTranspose2dOutputPadding(
                (5, 6), target, kernel_size, stride, padding, dilation
            )
            == output_padding
        )

    def test_a_gap_on_only_one_axis_leaves_the_other_untouched(self) -> None:
        """The property `TwoDCnnDecoder`'s per-axis auto-solve relies on."""
        kernel_size, stride, padding, dilation = (4, 4), (2, 2), (1, 1), (1, 1)
        natural = computeConvTranspose2dOutputShape(
            (6, 9), kernel_size, stride, padding, (0, 0), dilation
        )
        height_gap = (natural[0] + 1, natural[1])
        assert solveConvTranspose2dOutputPadding(
            (6, 9), height_gap, kernel_size, stride, padding, dilation
        ) == (1, 0)
        width_gap = (natural[0], natural[1] + 1)
        assert solveConvTranspose2dOutputPadding(
            (6, 9), width_gap, kernel_size, stride, padding, dilation
        ) == (0, 1)


class TestSolveMinimumInputShapeForConv2d:
    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation"), _NON_SQUARE_CONV_2D_CASES
    )
    def test_is_the_exact_minimum_on_both_axes(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        minimum = solveMinimumInputShapeForConv2d((1, 1), kernel_size, stride, padding, dilation)
        for axis in range(2):
            assert minimum[axis] >= 1
        # the minimum itself is acceptable to a real Conv2d
        conv = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, dilation=dilation)
        assert all(size >= 1 for size in conv(torch.zeros(1, 1, *minimum)).shape[-2:])
        # ...and shrinking either axis by one falls short on that axis
        for axis in range(2):
            if minimum[axis] > 1:
                smaller = list(minimum)
                smaller[axis] -= 1
                shape = computeConv2dOutputShape(
                    (smaller[0], smaller[1]), kernel_size, stride, padding, dilation
                )
                assert shape[axis] < 1

    def test_non_square_target_is_solved_per_axis(self) -> None:
        minimum = solveMinimumInputShapeForConv2d((2, 5), (3, 3), (2, 2), (1, 1), (1, 1))
        assert minimum == (
            solveMinimumInputLengthForConv1d(2, 3, 2, 1, 1),
            solveMinimumInputLengthForConv1d(5, 3, 2, 1, 1),
        )

    def test_also_applies_to_max_pooling_2d(self) -> None:
        minimum = solveMinimumInputShapeForConv2d((1, 1), (2, 3), (2, 3), (0, 0), (1, 1))
        pool = nn.MaxPool2d((2, 3), stride=(2, 3))
        assert all(size >= 1 for size in pool(torch.zeros(1, 1, *minimum)).shape[-2:])
        with pytest.raises(RuntimeError):
            pool(torch.zeros(1, 1, minimum[0] - 1, minimum[1]))


class TestConv2dLengthOffset:
    def test_is_the_1d_offset_applied_per_axis(self) -> None:
        kernel_size, padding, dilation = (3, 6), (1, 2), (1, 2)
        assert computeConv2dLengthOffset(kernel_size, padding, dilation) == (
            computeConv1dLengthOffset(3, 1, 1),
            computeConv1dLengthOffset(6, 2, 2),
        )

    @pytest.mark.parametrize(
        ("kernel_size", "padding", "dilation"),
        [((3, 5), (1, 2), (1, 1)), ((4, 2), (2, 0), (1, 1)), ((3, 3), (0, 2), (1, 2))],
    )
    def test_stride_one_output_is_input_plus_offset_on_both_axes(
        self, kernel_size: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]
    ) -> None:
        offset = computeConv2dLengthOffset(kernel_size, padding, dilation)
        conv = nn.Conv2d(1, 1, kernel_size, padding=padding, dilation=dilation)
        for shape in ((20, 27), (31, 22)):
            actual = tuple(conv(torch.zeros(1, 1, *shape)).shape[-2:])
            assert actual == (shape[0] + offset[0], shape[1] + offset[1])

    def test_equal_offsets_give_equal_shapes_for_every_input_shape(self) -> None:
        """What `Residual2DBlock` relies on: (3, 5)/(1, 2) and (1, 1)/(0, 0) are both
        offset (0, 0), so a residual shortcut built from the latter always lines up with a
        main path built from the former."""
        assert computeConv2dLengthOffset((3, 5), (1, 2), (1, 1)) == (0, 0)
        assert computeConv2dLengthOffset((1, 1), (0, 0), (1, 1)) == (0, 0)
        for shape in itertools.product(range(6, 20, 3), range(6, 20, 4)):
            assert computeConv2dOutputShape(
                shape, (3, 5), (1, 1), (1, 2), (1, 1)
            ) == computeConv2dOutputShape(shape, (1, 1), (1, 1), (0, 0), (1, 1))

    def test_a_mismatch_on_a_single_axis_is_detected(self) -> None:
        assert computeConv2dLengthOffset((3, 4), (1, 1), (1, 1)) == (0, -1)
        assert computeConv2dLengthOffset((3, 4), (1, 1), (1, 1)) != (0, 0)


class TestConvTranspose2dLengthOffset:
    def test_is_the_1d_offset_applied_per_axis(self) -> None:
        assert computeConvTranspose2dLengthOffset((4, 3), (1, 0), (0, 1), (1, 2)) == (
            computeConvTranspose1dLengthOffset(4, 1, 0, 1),
            computeConvTranspose1dLengthOffset(3, 0, 1, 2),
        )

    def test_output_is_scaled_input_plus_offset_on_both_axes(self) -> None:
        kernel_size, padding, output_padding, dilation = (4, 3), (1, 0), (0, 1), (1, 1)
        stride = (2, 3)
        offset = computeConvTranspose2dLengthOffset(kernel_size, padding, output_padding, dilation)
        module = nn.ConvTranspose2d(
            1,
            1,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )
        for shape in ((5, 7), (9, 4)):
            actual = tuple(module(torch.zeros(1, 1, *shape)).shape[-2:])
            expected = tuple(
                (shape[axis] - 1) * stride[axis] + offset[axis] + 1 for axis in range(2)
            )
            assert actual == expected


class TestUpsampleStack2dOutputShape:
    def test_matches_a_real_non_square_stack_in_both_modes(self) -> None:
        kernel_sizes = ((4, 4), (3, 5), (4, 2))
        strides = ((2, 2), (2, 3), (2, 2))
        paddings = ((1, 1), (1, 2), (1, 0))
        output_paddings = ((0, 0), (0, 0), (1, 0))
        dilations = ((1, 1), (1, 1), (1, 1))
        modes = ("conv_transpose", "interpolate_conv", "conv_transpose")

        layers: list[nn.Module] = []
        for stage in range(3):
            if modes[stage] == "conv_transpose":
                layers.append(
                    nn.ConvTranspose2d(
                        1,
                        1,
                        kernel_sizes[stage],
                        stride=strides[stage],
                        padding=paddings[stage],
                        output_padding=output_paddings[stage],
                        dilation=dilations[stage],
                    )
                )
            else:
                layers.append(nn.Upsample(scale_factor=strides[stage], mode="nearest"))
                layers.append(
                    nn.Conv2d(
                        1,
                        1,
                        kernel_sizes[stage],
                        padding=paddings[stage],
                        dilation=dilations[stage],
                    )
                )
        actual = tuple(nn.Sequential(*layers)(torch.zeros(1, 1, 6, 5)).shape[-2:])
        assert (
            computeUpsampleStack2dOutputShape(
                (6, 5), kernel_sizes, strides, paddings, output_paddings, dilations, modes
            )
            == actual
        )

    def test_reduces_to_the_1d_stack_when_both_axes_are_identical(self) -> None:
        length = computeUpsampleStackOutputLength(
            8, (4, 3), (2, 2), (1, 1), (0, 0), (1, 1), ("conv_transpose", "interpolate_conv")
        )
        shape = computeUpsampleStack2dOutputShape(
            (8, 8),
            ((4, 4), (3, 3)),
            ((2, 2), (2, 2)),
            ((1, 1), (1, 1)),
            ((0, 0), (0, 0)),
            ((1, 1), (1, 1)),
            ("conv_transpose", "interpolate_conv"),
        )
        assert shape == (length, length)

    def test_default_decoder_configuration_doubles_each_axis_independently(self) -> None:
        shape = computeUpsampleStack2dOutputShape(
            (6, 8),
            ((4, 4),) * 3,
            ((2, 2),) * 3,
            ((1, 1),) * 3,
            ((0, 0),) * 3,
            ((1, 1),) * 3,
            ("conv_transpose",) * 3,
        )
        assert shape == (48, 64)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="upsample_mode"):
            computeUpsampleStack2dOutputShape(
                (8, 8), ((3, 3),), ((2, 2),), ((1, 1),), ((0, 0),), ((1, 1),), ("magic",)
            )
