"""Unit tests for `utils.transformer_blocks`: positions, patches, transformer stack (P3-5)."""

import math

import pytest
import torch
from torch import nn

from global_vae.utils.autograd import backward
from global_vae.utils.transformer_blocks import (
    ImagePositions,
    PatchEmbed1d,
    PatchEmbed2d,
    PatchUnembed1d,
    PatchUnembed2d,
    SeriesPositions,
    buildPoolingHead,
    buildSinusoidalPositions1d,
    buildSinusoidalPositions2d,
    buildTransformerStack,
    checkAttentionWidth,
    checkModalitySpecs,
    computeNumPatches,
    resolvePatchSize2d,
)

_REFERENCE = torch.zeros(1)


class TestComputeNumPatches:
    @pytest.mark.parametrize(
        ("size", "patch", "expected"), [(8, 4, 2), (9, 4, 3), (1, 16, 1), (16, 16, 1), (17, 16, 2)]
    )
    def test_is_the_ceiling_of_the_ratio(self, size: int, patch: int, expected: int) -> None:
        assert computeNumPatches(size, patch) == expected

    @pytest.mark.parametrize(("size", "patch"), [(0, 4), (-1, 4), (4, 0), (4, -2)])
    def test_rejects_a_size_or_patch_below_one(self, size: int, patch: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            computeNumPatches(size, patch)


class TestResolvePatchSize2d:
    def test_one_integer_is_a_square_patch(self) -> None:
        assert resolvePatchSize2d(8) == (8, 8)

    def test_a_sequence_is_height_then_width(self) -> None:
        assert resolvePatchSize2d([4, 6]) == (4, 6)
        assert resolvePatchSize2d((2, 3)) == (2, 3)

    @pytest.mark.parametrize("bad", [[1, 2, 3], [4], []])
    def test_a_sequence_must_have_two_entries(self, bad: list[int]) -> None:
        with pytest.raises(ValueError, match="height, width"):
            resolvePatchSize2d(bad)

    @pytest.mark.parametrize("bad", [0, -3, [4, 0], [0, 4]])
    def test_rejects_a_side_below_one(self, bad: int | list[int]) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            resolvePatchSize2d(bad)


class TestCheckAttentionWidth:
    def test_accepts_a_width_the_heads_divide(self) -> None:
        checkAttentionWidth(64, 4)

    def test_rejects_a_width_the_heads_do_not_divide(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            checkAttentionWidth(10, 4)

    @pytest.mark.parametrize(("d_model", "heads"), [(0, 1), (8, 0), (-8, 2)])
    def test_rejects_non_positive_values(self, d_model: int, heads: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            checkAttentionWidth(d_model, heads)


class TestSinusoidalPositions1d:
    def test_shape(self) -> None:
        assert buildSinusoidalPositions1d(7, 16, _REFERENCE).shape == (7, 16)

    def test_position_zero_is_sine_zero_and_cosine_one(self) -> None:
        table = buildSinusoidalPositions1d(3, 8, _REFERENCE)
        assert torch.equal(table[0, 0::2], torch.zeros(4))
        assert torch.equal(table[0, 1::2], torch.ones(4))

    def test_the_first_frequency_is_one_radian_per_position(self) -> None:
        table = buildSinusoidalPositions1d(5, 8, _REFERENCE)
        for position in range(5):
            assert table[position, 0].item() == pytest.approx(math.sin(position), abs=1e-6)
            assert table[position, 1].item() == pytest.approx(math.cos(position), abs=1e-6)

    def test_the_last_frequency_is_the_slowest(self) -> None:
        dim = 8
        table = buildSinusoidalPositions1d(3, dim, _REFERENCE)
        slowest = 10000.0 ** (-(dim - 2) / dim)
        assert table[2, dim - 2].item() == pytest.approx(math.sin(2 * slowest), abs=1e-6)

    def test_a_longer_table_starts_with_the_shorter_one(self) -> None:
        short = buildSinusoidalPositions1d(4, 16, _REFERENCE)
        long = buildSinusoidalPositions1d(9, 16, _REFERENCE)
        assert torch.equal(long[:4], short)

    def test_values_stay_within_one(self) -> None:
        assert buildSinusoidalPositions1d(500, 32, _REFERENCE).abs().max() <= 1.0

    def test_takes_the_dtype_of_the_reference(self) -> None:
        reference = torch.zeros(1, dtype=torch.float64)
        assert buildSinusoidalPositions1d(3, 8, reference).dtype == torch.float64

    @pytest.mark.parametrize("dim", [0, 1, 3, 7])
    def test_rejects_a_width_that_is_not_a_positive_even_number(self, dim: int) -> None:
        with pytest.raises(ValueError, match="even width"):
            buildSinusoidalPositions1d(4, dim, _REFERENCE)

    def test_rejects_no_position(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            buildSinusoidalPositions1d(0, 8, _REFERENCE)


class TestSinusoidalPositions2d:
    def test_shape_is_row_major_over_the_grid(self) -> None:
        assert buildSinusoidalPositions2d(3, 5, 16, _REFERENCE).shape == (15, 16)

    def test_the_first_half_encodes_the_row_and_the_second_the_column(self) -> None:
        rows, columns, dim = 3, 4, 16
        table = buildSinusoidalPositions2d(rows, columns, dim, _REFERENCE).view(rows, columns, dim)
        # Same row, other column: the row half is shared, the column half differs.
        assert torch.equal(table[1, 0, : dim // 2], table[1, 3, : dim // 2])
        assert not torch.equal(table[1, 0, dim // 2 :], table[1, 3, dim // 2 :])
        # Same column, other row: the other way round.
        assert torch.equal(table[0, 2, dim // 2 :], table[2, 2, dim // 2 :])
        assert not torch.equal(table[0, 2, : dim // 2], table[2, 2, : dim // 2])

    def test_every_patch_gets_its_own_vector(self) -> None:
        table = buildSinusoidalPositions2d(4, 4, 32, _REFERENCE)
        assert torch.unique(table, dim=0).shape[0] == 16

    @pytest.mark.parametrize("dim", [0, 2, 6, 10])
    def test_rejects_a_width_that_is_not_a_multiple_of_four(self, dim: int) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            buildSinusoidalPositions2d(2, 2, dim, _REFERENCE)

    @pytest.mark.parametrize(("height", "width"), [(0, 3), (3, 0)])
    def test_rejects_an_empty_grid(self, height: int, width: int) -> None:
        with pytest.raises(ValueError, match="at least 1 x 1"):
            buildSinusoidalPositions2d(height, width, 16, _REFERENCE)


class TestSeriesPositions:
    def test_sinusoidal_has_no_parameters_and_any_length(self) -> None:
        positions = SeriesPositions(16)
        assert list(positions.parameters()) == []
        assert positions(5, _REFERENCE).shape == (5, 16)
        assert positions(5000, _REFERENCE).shape == (5000, 16)

    def test_learned_has_one_trained_vector_per_position(self) -> None:
        positions = SeriesPositions(16, "learned", max_tokens=10)
        assert [tuple(p.shape) for p in positions.parameters()] == [(10, 16)]

    def test_learned_returns_the_first_rows_of_its_table(self) -> None:
        positions = SeriesPositions(16, "learned", max_tokens=10)
        assert torch.equal(positions(4, _REFERENCE), positions.table[:4])  # type: ignore[index]

    def test_learned_refuses_more_tokens_than_it_covers(self) -> None:
        positions = SeriesPositions(16, "learned", max_tokens=4)
        with pytest.raises(ValueError, match="cover 4"):
            positions(5, _REFERENCE)

    @pytest.mark.parametrize("max_tokens", [None, 0, -1])
    def test_learned_needs_a_positive_max_tokens(self, max_tokens: int | None) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            SeriesPositions(16, "learned", max_tokens=max_tokens)

    def test_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown positional encoding 'rotary'"):
            SeriesPositions(16, "rotary")

    def test_sinusoidal_rejects_an_odd_width(self) -> None:
        with pytest.raises(ValueError, match="even d_model"):
            SeriesPositions(15)

    def test_learned_accepts_an_odd_width(self) -> None:
        assert SeriesPositions(15, "learned", max_tokens=3)(2, _REFERENCE).shape == (2, 15)


class TestImagePositions:
    def test_sinusoidal_has_no_parameters_and_any_grid(self) -> None:
        positions = ImagePositions(16)
        assert list(positions.parameters()) == []
        assert positions(2, 3, _REFERENCE).shape == (6, 16)
        assert positions(30, 40, _REFERENCE).shape == (1200, 16)

    def test_learned_has_one_trained_vector_per_patch_of_its_grid(self) -> None:
        positions = ImagePositions(16, "learned", grid_shape=(3, 4))
        assert [tuple(p.shape) for p in positions.parameters()] == [(12, 16)]
        assert positions(3, 4, _REFERENCE).shape == (12, 16)

    def test_learned_refuses_another_grid(self) -> None:
        positions = ImagePositions(16, "learned", grid_shape=(3, 4))
        with pytest.raises(ValueError, match="fix the image size"):
            positions(4, 3, _REFERENCE)

    @pytest.mark.parametrize("grid", [None, (0, 4), (4, 0)])
    def test_learned_needs_a_valid_grid(self, grid: tuple[int, int] | None) -> None:
        with pytest.raises(ValueError, match="at least 1 x 1"):
            ImagePositions(16, "learned", grid_shape=grid)

    def test_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            ImagePositions(16, "rotary")

    @pytest.mark.parametrize("d_model", [2, 6, 18])
    def test_sinusoidal_rejects_a_width_that_is_not_a_multiple_of_four(self, d_model: int) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            ImagePositions(d_model)


def _identityEmbed1d(channels: int, patch: int) -> PatchEmbed1d:
    """A patch embedding whose token is the patch itself, channel by channel."""
    embed = PatchEmbed1d(channels, channels * patch, patch)
    with torch.no_grad():
        for parameter in embed.parameters():
            parameter.zero_()
        for channel in range(channels):
            for offset in range(patch):
                embed.projection.weight[channel * patch + offset, channel, offset] = 1.0
    return embed


def _identityEmbed2d(channels: int, patch: tuple[int, int]) -> PatchEmbed2d:
    """A patch embedding whose token is the patch itself, channel by channel, row by row."""
    height, width = patch
    embed = PatchEmbed2d(channels, channels * height * width, patch)
    with torch.no_grad():
        for parameter in embed.parameters():
            parameter.zero_()
        for channel in range(channels):
            for row in range(height):
                for column in range(width):
                    index = channel * height * width + row * width + column
                    embed.projection.weight[index, channel, row, column] = 1.0
    return embed


def _identityUnembed1d(channels: int, patch: int, length: int) -> PatchUnembed1d:
    unembed = PatchUnembed1d(channels * patch, patch, channels, length)
    with torch.no_grad():
        unembed.projection.weight.copy_(torch.eye(channels * patch))
        unembed.projection.bias.zero_()
    return unembed


def _identityUnembed2d(
    channels: int, patch: tuple[int, int], shape: tuple[int, int]
) -> PatchUnembed2d:
    size = channels * patch[0] * patch[1]
    unembed = PatchUnembed2d(size, patch, channels, shape)
    with torch.no_grad():
        unembed.projection.weight.copy_(torch.eye(size))
        unembed.projection.bias.zero_()
    return unembed


class TestPatchEmbed1d:
    def test_one_token_per_patch(self) -> None:
        tokens = PatchEmbed1d(1, 12, 4)(torch.randn(3, 1, 16))
        assert tokens.shape == (3, 4, 12)

    def test_a_length_that_is_not_a_multiple_gets_one_more_token(self) -> None:
        assert PatchEmbed1d(1, 12, 4)(torch.randn(3, 1, 17)).shape == (3, 5, 12)
        assert PatchEmbed1d(1, 12, 16)(torch.randn(3, 1, 1)).shape == (3, 1, 12)

    def test_a_token_is_a_function_of_its_own_patch_only(self) -> None:
        embed = _identityEmbed1d(1, 4)
        series = torch.arange(12.0).view(1, 1, 12)
        assert torch.equal(embed(series), series.view(1, 3, 4))

    def test_channels_are_kept_apart_inside_a_token(self) -> None:
        embed = _identityEmbed1d(2, 3)
        series = torch.arange(12.0).view(1, 2, 6)
        tokens = embed(series)
        assert tokens.shape == (1, 2, 6)
        assert torch.equal(tokens[0, 0], torch.tensor([0.0, 1.0, 2.0, 6.0, 7.0, 8.0]))
        assert torch.equal(tokens[0, 1], torch.tensor([3.0, 4.0, 5.0, 9.0, 10.0, 11.0]))

    def test_the_tail_is_padded_by_repeating_the_last_value(self) -> None:
        embed = _identityEmbed1d(1, 4)
        series = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        tokens = embed(series)
        assert torch.equal(tokens[0, 1], torch.tensor([5.0, 6.0, 6.0, 6.0]))

    @pytest.mark.parametrize(("channels", "d_model", "patch"), [(0, 8, 4), (1, 0, 4), (1, 8, 0)])
    def test_rejects_a_dimension_below_one(self, channels: int, d_model: int, patch: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PatchEmbed1d(channels, d_model, patch)


class TestPatchEmbed2d:
    def test_one_token_per_patch_in_row_major_order(self) -> None:
        embed = PatchEmbed2d(1, 12, (4, 4))
        assert embed(torch.randn(2, 1, 8, 12)).shape == (2, 6, 12)
        assert embed.gridShape(8, 12) == (2, 3)

    def test_a_size_that_is_not_a_multiple_gets_one_more_row_or_column(self) -> None:
        embed = PatchEmbed2d(1, 12, (4, 4))
        assert embed.gridShape(9, 5) == (3, 2)
        assert embed(torch.randn(2, 1, 9, 5)).shape == (2, 6, 12)
        assert embed(torch.randn(2, 1, 1, 1)).shape == (2, 1, 12)

    def test_a_token_is_a_function_of_its_own_patch_only(self) -> None:
        embed = _identityEmbed2d(1, (2, 3))
        image = torch.arange(24.0).view(1, 1, 4, 6)
        tokens = embed(image)
        assert tokens.shape == (1, 4, 6)
        assert torch.equal(tokens[0, 0], torch.tensor([0.0, 1.0, 2.0, 6.0, 7.0, 8.0]))
        assert torch.equal(tokens[0, 1], torch.tensor([3.0, 4.0, 5.0, 9.0, 10.0, 11.0]))
        assert torch.equal(tokens[0, 2], torch.tensor([12.0, 13.0, 14.0, 18.0, 19.0, 20.0]))

    def test_the_bottom_and_right_edges_repeat_the_edge_pixels(self) -> None:
        embed = _identityEmbed2d(1, (2, 2))
        image = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        tokens = embed(image)
        assert tokens.shape == (1, 4, 4)
        assert torch.equal(tokens[0, 1], torch.tensor([3.0, 3.0, 6.0, 6.0]))  # right edge
        assert torch.equal(tokens[0, 2], torch.tensor([7.0, 8.0, 7.0, 8.0]))  # bottom edge
        assert torch.equal(tokens[0, 3], torch.tensor([9.0, 9.0, 9.0, 9.0]))  # corner

    @pytest.mark.parametrize("patch", [(0, 4), (4, 0)])
    def test_rejects_a_patch_side_below_one(self, patch: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PatchEmbed2d(1, 8, patch)


class TestPatchUnembed1d:
    def test_shape_and_crop(self) -> None:
        unembed = PatchUnembed1d(8, 4, 1, 10)
        assert unembed.num_tokens == 3
        assert unembed(torch.randn(2, 3, 8)).shape == (2, 1, 10)

    def test_a_wrong_number_of_tokens_is_refused(self) -> None:
        unembed = PatchUnembed1d(8, 4, 1, 10)
        with pytest.raises(ValueError, match="Expected 3 tokens"):
            unembed(torch.randn(2, 4, 8))

    @pytest.mark.parametrize(("length", "patch"), [(12, 4), (10, 4), (3, 16), (16, 16), (17, 16)])
    def test_embedding_then_unembedding_returns_the_series(self, length: int, patch: int) -> None:
        series = torch.randn(2, 1, length)
        restored = _identityUnembed1d(1, patch, length)(_identityEmbed1d(1, patch)(series))
        assert torch.allclose(restored, series)

    def test_the_round_trip_keeps_channels_apart(self) -> None:
        series = torch.randn(2, 3, 10)
        restored = _identityUnembed1d(3, 4, 10)(_identityEmbed1d(3, 4)(series))
        assert torch.allclose(restored, series)

    def test_rejects_a_dimension_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PatchUnembed1d(8, 4, 1, 0)


class TestPatchUnembed2d:
    def test_shape_grid_and_crop(self) -> None:
        unembed = PatchUnembed2d(16, (4, 4), 1, (6, 10))
        assert unembed.grid_shape == (2, 3)
        assert unembed.num_tokens == 6
        assert unembed(torch.randn(2, 6, 16)).shape == (2, 1, 6, 10)

    def test_a_wrong_number_of_tokens_is_refused(self) -> None:
        unembed = PatchUnembed2d(16, (4, 4), 1, (6, 10))
        with pytest.raises(ValueError, match="Expected 6 tokens"):
            unembed(torch.randn(2, 5, 16))

    @pytest.mark.parametrize(
        ("shape", "patch"), [((8, 8), (4, 4)), ((9, 5), (4, 4)), ((3, 7), (2, 3)), ((1, 1), (8, 8))]
    )
    def test_embedding_then_unembedding_returns_the_image(
        self, shape: tuple[int, int], patch: tuple[int, int]
    ) -> None:
        image = torch.randn(2, 1, *shape)
        restored = _identityUnembed2d(1, patch, shape)(_identityEmbed2d(1, patch)(image))
        assert torch.allclose(restored, image)

    def test_the_round_trip_keeps_channels_apart(self) -> None:
        image = torch.randn(2, 3, 5, 7)
        restored = _identityUnembed2d(3, (2, 3), (5, 7))(_identityEmbed2d(3, (2, 3))(image))
        assert torch.allclose(restored, image)

    @pytest.mark.parametrize("shape", [(0, 4), (4, 0)])
    def test_rejects_an_output_side_below_one(self, shape: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PatchUnembed2d(16, (4, 4), 1, shape)


class TestBuildTransformerStack:
    def test_keeps_the_shape_of_the_token_sequence(self) -> None:
        stack = buildTransformerStack(16, 4, 2)
        assert stack(torch.randn(3, 7, 16)).shape == (3, 7, 16)

    def test_has_the_requested_layers_and_a_final_norm(self) -> None:
        stack = buildTransformerStack(16, 4, 3)
        assert len(stack.layers) == 3
        assert isinstance(stack.norm, nn.LayerNorm)

    def test_layers_are_pre_norm_and_batch_first(self) -> None:
        layer = buildTransformerStack(16, 4, 1).layers[0]
        assert layer.norm_first is True
        assert layer.self_attn.batch_first is True

    def test_feedforward_width_defaults_to_four_times_the_token_width(self) -> None:
        assert buildTransformerStack(16, 4, 1).layers[0].linear1.out_features == 64
        assert (
            buildTransformerStack(16, 4, 1, feedforward_dim=10).layers[0].linear1.out_features == 10
        )

    def test_any_number_of_tokens_is_accepted(self) -> None:
        stack = buildTransformerStack(16, 4, 1)
        for tokens in (1, 2, 50):
            assert stack(torch.randn(2, tokens, 16)).shape == (2, tokens, 16)

    def test_no_warning_about_nested_tensors(self, recwarn: pytest.WarningsRecorder) -> None:
        buildTransformerStack(16, 4, 2)
        assert len(recwarn) == 0

    def test_dropout_acts_in_training_mode_only(self) -> None:
        torch.manual_seed(0)
        stack = buildTransformerStack(16, 4, 1, dropout=0.5)
        x = torch.randn(2, 5, 16)
        stack.eval()
        assert torch.equal(stack(x), stack(x))
        stack.train()
        assert not torch.equal(stack(x), stack(x))

    def test_gradient_reaches_every_parameter(self) -> None:
        stack = buildTransformerStack(16, 4, 2)
        backward(stack(torch.randn(2, 5, 16)).sum())
        assert [name for name, p in stack.named_parameters() if p.grad is None] == []

    def test_relu_is_accepted(self) -> None:
        assert buildTransformerStack(16, 4, 1, activation="relu")(torch.randn(1, 2, 16)).shape == (
            1,
            2,
            16,
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"d_model": 10, "num_heads": 4, "num_layers": 1}, "not evenly divisible"),
            ({"d_model": 16, "num_heads": 4, "num_layers": 0}, "num_layers"),
            ({"d_model": 16, "num_heads": 4, "num_layers": 1, "feedforward_dim": 0}, "feedforward"),
            ({"d_model": 16, "num_heads": 4, "num_layers": 1, "dropout": 1.0}, "dropout"),
            ({"d_model": 16, "num_heads": 4, "num_layers": 1, "dropout": -0.1}, "dropout"),
            ({"d_model": 16, "num_heads": 4, "num_layers": 1, "activation": "swish"}, "activation"),
        ],
    )
    def test_rejects_an_invalid_setting(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            buildTransformerStack(**kwargs)  # type: ignore[arg-type]


class TestBuildPoolingHead:
    def test_goes_straight_to_the_outputs_without_hidden_widths(self) -> None:
        head, to_mu, to_logvar = buildPoolingHead(16, (), 4)
        assert isinstance(head, nn.Identity)
        assert (to_mu.in_features, to_mu.out_features) == (16, 4)
        assert (to_logvar.in_features, to_logvar.out_features) == (16, 4)

    def test_hidden_widths_become_an_mlp(self) -> None:
        head, to_mu, _ = buildPoolingHead(16, (32, 8), 4)
        assert head(torch.randn(3, 16)).shape == (3, 8)
        assert to_mu.in_features == 8

    @pytest.mark.parametrize(("hidden", "latent"), [((0,), 4), ((8, -1), 4), ((), 0)])
    def test_rejects_a_width_below_one(self, hidden: tuple[int, ...], latent: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            buildPoolingHead(16, hidden, latent)


class TestCheckModalitySpecs:
    _ALLOWED = {"series": ("patch_size",), "image": ("patch_size", "in_channels")}
    _REQUIRED: dict[str, tuple[str, ...]] = {"series": (), "image": ("in_channels",)}

    def _check(self, modalities: dict[str, dict[str, object]]) -> None:
        checkModalitySpecs(modalities, self._ALLOWED, self._REQUIRED, "Owner")

    def test_accepts_valid_specs(self) -> None:
        self._check(
            {
                "signal": {"kind": "series", "patch_size": 4},
                "im": {"kind": "image", "in_channels": 1},
            }
        )

    def test_rejects_no_modality(self) -> None:
        with pytest.raises(ValueError, match="Owner needs at least one modality"):
            self._check({})

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            self._check({"": {"kind": "series"}})

    @pytest.mark.parametrize("spec", [{}, {"kind": "graph"}, {"kind": None}])
    def test_rejects_a_missing_or_unknown_kind(self, spec: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="modality 'x' needs kind"):
            self._check({"x": spec})

    def test_rejects_a_key_the_kind_does_not_take(self) -> None:
        with pytest.raises(ValueError, match=r"unknown key\(s\) \['in_channels'\]"):
            self._check({"x": {"kind": "series", "in_channels": 2}})

    def test_names_the_allowed_keys(self) -> None:
        with pytest.raises(ValueError, match=r"Allowed: \['kind', 'patch_size'\]"):
            self._check({"x": {"kind": "series", "typo": 1}})

    def test_rejects_a_missing_required_key(self) -> None:
        with pytest.raises(ValueError, match=r"missing required key\(s\) \['in_channels'\]"):
            self._check({"x": {"kind": "image"}})
