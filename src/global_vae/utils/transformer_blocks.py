"""Building blocks of the transformer backbones: positions, patches and the transformer stack.

See spec §6 (a small transformer over the series, a ViT for images) and roadmap P3-5. The
backbones in `encoders/` and `decoders/` (`1d_transformer_*`, `2d_vit_*`, `shared_transformer_*`)
are built from the pieces below, which is also where the tokenisation that a shared encoder owns
(ADR 0027) lives.

A series or an image becomes a sequence of tokens by cutting it into non-overlapping patches and
projecting each patch to `d_model` values (`PatchEmbed1d`, `PatchEmbed2d`). A decoder runs the
other way: one token per patch is projected back to the patch values and the patches are put
side by side (`PatchUnembed1d`, `PatchUnembed2d`).

Lengths and sizes need not be multiples of the patch size. An encoder pads the end of its input by
repeating the last value (the edge of a series or an image, not an invented zero) up to the next
multiple, and a decoder crops its output back to the requested length or shape, so the number of
tokens is `ceil(size / patch_size)` on each axis.

Positions come in two kinds (`SeriesPositions`, `ImagePositions`). `"sinusoidal"` has no
parameters and is computed for whatever number of tokens arrives, so an encoder built with it
accepts series and images of any size, as the CNN encoders do through adaptive pooling.
`"learned"` is one trained vector per position, up to a size fixed at construction.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

POSITION_KINDS = ("sinusoidal", "learned")
SERIES_KIND = "series"
IMAGE_KIND = "image"
MODALITY_KINDS = (SERIES_KIND, IMAGE_KIND)
_SINUSOID_BASE = 10000.0
_LEARNED_INIT_STD = 0.02


def computeNumPatches(size: int, patch_size: int) -> int:
    """Return how many patches cover `size` values, the last one possibly padded.

    Args:
        size: Number of values along one axis, at least `1`.
        patch_size: Number of values per patch along that axis, at least `1`.

    Returns:
        `ceil(size / patch_size)`.

    Raises:
        ValueError: If `size` or `patch_size` is below `1`.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}.")
    if patch_size < 1:
        raise ValueError(f"patch_size must be at least 1, got {patch_size}.")
    return -(-size // patch_size)


def resolvePatchSize2d(patch_size: int | Sequence[int]) -> tuple[int, int]:
    """Turn a patch size given as one number or as `(height, width)` into a pair.

    Args:
        patch_size: One integer (square patches) or a sequence of two integers.

    Returns:
        `(patch_height, patch_width)`.

    Raises:
        ValueError: If a sequence does not have exactly two entries, or an entry is below `1`.
    """
    if isinstance(patch_size, int):
        sides = [patch_size, patch_size]
    else:
        sides = [int(side) for side in patch_size]
        if len(sides) != 2:
            raise ValueError(f"patch_size must be an integer or (height, width), got {sides}.")
    if min(sides) < 1:
        raise ValueError(f"patch_size entries must be at least 1, got {sides}.")
    return sides[0], sides[1]


def checkAttentionWidth(d_model: int, num_heads: int) -> None:
    """Check that `num_heads` heads share `d_model` evenly, and that both are positive.

    Args:
        d_model: Token width.
        num_heads: Number of attention heads.

    Raises:
        ValueError: If either is below `1`, or `d_model` is not divisible by `num_heads`.
    """
    if d_model < 1 or num_heads < 1:
        raise ValueError(
            f"d_model and num_heads must be at least 1, got {d_model} and {num_heads}."
        )
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} is not evenly divisible by num_heads={num_heads}.")


def buildSinusoidalPositions1d(
    num_positions: int, dim: int, reference: torch.Tensor
) -> torch.Tensor:
    """Build the fixed sine and cosine position table of a sequence.

    Entry `[p, 2i]` is `sin(p / 10000^(2i / dim))` and entry `[p, 2i + 1]` the matching cosine, as
    in "Attention Is All You Need".

    Args:
        num_positions: Number of positions (tokens), at least `1`.
        dim: Width of each position vector, even.
        reference: A tensor whose device and dtype the table takes.

    Returns:
        A `(num_positions, dim)` tensor, not a parameter.

    Raises:
        ValueError: If `num_positions` is below `1` or `dim` is not a positive even number.
    """
    if num_positions < 1:
        raise ValueError(f"num_positions must be at least 1, got {num_positions}.")
    if dim < 2 or dim % 2 != 0:
        raise ValueError(f"A sinusoidal position needs an even width of at least 2, got {dim}.")
    positions = torch.arange(num_positions, dtype=torch.float32, device=reference.device)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=reference.device)
        * (-math.log(_SINUSOID_BASE) / dim)
    )
    angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    table = torch.zeros(num_positions, dim, device=reference.device)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table.to(reference.dtype)


def buildSinusoidalPositions2d(
    grid_height: int, grid_width: int, dim: int, reference: torch.Tensor
) -> torch.Tensor:
    """Build the fixed position table of a grid of patches, row by row.

    The first half of each vector encodes the row and the second half the column, each as in
    `buildSinusoidalPositions1d`.

    Args:
        grid_height: Number of patch rows, at least `1`.
        grid_width: Number of patch columns, at least `1`.
        dim: Width of each position vector, a multiple of `4`.
        reference: A tensor whose device and dtype the table takes.

    Returns:
        A `(grid_height * grid_width, dim)` tensor, not a parameter.

    Raises:
        ValueError: If the grid is empty or `dim` is not a positive multiple of `4`.
    """
    if grid_height < 1 or grid_width < 1:
        raise ValueError(
            f"The patch grid must be at least 1 x 1, got {grid_height} x {grid_width}."
        )
    if dim < 4 or dim % 4 != 0:
        raise ValueError(
            f"A 2D sinusoidal position needs a width that is a multiple of 4, got {dim}."
        )
    half = dim // 2
    rows = buildSinusoidalPositions1d(grid_height, half, reference)
    columns = buildSinusoidalPositions1d(grid_width, half, reference)
    grid = torch.cat(
        [
            rows.unsqueeze(1).expand(grid_height, grid_width, half),
            columns.unsqueeze(0).expand(grid_height, grid_width, half),
        ],
        dim=-1,
    )
    return grid.reshape(grid_height * grid_width, dim)


class SeriesPositions(nn.Module):
    """Position vectors of the tokens of a series, sinusoidal or learned."""

    def __init__(
        self, d_model: int, kind: str = "sinusoidal", max_tokens: int | None = None
    ) -> None:
        """Build the position source.

        Args:
            d_model: Token width.
            kind: `"sinusoidal"` (no parameters, any number of tokens) or `"learned"` (one
                trained vector per position, up to `max_tokens` tokens).
            max_tokens: Number of tokens a learned table has. Required for `"learned"`, ignored
                for `"sinusoidal"`.

        Raises:
            ValueError: If `kind` is not one of `POSITION_KINDS`, a learned table has no positive
                `max_tokens`, or a sinusoidal one has an odd `d_model`.
        """
        super().__init__()
        _checkPositionKind(kind)
        self._kind = kind
        self._d_model = d_model
        self._max_tokens = max_tokens
        self.table: nn.Parameter | None = None
        if kind == "learned":
            if max_tokens is None or max_tokens < 1:
                raise ValueError(
                    f"learned positions need a positive max_tokens, got {max_tokens!r}."
                )
            self.table = nn.Parameter(torch.randn(max_tokens, d_model) * _LEARNED_INIT_STD)
        elif d_model < 2 or d_model % 2 != 0:
            raise ValueError(f"sinusoidal positions need an even d_model, got {d_model}.")

    def forward(self, num_tokens: int, reference: torch.Tensor) -> torch.Tensor:
        """Return the position vectors of the first `num_tokens` tokens.

        Args:
            num_tokens: Number of tokens of the series, at least `1`.
            reference: A tensor whose device and dtype a sinusoidal table takes.

        Returns:
            A `(num_tokens, d_model)` tensor.

        Raises:
            ValueError: If a learned table has fewer than `num_tokens` positions.
        """
        if self.table is None:
            return buildSinusoidalPositions1d(num_tokens, self._d_model, reference)
        if num_tokens > self.table.shape[0]:
            raise ValueError(
                f"The series has {num_tokens} tokens, but its learned positions cover "
                f"{self.table.shape[0]}. Build the model with a larger max_length, or use "
                f"sinusoidal positions, which have no limit."
            )
        return self.table[:num_tokens].to(reference.dtype)


class ImagePositions(nn.Module):
    """Position vectors of the tokens of an image, in row-major order."""

    def __init__(
        self,
        d_model: int,
        kind: str = "sinusoidal",
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        """Build the position source.

        Args:
            d_model: Token width.
            kind: `"sinusoidal"` (no parameters, any grid) or `"learned"` (one trained vector per
                patch of a grid fixed at construction).
            grid_shape: `(rows, columns)` of patches a learned table has. Required for
                `"learned"`, ignored for `"sinusoidal"`.

        Raises:
            ValueError: If `kind` is not one of `POSITION_KINDS`, a learned table has no valid
                `grid_shape`, or a sinusoidal one has a `d_model` that is not a multiple of `4`.
        """
        super().__init__()
        _checkPositionKind(kind)
        self._kind = kind
        self._d_model = d_model
        self._grid_shape = grid_shape
        self.table: nn.Parameter | None = None
        if kind == "learned":
            if grid_shape is None or min(grid_shape) < 1:
                raise ValueError(
                    f"learned positions need a grid of at least 1 x 1 patches, got {grid_shape!r}."
                )
            self.table = nn.Parameter(
                torch.randn(grid_shape[0] * grid_shape[1], d_model) * _LEARNED_INIT_STD
            )
        elif d_model < 4 or d_model % 4 != 0:
            raise ValueError(
                f"2D sinusoidal positions need a d_model multiple of 4, got {d_model}."
            )

    def forward(self, grid_height: int, grid_width: int, reference: torch.Tensor) -> torch.Tensor:
        """Return the position vectors of a grid of patches, row by row.

        Args:
            grid_height: Number of patch rows.
            grid_width: Number of patch columns.
            reference: A tensor whose device and dtype a sinusoidal table takes.

        Returns:
            A `(grid_height * grid_width, d_model)` tensor.

        Raises:
            ValueError: If learned positions were built for another grid.
        """
        if self.table is None:
            return buildSinusoidalPositions2d(grid_height, grid_width, self._d_model, reference)
        if self._grid_shape != (grid_height, grid_width):
            raise ValueError(
                f"The image has a {grid_height} x {grid_width} grid of patches, but its learned "
                f"positions were built for {self._grid_shape}. Learned positions fix the image "
                f"size; use sinusoidal positions for another one."
            )
        return self.table.to(reference.dtype)


def _checkPositionKind(kind: str) -> None:
    """Raise if `kind` is not a known kind of position.

    Args:
        kind: The requested kind.

    Raises:
        ValueError: If `kind` is not one of `POSITION_KINDS`.
    """
    if kind not in POSITION_KINDS:
        raise ValueError(f"Unknown positional encoding '{kind}'. Expected one of {POSITION_KINDS}.")


class PatchEmbed1d(nn.Module):
    """Cut a series into patches and project each one to a `d_model` token."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int) -> None:
        """Build the projection, a convolution whose kernel and stride are the patch size.

        Args:
            in_channels: Number of channels of the series.
            d_model: Token width.
            patch_size: Number of values per patch, at least `1`.

        Raises:
            ValueError: If `in_channels`, `d_model` or `patch_size` is below `1`.
        """
        super().__init__()
        if in_channels < 1 or d_model < 1 or patch_size < 1:
            raise ValueError(
                f"in_channels, d_model and patch_size must be at least 1, got {in_channels}, "
                f"{d_model} and {patch_size}."
            )
        self.patch_size = patch_size
        self.projection = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Embed a batch of series.

        Args:
            series: `(batch, in_channels, length)`, `length` at least `1`.

        Returns:
            `(batch, ceil(length / patch_size), d_model)`.
        """
        padding = -series.shape[-1] % self.patch_size
        if padding:
            series = functional.pad(series, (0, padding), mode="replicate")
        embedded: torch.Tensor = self.projection(series)
        return embedded.transpose(1, 2)


class PatchEmbed2d(nn.Module):
    """Cut an image into patches and project each one to a `d_model` token."""

    def __init__(self, in_channels: int, d_model: int, patch_size: tuple[int, int]) -> None:
        """Build the projection, a convolution whose kernel and stride are the patch size.

        Args:
            in_channels: Number of channels of the image.
            d_model: Token width.
            patch_size: `(patch_height, patch_width)`, each at least `1`.

        Raises:
            ValueError: If `in_channels`, `d_model` or a patch side is below `1`.
        """
        super().__init__()
        if in_channels < 1 or d_model < 1 or min(patch_size) < 1:
            raise ValueError(
                f"in_channels, d_model and the patch sides must be at least 1, got {in_channels}, "
                f"{d_model} and {list(patch_size)}."
            )
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def gridShape(self, height: int, width: int) -> tuple[int, int]:
        """Return the number of patch rows and columns an image of this size gives.

        Args:
            height: Image height.
            width: Image width.

        Returns:
            `(rows, columns)`.
        """
        return (
            computeNumPatches(height, self.patch_size[0]),
            computeNumPatches(width, self.patch_size[1]),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a batch of images.

        Args:
            images: `(batch, in_channels, height, width)`.

        Returns:
            `(batch, rows * columns, d_model)`, the patches in row-major order.
        """
        pad_height = -images.shape[-2] % self.patch_size[0]
        pad_width = -images.shape[-1] % self.patch_size[1]
        if pad_height or pad_width:
            images = functional.pad(images, (0, pad_width, 0, pad_height), mode="replicate")
        embedded: torch.Tensor = self.projection(images)
        return embedded.flatten(2).transpose(1, 2)


class PatchUnembed1d(nn.Module):
    """Project one token per patch back to patch values and lay the patches side by side."""

    def __init__(
        self, d_model: int, patch_size: int, out_channels: int, output_length: int
    ) -> None:
        """Build the projection.

        Args:
            d_model: Token width.
            patch_size: Number of values per patch.
            out_channels: Number of channels of the series.
            output_length: Length of the series to produce. The last patch is cropped to it.

        Raises:
            ValueError: If any argument is below `1`.
        """
        super().__init__()
        if min(d_model, patch_size, out_channels, output_length) < 1:
            raise ValueError(
                "d_model, patch_size, out_channels and output_length must be at least 1, got "
                f"{d_model}, {patch_size}, {out_channels} and {output_length}."
            )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.output_length = output_length
        self.num_tokens = computeNumPatches(output_length, patch_size)
        self.projection = nn.Linear(d_model, out_channels * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Turn the tokens of a series into the series.

        Args:
            tokens: `(batch, num_tokens, d_model)`, with `num_tokens = ceil(output_length /
                patch_size)`.

        Returns:
            `(batch, out_channels, output_length)`.

        Raises:
            ValueError: If the number of tokens is not `num_tokens`.
        """
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens for a series of length {self.output_length}, "
                f"got {tokens.shape[1]}."
            )
        batch = tokens.shape[0]
        patches = self.projection(tokens).view(batch, self.num_tokens, self.out_channels, -1)
        series = patches.permute(0, 2, 1, 3).reshape(batch, self.out_channels, -1)
        cropped: torch.Tensor = series[..., : self.output_length]
        return cropped


class PatchUnembed2d(nn.Module):
    """Project one token per patch back to patch pixels and tile the patches into an image."""

    def __init__(
        self,
        d_model: int,
        patch_size: tuple[int, int],
        out_channels: int,
        output_shape: tuple[int, int],
    ) -> None:
        """Build the projection.

        Args:
            d_model: Token width.
            patch_size: `(patch_height, patch_width)`.
            out_channels: Number of channels of the image.
            output_shape: `(height, width)` of the image to produce. Patches on the bottom and
                right edges are cropped to it.

        Raises:
            ValueError: If any argument is below `1`.
        """
        super().__init__()
        if min(d_model, out_channels, *patch_size, *output_shape) < 1:
            raise ValueError(
                "d_model, out_channels, the patch sides and the output sides must be at least 1, "
                f"got {d_model}, {out_channels}, {list(patch_size)} and {list(output_shape)}."
            )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.grid_shape = (
            computeNumPatches(output_shape[0], patch_size[0]),
            computeNumPatches(output_shape[1], patch_size[1]),
        )
        self.num_tokens = self.grid_shape[0] * self.grid_shape[1]
        self.projection = nn.Linear(d_model, out_channels * patch_size[0] * patch_size[1])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Turn the tokens of an image into the image.

        Args:
            tokens: `(batch, rows * columns, d_model)` in row-major order, the grid being
                `grid_shape`.

        Returns:
            `(batch, out_channels, height, width)`.

        Raises:
            ValueError: If the number of tokens is not `num_tokens`.
        """
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens for an image of shape "
                f"{list(self.output_shape)}, got {tokens.shape[1]}."
            )
        batch = tokens.shape[0]
        rows, columns = self.grid_shape
        patch_height, patch_width = self.patch_size
        patches = self.projection(tokens).view(
            batch, rows, columns, self.out_channels, patch_height, patch_width
        )
        image = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.out_channels, rows * patch_height, columns * patch_width
        )
        cropped: torch.Tensor = image[..., : self.output_shape[0], : self.output_shape[1]]
        return cropped


def buildTransformerStack(
    d_model: int,
    num_heads: int,
    num_layers: int,
    feedforward_dim: int | None = None,
    dropout: float = 0.0,
    activation: str = "gelu",
) -> nn.TransformerEncoder:
    """Build a stack of pre-norm transformer layers followed by a final layer norm.

    Pre-norm (`norm_first`) is the usual choice for a small transformer trained from scratch:
    each sub-layer sees a normalized input and the residual path stays untouched.

    Args:
        d_model: Token width.
        num_heads: Number of attention heads, dividing `d_model`.
        num_layers: Number of layers, at least `1`.
        feedforward_dim: Hidden width of each feed-forward block. `None` (default) is
            `4 * d_model`, the usual ratio.
        dropout: Dropout probability inside the layers.
        activation: `"gelu"` or `"relu"`.

    Returns:
        An `nn.TransformerEncoder` taking and returning `(batch, tokens, d_model)`.

    Raises:
        ValueError: If `num_heads` does not divide `d_model`, `num_layers` is below `1`,
            `feedforward_dim` is below `1`, `dropout` is outside `[0, 1)`, or `activation` is
            neither `"gelu"` nor `"relu"`.
    """
    checkAttentionWidth(d_model, num_heads)
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}.")
    if feedforward_dim is not None and feedforward_dim < 1:
        raise ValueError(f"feedforward_dim must be at least 1, got {feedforward_dim}.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
    if activation not in ("gelu", "relu"):
        raise ValueError(f"Unknown activation '{activation}'. Expected 'gelu' or 'relu'.")
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=feedforward_dim if feedforward_dim is not None else 4 * d_model,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
    )


def buildPoolingHead(
    in_features: int, hidden_dims: Sequence[int], latent_dim: int
) -> tuple[nn.Module, nn.Linear, nn.Linear]:
    """Build the layers between a pooled token and `(mu, logvar)`.

    Args:
        in_features: Width of the pooled token.
        hidden_dims: Widths of an optional MLP between the pooled token and the two output
            layers. Empty (default of the encoders) goes straight from the token to the outputs.
        latent_dim: Width of `mu` and `logvar`.

    Returns:
        `(head, to_mu, to_logvar)`: the optional MLP (an identity if `hidden_dims` is empty) and
        the two linear output layers.

    Raises:
        ValueError: If a hidden width or `latent_dim` is below `1`.
    """
    if latent_dim < 1 or any(width < 1 for width in hidden_dims):
        raise ValueError(
            f"latent_dim and head_hidden_dims must be at least 1, got {latent_dim} and "
            f"{list(hidden_dims)}."
        )
    layers: list[nn.Module] = []
    width = in_features
    for hidden in hidden_dims:
        layers.append(nn.Linear(width, hidden))
        layers.append(nn.GELU())
        width = hidden
    head: nn.Module = nn.Sequential(*layers) if layers else nn.Identity()
    return head, nn.Linear(width, latent_dim), nn.Linear(width, latent_dim)


def checkModalitySpecs(
    modalities: Mapping[str, Mapping[str, Any]],
    allowed_keys: Mapping[str, Sequence[str]],
    required_keys: Mapping[str, Sequence[str]],
    owner: str,
) -> None:
    """Validate the per-modality settings of a shared transformer encoder or decoder.

    A shared module is configured with one small mapping per modality, `kind` (`"series"` or
    `"image"`) plus the settings of that kind. The keys are checked here, so a typo or a missing
    size is an error that names the modality and the key, not a silent default.

    Args:
        modalities: Modality name -> its settings, `kind` included.
        allowed_keys: Kind -> the keys (besides `kind`) a modality of that kind may set.
        required_keys: Kind -> the keys (besides `kind`) a modality of that kind must set.
        owner: Name of the calling class, for the error messages.

    Raises:
        ValueError: If there is no modality, a name is empty, a modality has no valid `kind`, sets
            a key its kind does not take, or lacks a required one.
    """
    if not modalities:
        raise ValueError(f"{owner} needs at least one modality in `modalities`.")
    for name, spec in modalities.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{owner}: modality names must be non-empty strings, got {name!r}.")
        kind = spec.get("kind")
        if kind not in MODALITY_KINDS:
            raise ValueError(
                f"{owner}: modality '{name}' needs kind in {MODALITY_KINDS}, got {kind!r}."
            )
        unknown = sorted(set(spec) - {"kind", *allowed_keys[kind]})
        if unknown:
            raise ValueError(
                f"{owner}: modality '{name}' (kind '{kind}') has unknown key(s) {unknown}. "
                f"Allowed: {sorted({'kind', *allowed_keys[kind]})}."
            )
        missing = sorted(key for key in required_keys[kind] if key not in spec)
        if missing:
            raise ValueError(
                f"{owner}: modality '{name}' (kind '{kind}') is missing required key(s) {missing}."
            )
