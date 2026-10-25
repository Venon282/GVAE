"""Shared transformer decoder: one token stream out, split by modality (spec §2.1, P3-5)."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from global_vae.decoders.registry import registerDecoder
from global_vae.decoders.shared_base import AbstractSharedDecoder
from global_vae.utils.transformer_blocks import (
    IMAGE_KIND,
    SERIES_KIND,
    ImagePositions,
    PatchUnembed1d,
    PatchUnembed2d,
    SeriesPositions,
    buildTransformerStack,
    checkModalitySpecs,
    computeNumPatches,
    resolvePatchSize2d,
)

_ALLOWED_KEYS = {
    SERIES_KIND: ("output_length", "out_channels", "patch_size"),
    IMAGE_KIND: ("output_shape", "out_channels", "patch_size"),
}
_REQUIRED_KEYS = {SERIES_KIND: ("output_length",), IMAGE_KIND: ("output_shape",)}
_DEFAULT_SERIES_PATCH = 16
_DEFAULT_IMAGE_PATCH = 8


@registerDecoder("shared_transformer_decoder_v1")
class SharedTransformerDecoder(AbstractSharedDecoder):
    """One transformer reconstructing several modalities from one latent vector (the `D1` rows).

    There is one token per patch of every modality: its position vector (1D or 2D, sinusoidal),
    a learned embedding of its modality, and a linear projection of the latent vector, the same
    for every token. The tokens of all the modalities form one stream, one stack of pre-norm
    transformer layers lets them attend to one another, and the output stream is split back by
    modality: each part is projected to the patch values of its modality, tiled, and cropped to
    the requested length or shape (`PatchUnembed1d`, `PatchUnembed2d`).

    The decoder of `SharedTransformerEncoder`'s modalities: the same `kind` and `patch_size`
    settings, with the size of what to reconstruct instead of the channel count it reads.
    """

    def __init__(
        self,
        latent_dim: int,
        modalities: Mapping[str, Mapping[str, Any]],
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            modalities: Modality name -> its settings. The names are the keys of the returned
                dict, in the order of `output_modalities`. Each has `kind`, `"series"` or
                `"image"`; a series has `output_length` (required), an image `output_shape`
                (`[height, width]`, required); both take optionally `out_channels` (default `1`)
                and `patch_size` (an integer; for an image also `[height, width]`; default `16`
                for a series and `8` for an image).
            d_model: Token width. A multiple of `4` if any modality is an image.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.

        Raises:
            ValueError: If `modalities` is empty or a modality's settings are invalid (no
                `kind`, a missing size, a key its kind does not take), `num_heads` does not
                divide `d_model`, or another argument is out of range.
        """
        super().__init__()
        checkModalitySpecs(modalities, _ALLOWED_KEYS, _REQUIRED_KEYS, "SharedTransformerDecoder")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        self._output_modalities = tuple(modalities)
        self.latent_projection = nn.Linear(latent_dim, d_model)

        unembeds: dict[str, nn.Module] = {}
        positions: dict[str, nn.Module] = {}
        for name, spec in modalities.items():
            out_channels = int(spec.get("out_channels", 1))
            if spec["kind"] == SERIES_KIND:
                patch = int(spec.get("patch_size", _DEFAULT_SERIES_PATCH))
                length = int(spec["output_length"])
                unembeds[name] = PatchUnembed1d(d_model, patch, out_channels, length)
                positions[name] = SeriesPositions(d_model)
            else:
                output_shape = spec["output_shape"]
                if len(output_shape) != 2:
                    raise ValueError(
                        f"SharedTransformerDecoder: modality '{name}' needs output_shape="
                        f"[height, width], got {list(output_shape)}."
                    )
                shape = (int(output_shape[0]), int(output_shape[1]))
                patch_2d = resolvePatchSize2d(spec.get("patch_size", _DEFAULT_IMAGE_PATCH))
                computeNumPatches(shape[0], patch_2d[0])  # validates the sizes early
                unembeds[name] = PatchUnembed2d(d_model, patch_2d, out_channels, shape)
                positions[name] = ImagePositions(d_model)
        self.unembeds = nn.ModuleDict(unembeds)
        self.positions = nn.ModuleDict(positions)
        self.modality_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(d_model) * 0.02) for name in modalities}
        )
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )

    def _queries(self, name: str, latent: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Build the tokens of one modality before the shared layers.

        Args:
            name: The modality, one of `output_modalities`.
            latent: The projected latent vector, `(batch, 1, d_model)`.

        Returns:
            The `(batch, num_tokens, d_model)` tokens, and `num_tokens`.
        """
        unembed = self.unembeds[name]
        positions = self.positions[name]
        if isinstance(unembed, PatchUnembed1d) and isinstance(positions, SeriesPositions):
            table = positions(unembed.num_tokens, latent)
            count = unembed.num_tokens
        elif isinstance(unembed, PatchUnembed2d) and isinstance(positions, ImagePositions):
            table = positions(unembed.grid_shape[0], unembed.grid_shape[1], latent)
            count = unembed.num_tokens
        else:  # pragma: no cover  (the constructor builds only the two pairs above)
            raise TypeError(f"Modality '{name}' has an unexpected unembedding.")
        return table.unsqueeze(0) + self.modality_embeddings[name] + latent, count

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a batch of latent vectors into one reconstruction per modality.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Modality name -> reconstruction, with exactly the keys of `output_modalities`. A
            series has shape `(batch, output_length)` (`(batch, out_channels, output_length)`
            above one channel), an image `(batch, height, width)` (`(batch, out_channels, height,
            width)` above one channel), as the one-modality decoders return them.
        """
        latent = self.latent_projection(z).unsqueeze(1)
        queries: list[torch.Tensor] = []
        counts: list[int] = []
        for name in self._output_modalities:
            tokens, count = self._queries(name, latent)
            queries.append(tokens)
            counts.append(count)
        decoded = self.transformer(torch.cat(queries, dim=1))
        reconstructions: dict[str, torch.Tensor] = {}
        for name, part in zip(
            self._output_modalities, torch.split(decoded, counts, dim=1), strict=True
        ):
            output = self.unembeds[name](part)
            reconstructions[name] = output.squeeze(1) if output.shape[1] == 1 else output
        return reconstructions

    @property
    def output_modalities(self) -> tuple[str, ...]:
        """The modalities this decoder reconstructs, in the order the config lists them."""
        return self._output_modalities
