"""Building blocks of the time series backbones: missing values, padding and a masked loss.

See spec §6 and §10 (the new-modality checklist), roadmap P3-7 and P3-8, ADR 0042. A time series
differs from a generic 1D signal by what real recordings do: they have gaps (a sensor drops out)
and different lengths (a window cut at the end of a recording is shorter). Both are the same thing
to a model, **values that were not observed**, and this module gives them one convention: a `NaN`
in the batch tensor is an unobserved value. A gap is `NaN` in the middle, and the padding of a short
series up to the length of the batch is `NaN` at the end.

The convention needs no second tensor. The loss of the framework compares a decoder's
reconstruction with the batch tensor of the same name, so the batch tensor is the only place a mask
can travel, and a `NaN` in it carries the mask to the loss as well as to the encoder:

- `splitObservations` turns a series into its values (zero where unobserved) and its observed
  mask. `ObservedPatchEmbed` feeds both to the encoder as channels, so that a real zero is not
  mistaken for a gap, and returns the mask of the patches that hold no observed value at all, which
  the attention layers skip as keys and the pooling leaves out.
- `computeMaskedSquaredError` is a loss that counts the observed values only. The loss names of a
  YAML file (`mse`, ...) would return `NaN` on such a target, so this one goes to
  `Trainer(reconstruction_loss_fn=...)`, as the cross-entropy of the text modality does.
"""

import torch
import torch.nn.functional as functional
from torch import nn

from global_vae.utils.transformer_blocks import PatchEmbed1d


def splitObservations(x: torch.Tensor, in_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a series into its values and the mask of the values that were observed.

    Args:
        x: A float series `(batch, length)` (one channel) or `(batch, channels, length)`, `NaN`
            where unobserved. `length` is at least `1`.
        in_channels: The number of channels the caller was built for.

    Returns:
        `(values, observed)`, both `(batch, channels, length)`: the values with `0` where they were
        `NaN`, and a boolean mask, `True` where observed.

    Raises:
        ValueError: If `x` is not a float tensor of 2 or 3 dimensions, has no time step, or has a
            channel count other than `in_channels`.
    """
    if not x.is_floating_point():
        raise ValueError(f"A time series must be a float tensor (NaN marks a gap), got {x.dtype}.")
    series = x.unsqueeze(1) if x.dim() == 2 else x
    if series.dim() != 3:
        raise ValueError(
            "A time series must have shape (batch, length) or (batch, channels, length), got "
            f"{tuple(x.shape)}."
        )
    if series.shape[1] != in_channels:
        raise ValueError(
            f"The model was built for in_channels={in_channels}, got {series.shape[1]} channel(s) "
            f"in shape {tuple(x.shape)}."
        )
    if series.shape[2] < 1:
        raise ValueError(f"A time series needs at least one time step, got shape {tuple(x.shape)}.")
    observed = ~torch.isnan(series)
    values = torch.where(observed, series, torch.zeros_like(series))
    return values, observed


def computeEmptyPatchMask(observed: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Mark the patches that hold no observed value, in any channel.

    A series that is empty everywhere keeps its first patch unmasked, so attention never has every
    key masked, which would produce `NaN`.

    Args:
        observed: Boolean `(batch, channels, length)`, `True` where observed.
        patch_size: Number of time steps per patch, at least `1`.

    Returns:
        A boolean `(batch, ceil(length / patch_size))` tensor, `True` for an empty patch.
    """
    anywhere = observed.any(dim=1)
    padding = -anywhere.shape[1] % patch_size
    if padding:
        anywhere = functional.pad(anywhere, (0, padding), value=False)
    batch = anywhere.shape[0]
    has_data = anywhere.view(batch, -1, patch_size).any(dim=-1)
    empty = ~has_data
    keep_first = torch.zeros_like(empty)
    keep_first[:, 0] = True
    return empty & ~(empty.all(dim=1, keepdim=True) & keep_first)


class ObservedPatchEmbed(nn.Module):
    """Cut a series with gaps into patches and project each one to a `d_model` token.

    The values and the observed mask are stacked as `2 * in_channels` channels, so a patch tells
    the projection which of its zeros are values and which are gaps. The end of the series is padded
    with unobserved steps up to a whole patch, which is the same as the padding of a short series,
    so a series padded with `NaN` gives the same tokens as the unpadded one.
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: int) -> None:
        """Build the projection.

        Args:
            in_channels: Number of channels of the series, at least `1`.
            d_model: Token width, at least `1`.
            patch_size: Number of time steps per patch, at least `1`.

        Raises:
            ValueError: If any argument is below `1`.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be at least 1, got {in_channels}.")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed = PatchEmbed1d(2 * in_channels, d_model, patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of series with gaps.

        Args:
            x: `(batch, length)` or `(batch, in_channels, length)`, `NaN` where unobserved.

        Returns:
            `(tokens, empty)`: the tokens `(batch, ceil(length / patch_size), d_model)` and the
            boolean `(batch, num_tokens)` mask of the patches with no observed value.

        Raises:
            ValueError: As `splitObservations`.
        """
        values, observed = splitObservations(x, self.in_channels)
        features = torch.cat([values, observed.to(values.dtype)], dim=1)
        padding = -features.shape[-1] % self.patch_size
        if padding:
            features = functional.pad(features, (0, padding), value=0.0)
        return self.embed(features), computeEmptyPatchMask(observed, self.patch_size)


def computeMaskedSquaredError(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the squared error over the observed values of a batch of series.

    It is summed over the observed time steps and channels of a series and averaged over the
    batch, so with a weight of `0.5` it is the reconstruction term of the ELBO of a Gaussian
    decoder of unit variance, as `reduction="sum"` with weight `0.5` is for the other
    modalities. A `NaN` in the target is a value that was not observed: it adds nothing, and the
    reconstruction there receives no gradient (the target is zeroed first, since `NaN * 0` is
    `NaN`). On a series with no gap it is the sum of squared errors, so a model can be trained
    with it from the start.

    Pass it as the `reconstruction_loss_fn` of a `Trainer` with the default `"mean"` reduction,
    which leaves the sum alone (the `"sum"` reduction would multiply it by the length again).

    Args:
        reconstruction: The decoder's output, the same shape as `target`.
        target: The batch tensor, `NaN` where unobserved.

    Returns:
        A scalar tensor.

    Raises:
        ValueError: If the two shapes differ, or `reconstruction` has no batch axis.
    """
    if reconstruction.dim() < 2 or reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and target must have one shape (batch, ...), got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}."
        )
    observed = ~torch.isnan(target)
    filled = torch.where(observed, target, torch.zeros_like(target))
    squared = (reconstruction - filled).pow(2) * observed.to(reconstruction.dtype)
    loss: torch.Tensor = squared.sum() / reconstruction.shape[0]
    return loss
