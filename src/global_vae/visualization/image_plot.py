"""Image reconstruction visualization: original vs. reconstructed 2D images.

See spec §10 "reconstructions" and spec §6.1 milestone 2 (the image modality), and roadmap P3-4.

`reconstruction_plot.py` overlays two 1D series on one axis, which has no image counterpart: two
images cannot share a pixel grid. This module lays them side by side instead, with the same
conventions as its 1D sibling: every function returns a plain matplotlib `Figure` and never
saves or shows it, `inverse_transform` is the caller's own callable (spec §6: preprocessing lives
in the caller's data pipeline), and the collection helpers of `reconstruction_plot`
(`collectReconstructions`, `collectCrossModalReconstructions`) feed these functions unchanged,
since they already return tensors of any shape.

An image batch is `(N, H, W)` (one grayscale channel, what `OneDCnnEncoder`'s 2D siblings
produce) or `(N, C, H, W)` with `C` of 1 or 3 (grayscale or RGB); a batch of 1D series is
`(N, L)`. `isImageBatch` is the one place that tells them apart, so the evaluation exporters can
choose the right plot per modality.

Two functions:

- `plotImageReconstructionGrid`: one row per example, columns `original`, `reconstruction` and,
  optionally, the absolute error. The counterpart of `plotReconstructionGrid`.
- `plotImageCrossModalReconstructionMatrix`: for one decoder, one row per example and one column
  per input-modality subset (what the decoder produces from `signal` alone, from `image` alone,
  from both), next to the target and, optionally, the inputs that went in. The counterpart of
  `plotCrossModalReconstructionMatrix`, laid out differently on purpose: images are large, so the
  figure is per decoder with examples down the rows, where the 1D matrix puts every decoder in
  one figure and shows one example. Call it once per image decoder.

Colour scale: the original and the outputs of one row share a scale (the range of that row's
values, or the `vmin`/`vmax` given), so the same colour means the same value across a row. The
error panel has its own scale, from 0 to its row's maximum, which is printed under it.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from global_vae.visualization.reconstruction_plot import InverseTransform

_COLOR_CHANNELS = 3
_ERROR_FONT_SIZE = 8


def isImageBatch(batch: torch.Tensor) -> bool:
    """Tell a batch of images from a batch of 1D series by its number of dimensions.

    Args:
        batch: A batch tensor of one modality, as collected by `collectReconstructions`.

    Returns:
        `True` for `(N, H, W)` and `(N, C, H, W)` (an image batch), `False` otherwise, which in
        practice means `(N, L)`, a batch of 1D series. A 4D tensor with a channel count other
        than 1 or 3 is reported as an image batch too, and the plot functions reject it with a
        clear error.
    """
    return batch.dim() in (3, 4)


def _prepareImage(image: torch.Tensor, inverse_transform: InverseTransform | None) -> np.ndarray:
    """Turn one sample into a displayable array: `(H, W)` or `(H, W, 3)`.

    Args:
        image: One image, `(H, W)`, `(1, H, W)` or `(3, H, W)`.
        inverse_transform: Optional callable applied to the image first, undoing the caller's
            preprocessing (spec §6).

    Returns:
        A float32 NumPy array.

    Raises:
        ValueError: If the image is not `(H, W)`, `(1, H, W)` or `(3, H, W)`.
    """
    plotted = image.detach().cpu()
    if inverse_transform is not None:
        plotted = inverse_transform(plotted)
    if plotted.dim() == 3 and plotted.shape[0] == 1:
        plotted = plotted[0]
    elif plotted.dim() == 3 and plotted.shape[0] == _COLOR_CHANNELS:
        plotted = plotted.permute(1, 2, 0)
    elif plotted.dim() != 2:
        raise ValueError(
            f"an image must be (H, W), (1, H, W) or (3, H, W), got shape {tuple(plotted.shape)}."
        )
    return plotted.to(torch.float32).numpy()


def _sharedRange(
    arrays: Sequence[np.ndarray], vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """Return the colour range shared by several panels of one row.

    Args:
        arrays: The displayable arrays of the row.
        vmin: Lower bound given by the caller, or `None` to take the row's minimum.
        vmax: Upper bound given by the caller, or `None` to take the row's maximum.

    Returns:
        `(low, high)`.
    """
    low = vmin if vmin is not None else float(min(array.min() for array in arrays))
    high = vmax if vmax is not None else float(max(array.max() for array in arrays))
    return low, high


def _showImage(
    ax: Axes, array: np.ndarray, cmap: str, vmin: float | None, vmax: float | None
) -> None:
    """Draw one displayable array onto an axis, without ticks.

    Args:
        ax: The axis.
        array: `(H, W)` (coloured through `cmap`) or `(H, W, 3)` (RGB, clipped to `[0, 1]`).
        cmap: Matplotlib colour map name, for `(H, W)` arrays.
        vmin: Lower end of the colour scale, or `None` to autoscale.
        vmax: Upper end of the colour scale, or `None` to autoscale.
    """
    if array.ndim == 3:
        ax.imshow(np.clip(array, 0.0, 1.0))
    else:
        ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])


def plotImageReconstructionGrid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    inverse_transform: InverseTransform | None = None,
    max_examples: int = 8,
    show_error: bool = False,
    titles: Sequence[str] | None = None,
    title: str | None = None,
    original_label: str = "original",
    reconstruction_label: str = "reconstruction",
    error_label: str = "absolute error",
    cmap: str = "viridis",
    error_cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize_per_plot: tuple[float, float] = (2.4, 2.4),
) -> Figure:
    """Lay out original and reconstructed images side by side, one row per example.

    The image counterpart of `plotReconstructionGrid`. Columns are the original, the
    reconstruction and, with `show_error`, their absolute difference.

    Args:
        originals: Ground-truth images, `(N, H, W)` or `(N, C, H, W)` with `C` of 1 or 3.
        reconstructions: Reconstructed images, same shape as `originals`.
        inverse_transform: Optional callable applied to every original and reconstruction (one
            image at a time, as the data pipeline saw it) before plotting, undoing the caller's
            preprocessing. The error is computed after it, so it is in the plotted units.
        max_examples: Plot at most this many rows (the first `max_examples` images; choose
            which images to pass in yourself for a random subset).
        show_error: Add a column with the absolute error `|original - reconstruction|` (the mean
            over channels for RGB), scaled from 0 to its row's maximum, which is written under it.
        titles: Optional per-row labels, one per plotted example. Defaults to `"example 0"`,
            `"example 1"`, ...
        title: Optional whole-figure title (`fig.suptitle`).
        original_label: Title of the original column.
        reconstruction_label: Title of the reconstruction column.
        error_label: Title of the error column.
        cmap: Colour map of the original and reconstruction (grayscale images only).
        error_cmap: Colour map of the error column.
        vmin: Lower end of the colour scale of the original and the reconstruction. `None`
            (default) takes the minimum of the two images of each row.
        vmax: Upper end of the colour scale; `None` takes the maximum of the two images of each
            row.
        figsize_per_plot: Figure size of one panel; the figure scales with the grid.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If the tensors are empty, differ in shape, are not `(N, H, W)` or
            `(N, C, H, W)` with `C` of 1 or 3, `max_examples` is not positive, or `titles` has the
            wrong length.
    """
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"originals and reconstructions must have the same shape, got "
            f"{tuple(originals.shape)} and {tuple(reconstructions.shape)}."
        )
    if originals.numel() == 0:
        raise ValueError("plotImageReconstructionGrid received empty originals/reconstructions.")
    if not isImageBatch(originals):
        raise ValueError(
            f"images must be (N, H, W) or (N, C, H, W), got shape {tuple(originals.shape)}. "
            "For (N, L) series use plotReconstructionGrid."
        )
    if max_examples <= 0:
        raise ValueError(f"max_examples must be positive, got {max_examples}.")

    num_examples = min(originals.shape[0], max_examples)
    if titles is not None and len(titles) != num_examples:
        raise ValueError(
            f"titles has {len(titles)} entries but {num_examples} examples are plotted."
        )

    column_titles = [original_label, reconstruction_label] + ([error_label] if show_error else [])
    fig, axes = plt.subplots(
        num_examples,
        len(column_titles),
        figsize=(figsize_per_plot[0] * len(column_titles), figsize_per_plot[1] * num_examples),
        squeeze=False,
    )
    if title:
        fig.suptitle(title)

    for row in range(num_examples):
        original = _prepareImage(originals[row], inverse_transform)
        reconstruction = _prepareImage(reconstructions[row], inverse_transform)
        low, high = _sharedRange([original, reconstruction], vmin, vmax)
        _showImage(axes[row][0], original, cmap, low, high)
        _showImage(axes[row][1], reconstruction, cmap, low, high)
        if show_error:
            error = np.abs(original - reconstruction)
            if error.ndim == 3:
                error = error.mean(axis=2)
            _showImage(axes[row][2], error, error_cmap, 0.0, None)
            axes[row][2].set_xlabel(f"max {error.max():.3g}", fontsize=_ERROR_FONT_SIZE)
        axes[row][0].set_ylabel(titles[row] if titles is not None else f"example {row}")

    for column, column_title in enumerate(column_titles):
        axes[0][column].set_title(column_title, fontsize=9)
    fig.tight_layout()
    return fig


def _drawInput(
    ax: Axes, value: torch.Tensor, cmap: str, vmin: float | None, vmax: float | None
) -> None:
    """Draw one input sample: a line for a 1D series, an image otherwise.

    Args:
        ax: The axis.
        value: One sample, `(L,)` or an image (`(H, W)`, `(1, H, W)`, `(3, H, W)`).
        cmap: Colour map for a grayscale image.
        vmin: Lower end of the colour scale of an image, `None` to autoscale.
        vmax: Upper end of the colour scale of an image, `None` to autoscale.
    """
    if value.dim() == 1:
        ax.plot(value.detach().cpu().numpy(), linewidth=1.2)
        ax.tick_params(labelsize=7)
    else:
        _showImage(ax, _prepareImage(value, None), cmap, vmin, vmax)


def plotImageCrossModalReconstructionMatrix(
    collected: Mapping[frozenset[str], Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
    decoder_name: str,
    example_indices: Sequence[int] = (0,),
    inputs: Mapping[str, torch.Tensor] | None = None,
    subset_order: Sequence[frozenset[str]] | None = None,
    inverse_transform: InverseTransform | None = None,
    title: str | None = None,
    original_label: str = "original (target)",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize_per_plot: tuple[float, float] = (2.4, 2.4),
) -> Figure:
    """Show what one image decoder produces from each subset of input modalities (spec §5).

    The image counterpart of `plotCrossModalReconstructionMatrix`. One row per example; the
    columns are the `inputs` that went in (if given), the target, then the decoder's output under
    each input-modality subset. Because images are large, this draws one decoder per figure: call
    it once per image decoder.

    Args:
        collected: As returned by `collectCrossModalReconstructions`: input subset -> decoder name
            -> `(originals, reconstructions)`.
        decoder_name: The image decoder to draw. Its tensors must be image batches
            (`isImageBatch`).
        example_indices: Which samples to show, one row each; the same indices in every column,
            so a row is one underlying example under different input conditions.
        inputs: Optional context columns drawn before the target: modality name -> tensor
            `(N, ...)` aligned with `collected` (the same samples, in the same order). A `(N, L)`
            tensor is drawn as a line, an image batch as an image. Drawn as given: apply an
            inverse transform yourself if the values are in model space.
        subset_order: Explicit order of the output columns. `None` (default) sorts the subsets of
            `collected` by their label, so a subset comes before any superset of it.
        inverse_transform: Optional callable applied to the target and to every output of
            `decoder_name`, one image at a time, before plotting.
        title: Optional whole-figure title (`fig.suptitle`).
        original_label: Title of the target column.
        cmap: Colour map for grayscale images.
        vmin: Lower end of the colour scale. `None` (default) uses, per row, the range of the
            target and the outputs; image inputs are autoscaled on their own. Given, it applies to
            every image panel.
        vmax: Upper end of the colour scale; same rule.
        figsize_per_plot: Figure size of one panel; the figure scales with the grid.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `collected` is empty, `decoder_name` is in none of the requested subsets,
            its tensors are not image batches, `subset_order` names a subset absent from
            `collected`, `example_indices` is empty or out of range for a tensor drawn, or an
            input is neither `(N, L)` nor an image batch.
    """
    if not collected:
        raise ValueError("plotImageCrossModalReconstructionMatrix received an empty `collected`.")
    subsets = (
        list(subset_order)
        if subset_order is not None
        else sorted(collected, key=lambda subset: " + ".join(sorted(subset)))
    )
    unknown = [subset for subset in subsets if subset not in collected]
    if unknown:
        raise ValueError(
            f"subset_order names subset(s) {[sorted(subset) for subset in unknown]} that are "
            "not in `collected`."
        )
    present = [subset for subset in subsets if decoder_name in collected[subset]]
    if not present:
        available = sorted({name for per_decoder in collected.values() for name in per_decoder})
        raise ValueError(
            f"decoder {decoder_name!r} is in none of the requested input subsets. "
            f"Available decoders: {available}."
        )
    for subset in present:
        for tensor in collected[subset][decoder_name]:
            if not isImageBatch(tensor):
                raise ValueError(
                    f"decoder {decoder_name!r} does not produce images (shape "
                    f"{tuple(tensor.shape)}); use plotCrossModalReconstructionMatrix for series."
                )
    indices = list(example_indices)
    if not indices:
        raise ValueError("example_indices must not be empty.")

    target_batch = collected[present[0]][decoder_name][0]
    sized: dict[str, int] = {f"target of {decoder_name!r}": target_batch.shape[0]}
    for subset in present:
        sized[f"output of {decoder_name!r} from {sorted(subset)}"] = collected[subset][
            decoder_name
        ][1].shape[0]
    context = dict(inputs) if inputs is not None else {}
    for name, value in context.items():
        if not (value.dim() == 2 or isImageBatch(value)):
            raise ValueError(
                f"input {name!r} must be (N, L) or an image batch, got shape {tuple(value.shape)}."
            )
        sized[f"input {name!r}"] = value.shape[0]
    for label, count in sized.items():
        out_of_range = [index for index in indices if not 0 <= index < count]
        if out_of_range:
            raise ValueError(
                f"example_indices {out_of_range} are out of range for the {label}, which has "
                f"{count} example(s)."
            )

    column_titles = (
        [f"{name} (input)" for name in context]
        + [original_label]
        + [f"output from\n{' + '.join(sorted(subset))}" for subset in subsets]
    )
    fig, axes = plt.subplots(
        len(indices),
        len(column_titles),
        figsize=(figsize_per_plot[0] * len(column_titles), figsize_per_plot[1] * len(indices)),
        squeeze=False,
    )
    if title:
        fig.suptitle(title)

    first_output_column = len(context) + 1
    for row, index in enumerate(indices):
        target = _prepareImage(target_batch[index], inverse_transform)
        outputs = {
            subset: _prepareImage(collected[subset][decoder_name][1][index], inverse_transform)
            for subset in present
        }
        low, high = _sharedRange([target, *outputs.values()], vmin, vmax)
        for column, value in enumerate(context.values()):
            _drawInput(axes[row][column], value[index], cmap, vmin, vmax)
        _showImage(axes[row][len(context)], target, cmap, low, high)
        for offset, subset in enumerate(subsets):
            ax = axes[row][first_output_column + offset]
            if subset in outputs:
                _showImage(ax, outputs[subset], cmap, low, high)
            else:
                ax.axis("off")  # the decoder did not run under this subset
        axes[row][0].set_ylabel(f"example {index}")

    for column, column_title in enumerate(column_titles):
        axes[0][column].set_title(column_title, fontsize=9)
    fig.tight_layout()
    return fig
