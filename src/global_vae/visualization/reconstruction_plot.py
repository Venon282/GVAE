"""Reconstruction visualization: original vs. reconstructed 1D series (spec §10
"reconstructions"; spec §6.1 milestone 1's `signal -> z -> signal` case).

A line-overlay plot, not an image comparison: the only concrete decoder built so far
(`OneDCnnDecoder`) reconstructs a 1D series, matching spec §6's Phase-1 signal
modality. An image-comparison variant (side-by-side original/reconstruction images)
is a natural future addition once an image decoder exists (spec §6.1 milestone 2), not
built here.

`inverse_transform` is how this module respects the framework/data boundary: spec §6
is explicit that preprocessing (e.g. log-scale SAXS intensity) lives in the caller's
own data pipeline, entirely outside this framework's scope. This module therefore has
no built-in transforms of its own to invert; it only ever accepts the caller's own
inverse function as a plain callable, applied right before plotting.
"""

from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from global_vae.models.global_vae import GlobalVae

InverseTransform = Callable[[torch.Tensor], torch.Tensor]


def _plotOnePair(
    ax: Axes,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    inverse_transform: InverseTransform | None,
    x_values: torch.Tensor | None,
    original_label: str,
    reconstruction_label: str,
    xlabel: str,
    ylabel: str,
    title: str | None,
) -> None:
    """Draw one original/reconstruction overlay onto an already-created `Axes`."""
    if original.dim() != 1 or reconstruction.dim() != 1:
        raise ValueError(
            f"original and reconstruction must be 1-dimensional series, got shapes "
            f"{tuple(original.shape)} and {tuple(reconstruction.shape)}."
        )
    if original.shape != reconstruction.shape:
        raise ValueError(
            f"original and reconstruction must have the same length, got "
            f"{original.shape[0]} and {reconstruction.shape[0]}."
        )

    plotted_original = original.detach().cpu()
    plotted_reconstruction = reconstruction.detach().cpu()
    if inverse_transform is not None:
        plotted_original = inverse_transform(plotted_original)
        plotted_reconstruction = inverse_transform(plotted_reconstruction)

    x = x_values.detach().cpu().numpy() if x_values is not None else range(original.shape[0])
    ax.plot(x, plotted_original.numpy(), label=original_label, linewidth=1.5)
    ax.plot(
        x,
        plotted_reconstruction.numpy(),
        label=reconstruction_label,
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def plotReconstruction(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    inverse_transform: InverseTransform | None = None,
    x_values: torch.Tensor | None = None,
    title: str | None = None,
    original_label: str = "original",
    reconstruction_label: str = "reconstruction",
    xlabel: str = "index",
    ylabel: str = "value",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Figure:
    """Overlay a single original series against its reconstruction.

    Args:
        original: The ground-truth series, shape `(length,)`.
        reconstruction: The model's reconstruction of the same series,
            shape `(length,)`. Must match `original`'s length: this
            function never resamples or otherwise reconciles a length
            mismatch, matching `OneDCnnDecoder`'s own "verify the exact
            shape, never blur a mismatch away" philosophy.
        inverse_transform: Optional callable applied to both `original`
            and `reconstruction` before plotting, undoing whatever
            preprocessing the caller's own data pipeline applied (spec
            §6: preprocessing lives outside this framework, so it has
            no built-in transforms of its own to invert). `None`
            (default) plots the raw model-space values unchanged.
        x_values: Optional x-axis coordinates (e.g. the actual *q*
            values for a SAXS curve), same length as `original`.
            Defaults to a plain sample index `0, 1, 2, ...`.
        title: Plot title.
        original_label: Legend label for the original series.
        reconstruction_label: Legend label for the reconstructed series.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `original`/`reconstruction` are not
            1-dimensional or do not have the same length.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _plotOnePair(
        ax,
        original,
        reconstruction,
        inverse_transform,
        x_values,
        original_label,
        reconstruction_label,
        xlabel,
        ylabel,
        title,
    )
    fig.tight_layout()
    return fig


def plotReconstructionGrid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    inverse_transform: InverseTransform | None = None,
    x_values: torch.Tensor | None = None,
    max_examples: int = 8,
    ncols: int = 2,
    titles: Sequence[str] | None = None,
    original_label: str = "original",
    reconstruction_label: str = "reconstruction",
    xlabel: str = "index",
    ylabel: str = "value",
    figsize_per_plot: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Overlay several original/reconstruction pairs in a grid of subplots.

    Args:
        originals: Ground-truth series, shape `(N, length)`.
        reconstructions: Reconstructed series, shape `(N, length)`.
        inverse_transform: As in `plotReconstruction`, applied to every
            pair.
        x_values: As in `plotReconstruction`, shared across every
            subplot.
        max_examples: Plot at most this many pairs (the first
            `max_examples` rows of `originals`/`reconstructions`;
            select which rows to pass in yourself for e.g. a random
            subset).
        ncols: Number of subplot columns; the number of rows is
            derived from `min(N, max_examples)` and `ncols`.
        titles: Optional per-example subplot titles, length matching
            the number of examples actually plotted. Defaults to
            `"example 0"`, `"example 1"`, ...
        original_label: As in `plotReconstruction`.
        reconstruction_label: As in `plotReconstruction`.
        xlabel: As in `plotReconstruction`.
        ylabel: As in `plotReconstruction`.
        figsize_per_plot: Figure size of *one* subplot; the overall
            figure size scales with the grid shape.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `originals`/`reconstructions` are empty, have
            mismatched shapes, `ncols` is not positive, or `titles` is
            given with the wrong length.
    """
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"originals and reconstructions must have the same shape, got "
            f"{tuple(originals.shape)} and {tuple(reconstructions.shape)}."
        )
    if originals.numel() == 0:
        raise ValueError("plotReconstructionGrid received empty originals/reconstructions.")
    if ncols <= 0:
        raise ValueError(f"ncols must be positive, got {ncols}.")

    num_examples = min(originals.shape[0], max_examples)
    if titles is not None and len(titles) != num_examples:
        raise ValueError(
            f"titles has {len(titles)} entries but {num_examples} examples are plotted."
        )

    nrows = -(-num_examples // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for index in range(num_examples):
        ax = axes[index // ncols][index % ncols]
        example_title = titles[index] if titles is not None else f"example {index}"
        _plotOnePair(
            ax,
            originals[index],
            reconstructions[index],
            inverse_transform,
            x_values,
            original_label,
            reconstruction_label,
            xlabel,
            ylabel,
            example_title,
        )

    for index in range(num_examples, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.tight_layout()
    return fig


def collectReconstructions(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    modality_name: str,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `dataloader` and collect `(original, reconstruction)` pairs.

    Args:
        model: A `GlobalVae` instance. Does not call `model.eval()`
            itself; the caller decides the mode.
        dataloader: Yields `dict[str, torch.Tensor]` batches (modality
            name -> raw tensor, used as both encoder input and
            reconstruction target), the same convention `Trainer` uses.
        modality_name: Which modality/decoder to collect (a key of
            both the batch dicts and `model.decoders`).
        device: Batches are moved here before the forward pass.
            Defaults to `model`'s own device.
        max_samples: Stop after collecting at least this many pairs.
            `None` (default) collects the entire dataloader.

    Returns:
        `(originals, reconstructions)`, each shape `(N, ...)` matching
        that modality's own tensor shape, on CPU.

    Raises:
        ValueError: If `dataloader` yields no batches, or no batch
            ever produced a reconstruction for `modality_name`.
    """
    resolved_device = device if device is not None else next(model.parameters()).device
    collected_originals: list[torch.Tensor] = []
    collected_reconstructions: list[torch.Tensor] = []
    total = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = {name: tensor.to(resolved_device) for name, tensor in raw_batch.items()}
            outputs = model(batch)
            if modality_name not in outputs["reconstructions"]:
                continue
            collected_originals.append(batch[modality_name].cpu())
            collected_reconstructions.append(outputs["reconstructions"][modality_name].cpu())
            total += batch[modality_name].shape[0]
            if max_samples is not None and total >= max_samples:
                break

    if not collected_originals:
        raise ValueError(
            f"collectReconstructions never observed a reconstruction for modality "
            f"'{modality_name}' across the given dataloader."
        )
    originals_all = torch.cat(collected_originals, dim=0)
    reconstructions_all = torch.cat(collected_reconstructions, dim=0)
    if max_samples is not None:
        originals_all = originals_all[:max_samples]
        reconstructions_all = reconstructions_all[:max_samples]
    return originals_all, reconstructions_all
