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

`resolveDefaultInputSubsets`/`collectCrossModalReconstructions`/
`plotCrossModalReconstructionMatrix` extend this to the cross-modal case (spec §5:
"the model can be trained and queried with any subset of available modalities"):
running the same batch through several input-modality subsets and laying out what
every decoder reconstructs under each, e.g. what an "image" decoder produces when
only "signal" is fed in. No change to `GlobalVae.forward` was needed for this
(`docs/adr/0016-cross-modal-reconstruction-reporting.md`); these three functions are
purely a reporting layer over behavior the model already has.
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
    title: str | None = None,
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
        title: Optional whole-figure title (`fig.suptitle`), distinct
            from `titles`' per-subplot ones, e.g. `"Reconstructions:
            signal"` when calling this once per modality. `None`
            (default) adds no overall title.
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
    if title:
        fig.suptitle(title)

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


def resolveDefaultInputSubsets(model: GlobalVae) -> list[frozenset[str]]:
    """Default input-modality subsets for `collectCrossModalReconstructions`.

    One singleton subset per encoder (spec §5: the model can be queried with
    any non-empty subset of its configured modalities), plus the full set of
    every encoder together, since that "everything present" case is the
    normal operating condition every partial subset is naturally compared
    against.

    Beyond singles and "everything", the space of partial subsets grows
    combinatorially (`2**N - 1` non-empty subsets for `N` modalities)
    without adding much that is independently informative as a *default*:
    for the fusion strategies this framework ships (spec §4), a fused
    posterior only depends on *which* encoders are active, not on any
    further structure. A caller wanting a specific further combination
    passes an explicit `input_subsets` to `collectCrossModalReconstructions`
    instead, e.g. `itertools.combinations(model.encoders, 2)` for every pair.

    Args:
        model: A `GlobalVae` instance.

    Returns:
        One `frozenset[str]` per encoder name, plus (only if there is more
        than one encoder, since it would otherwise exactly duplicate the
        one singleton) a further `frozenset` containing every encoder name.
        Empty if `model` has no encoders at all.
    """
    names = list(model.encoders)
    subsets = [frozenset({name}) for name in names]
    if len(names) > 1:
        subsets.append(frozenset(names))
    return subsets


def collectCrossModalReconstructions(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    input_subsets: Iterable[Iterable[str]] | None = None,
    device: str | torch.device | None = None,
    use_mean: bool = True,
    max_samples: int | None = None,
) -> dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Run `model` under several input-modality subsets and collect every
    resulting `(original, reconstruction)` pair (spec §5).

    For every batch and every subset in `input_subsets`, only that subset of
    the batch is fed to `model.forward` as `inputs`, mirroring
    `Trainer._applyModalityDropout`'s own "restrict the encoder input, keep
    the full batch as the reconstruction target" convention. Every decoder
    that produces a reconstruction this pass is paired against that
    decoder's own ground truth from the *full*, unrestricted batch, so a
    decoder can be compared against ground truth even when its own modality
    was withheld from this subset, e.g. reconstructing "image" from "signal"
    alone. This is exactly the behavior `GlobalVae.forward` already has
    (`docs/adr/0016-cross-modal-reconstruction-reporting.md`); this function
    only runs it under several subsets and collects the results.

    Args:
        model: A `GlobalVae` instance. Does not call `model.eval()` itself;
            the caller decides the mode.
        dataloader: Yields `dict[str, torch.Tensor]` batches (modality name
            -> raw tensor for the whole batch), the same convention
            `Trainer` uses. Walked exactly once regardless of how many
            subsets are requested (every subset is handled inside the same
            per-batch loop), so any `Iterable` works, single-use iterators
            included, unlike `evaluation.visual_export.exportEvaluationFigures`
            (which needs a `list`: it walks its dataloader once per
            modality/latent space).
        input_subsets: Which modality subsets to feed as `inputs`, each an
            iterable of modality names (a key of `model.encoders`). `None`
            (default) resolves via `resolveDefaultInputSubsets`.
        device: Batches are moved here before the forward pass. Defaults to
            `model`'s own device.
        use_mean: Forwarded to `GlobalVae.forward`. `True` (default,
            matching `evaluation.evaluate`'s own default, unlike
            `collectReconstructions`, which never sets it): deterministic
            reconstructions from the posterior mean, so a difference
            between two cells of the resulting matrix reflects which
            modalities were available, not sampling noise.
        max_samples: Stop collecting a given subset once it has at least
            this many samples (the last batch may slightly overshoot before
            being trimmed), tracked independently per subset. `None`
            (default) collects the entire dataloader for every subset.

    Returns:
        Input subset (as `frozenset[str]`, matching the entries of
        `input_subsets`) -> decoder name -> `(originals, reconstructions)`,
        each shape `(N, ...)` matching that decoder's own tensor shape, on
        CPU. A `(subset, decoder)` pair that never had both a reconstruction
        and a matching ground-truth batch key across the whole dataloader is
        simply absent from that subset's dict, not raised as an error,
        mirroring `GlobalVae.forward`'s own "absent, not an error" handling
        of a latent space or decoder with no available input this pass.

    Raises:
        ValueError: If `dataloader` yields no batches, if `input_subsets`
            (or its default) resolves to no subsets at all, or if any given
            subset is empty or references a name absent from
            `model.encoders`.
    """
    resolved_subsets = (
        [frozenset(subset) for subset in input_subsets]
        if input_subsets is not None
        else resolveDefaultInputSubsets(model)
    )
    if not resolved_subsets:
        raise ValueError(
            "collectCrossModalReconstructions requires at least one input subset, but "
            "none were given: input_subsets was empty, or resolveDefaultInputSubsets(model) "
            "produced none (model has no encoders)."
        )
    known_names = set(model.encoders)
    for subset in resolved_subsets:
        if not subset:
            raise ValueError("collectCrossModalReconstructions received an empty subset.")
        unknown = subset - known_names
        if unknown:
            raise ValueError(
                f"input_subsets references unknown modality name(s) {sorted(unknown)}. "
                f"Available: {sorted(known_names)}."
            )

    resolved_device = device if device is not None else next(model.parameters()).device
    collected_originals: dict[frozenset[str], dict[str, list[torch.Tensor]]] = {
        subset: {} for subset in resolved_subsets
    }
    collected_reconstructions: dict[frozenset[str], dict[str, list[torch.Tensor]]] = {
        subset: {} for subset in resolved_subsets
    }
    totals: dict[frozenset[str], int] = dict.fromkeys(resolved_subsets, 0)
    num_batches = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = {name: tensor.to(resolved_device) for name, tensor in raw_batch.items()}
            num_batches += 1

            for subset in resolved_subsets:
                if max_samples is not None and totals[subset] >= max_samples:
                    continue
                restricted = {name: tensor for name, tensor in batch.items() if name in subset}
                if not restricted:
                    continue
                outputs = model(restricted, use_mean=use_mean)
                for decoder_name, reconstruction in outputs["reconstructions"].items():
                    if decoder_name not in batch:
                        continue
                    collected_originals[subset].setdefault(decoder_name, []).append(
                        batch[decoder_name].cpu()
                    )
                    collected_reconstructions[subset].setdefault(decoder_name, []).append(
                        reconstruction.cpu()
                    )
                totals[subset] += next(iter(restricted.values())).shape[0]

            if max_samples is not None and all(
                totals[subset] >= max_samples for subset in resolved_subsets
            ):
                break

    if num_batches == 0:
        raise ValueError(
            "collectCrossModalReconstructions received an empty dataloader: at least "
            "one batch is required."
        )

    result: dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for subset in resolved_subsets:
        per_decoder: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for decoder_name, original_chunks in collected_originals[subset].items():
            originals = torch.cat(original_chunks, dim=0)
            reconstructions = torch.cat(collected_reconstructions[subset][decoder_name], dim=0)
            if max_samples is not None:
                originals = originals[:max_samples]
                reconstructions = reconstructions[:max_samples]
            per_decoder[decoder_name] = (originals, reconstructions)
        result[subset] = per_decoder
    return result


def _formatSubsetLabel(subset: frozenset[str]) -> str:
    """Human-readable label for an input-modality subset, e.g. `"image + signal"`.

    Args:
        subset: A non-empty set of modality names.

    Returns:
        Every name in `subset`, sorted, joined with `" + "`.
    """
    return " + ".join(sorted(subset))


def plotCrossModalReconstructionMatrix(
    collected: dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]],
    example_index: int = 0,
    row_order: Sequence[frozenset[str]] | None = None,
    column_order: Sequence[str] | None = None,
    inverse_transform: dict[str, InverseTransform] | None = None,
    x_values: dict[str, torch.Tensor] | None = None,
    title: str | None = None,
    original_label: str = "original",
    reconstruction_label: str = "reconstruction",
    xlabel: str = "index",
    ylabel: str = "value",
    figsize_per_plot: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Lay out every (input subset, decoder) cell of `collected` as a grid of
    original/reconstruction overlays (spec §5).

    Rows are input-modality subsets (`collected`'s keys); columns are every
    decoder name appearing in any of them. A cell is left blank (axis turned
    off, matching `plotReconstructionGrid`'s own convention for unused grid
    positions) wherever `collected[subset]` has no entry for that column's
    decoder, e.g. because that (subset, decoder) pair never appeared
    together in the dataloader `collectCrossModalReconstructions` was run
    over.

    Args:
        collected: As returned by `collectCrossModalReconstructions` (or any
            dict shaped the same way, e.g. hand-built for a single cell).
        example_index: Which sample (row index into every collected tensor)
            to plot; the same index is used in every cell, so every cell
            shows the exact same underlying example under a different input
            condition.
        row_order: Explicit row order. `None` (default) sorts `collected`'s
            keys by `_formatSubsetLabel` for a stable, readable order
            (alphabetical; a subset sorts before any of its supersets,
            since its label is always a prefix of theirs).
        column_order: Explicit column order (decoder names). `None`
            (default) is every decoder name appearing in any subset,
            sorted.
        inverse_transform: Decoder name -> callable applied to both the
            original and the reconstruction of that column before plotting
            (matching `evaluation.visual_export.exportEvaluationFigures`'s
            own `inverse_transforms` convention), since different columns
            can be different modalities with different preprocessing.
            `None` (default), or a decoder name absent from this dict,
            plots that column's raw model-space values unchanged.
        x_values: As `inverse_transform`: decoder name -> x-axis
            coordinates for that column.
        title: Optional whole-figure title (`fig.suptitle`).
        original_label: As in `plotReconstruction`.
        reconstruction_label: As in `plotReconstruction`.
        xlabel: As in `plotReconstruction`.
        ylabel: As in `plotReconstruction`.
        figsize_per_plot: Figure size of *one* subplot; the overall figure
            size scales with the grid shape.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `collected` is empty, or if `example_index` is out
            of range for some cell actually being plotted.
    """
    if not collected:
        raise ValueError("plotCrossModalReconstructionMatrix received an empty `collected`.")

    resolved_rows = (
        list(row_order) if row_order is not None else sorted(collected, key=_formatSubsetLabel)
    )
    all_decoders = {name for per_decoder in collected.values() for name in per_decoder}
    resolved_columns = list(column_order) if column_order is not None else sorted(all_decoders)

    nrows, ncols = len(resolved_rows), len(resolved_columns)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title)

    for row_index, subset in enumerate(resolved_rows):
        for col_index, decoder_name in enumerate(resolved_columns):
            ax = axes[row_index][col_index]
            entry = collected[subset].get(decoder_name)
            if entry is None:
                ax.axis("off")
                continue

            original, reconstruction = entry
            if not (0 <= example_index < original.shape[0]):
                raise ValueError(
                    f"example_index={example_index} is out of range for "
                    f"collected[{sorted(subset)}][{decoder_name!r}], which has "
                    f"{original.shape[0]} example(s)."
                )

            resolved_inverse = (
                inverse_transform.get(decoder_name) if inverse_transform is not None else None
            )
            resolved_x = x_values.get(decoder_name) if x_values is not None else None
            cell_title = f"in: {_formatSubsetLabel(subset)} -> out: {decoder_name}"
            _plotOnePair(
                ax,
                original[example_index],
                reconstruction[example_index],
                resolved_inverse,
                resolved_x,
                original_label,
                reconstruction_label,
                xlabel,
                ylabel,
                cell_title,
            )

    fig.tight_layout()
    return fig
