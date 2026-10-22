"""Cross-modal reconstruction reporting: what each decoder produces from a subset of inputs.

See spec §5: "the model can be trained and queried with any subset of available
modalities".

`GlobalVae.forward` already supports this with no change needed anywhere in this
framework: when a latent space is fed by more than one encoder but only a subset of
them is present in `inputs`, Fusion is only invoked once more than one encoder is
active; with exactly one active encoder, that latent space's `(mu, logvar)` is
simply that encoder's own output, and every decoder consuming the latent space still
runs regardless of which encoder(s) produced it (`models/global_vae.py`). A decoder
that consumes several latent spaces and is missing some of them (the private latent of
a modality that was left out, in a shared plus private model) runs only if its
assembler accepts a subset (`AbstractAssembler.handlesMissingLatents`: `sum`,
`average`); with `concat` it is absent from that subset's results, and the matrix
simply has no cell for it (ADR 0024). This
module is purely a reporting layer over that existing mechanism
(`docs/adr/0016-cross-modal-reconstruction-reporting.md`): it never changes how
`GlobalVae.forward` behaves, it only runs it under several input-modality subsets
and lays the results out for comparison, both as figures
(`visualization.reconstruction_plot.plotCrossModalReconstructionMatrix`) and as
numbers (`computeCrossModalReconstructionMetrics`, reusing `evaluation.metrics`'s
existing metric functions). The figure for a decoder of 1D series is that matrix; the figure
for a decoder of images is `visualization.image_plot.plotImageCrossModalReconstructionMatrix`,
one file per image decoder.

Deliberately its own module, separate from `evaluate()`/`exportEvaluationFigures()`
(`evaluate.py`/`visual_export.py`): cross-modal reporting is opt-in and, for a
single-modality model (spec §6.1 milestone 1), does not apply at all (there is only
ever one possible input subset), so every existing `evaluate()`/
`exportEvaluationFigures()` call site keeps its current behavior unchanged, and a
single-modality caller never has to reason about a no-op branch inside its own
evaluation pass. `exportCrossModalFigures` reflects this directly: it returns an
empty list for a model with fewer than two input modalities, rather than raising, matching
this framework's "absent, not an error" convention for a configuration that simply
does not apply (see e.g. `GlobalVae.forward`'s own handling of a latent space with
no active encoder this pass).
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from global_vae.evaluation.metrics import DEFAULT_RECONSTRUCTION_METRICS, MetricFn
from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.image_plot import (
    isImageBatch,
    plotImageCrossModalReconstructionMatrix,
)
from global_vae.visualization.reconstruction_plot import (
    InverseTransform,
    collectCrossModalReconstructions,
    plotCrossModalReconstructionMatrix,
)


def computeCrossModalReconstructionMetrics(
    collected: dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]],
    metrics: dict[str, MetricFn] | None = None,
) -> dict[frozenset[str], dict[str, dict[str, float]]]:
    """Compute reconstruction metrics for every (input subset, decoder) cell.

    A thin wrapper: every metric function is the exact same one
    `evaluation.evaluate` uses (`evaluation.metrics`), applied once per cell of
    `collected` instead of once per decoder. Pairs directly with
    `visualization.reconstruction_plot.plotCrossModalReconstructionMatrix`'s own
    layout: the two together give both a qualitative (figure) and quantitative
    (this function) view of the same comparison, e.g. to answer "how much does
    reconstructing 'image' from 'signal' alone actually cost in MSE", not only
    "what does it look like".

    Args:
        collected: As returned by
            `visualization.reconstruction_plot.collectCrossModalReconstructions`.
        metrics: Metric name -> `(reconstruction, target) -> float`. Defaults to
            `evaluation.metrics.DEFAULT_RECONSTRUCTION_METRICS` (mse, rmse, mae,
            r2, pearson_r), matching `evaluate`'s own default.

    Returns:
        Input subset -> decoder name -> metric name -> value, computed once over
        the entire pooled `(originals, reconstructions)` pair for that cell
        (matching `evaluate`'s own pooled, not batch-averaged, computation for
        these same metrics, for the same reason: R^2/Pearson r are not correctly
        computable as an average of per-batch values).

    Raises:
        ValueError: If `collected` is empty.
    """
    if not collected:
        raise ValueError("computeCrossModalReconstructionMetrics received an empty `collected`.")
    resolved_metrics = metrics or DEFAULT_RECONSTRUCTION_METRICS

    return {
        subset: {
            decoder_name: {
                metric_name: metric_fn(reconstructions, originals)
                for metric_name, metric_fn in resolved_metrics.items()
            }
            for decoder_name, (originals, reconstructions) in per_decoder.items()
        }
        for subset, per_decoder in collected.items()
    }


def exportCrossModalFigures(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    output_dir: str | Path,
    input_subsets: list[frozenset[str]] | None = None,
    device: str | torch.device | None = None,
    use_mean: bool = True,
    example_index: int = 0,
    inverse_transforms: dict[str, InverseTransform] | None = None,
    max_samples: int | None = None,
) -> list[Path]:
    """Save the cross-modal reconstruction figures if `model` has more than one modality.

    Decoders of 1D series share one matrix figure (`cross_modal_reconstructions.png`, written
    only if the model has such a decoder); every decoder that produces images gets a figure of
    its own (`cross_modal_reconstructions_<decoder>.png`), since images are too large to tile
    several decoders and several input subsets in one picture.

    No-op (returns an empty list, writes nothing) for a model with fewer than two
    input modalities (`model.input_modalities`, which counts the modalities of a
    shared encoder one by one, ADR 0027): there is only ever one possible input
    subset in that case, so there is nothing cross-modal to report. This lets a
    caller call this function unconditionally from a reporting pipeline (e.g.
    `scripts/evaluate.py`) without special-casing single-modality models
    itself; a genuinely misused explicit `input_subsets` (e.g. an empty one, or
    one naming an unknown modality) still raises `ValueError` from
    `collectCrossModalReconstructions`, not silently swallowed here.

    Args:
        model: The model to report on.
        dataloader: Yields `dict[str, torch.Tensor]` batches. Any `Iterable`
            works: unlike `evaluation.visual_export.exportEvaluationFigures`,
            this only ever walks it once (see
            `visualization.reconstruction_plot.collectCrossModalReconstructions`).
        output_dir: Directory the figure is saved into (created if missing).
        input_subsets: Forwarded to `collectCrossModalReconstructions`. `None`
            (default) uses `resolveDefaultInputSubsets(model)`.
        device: Batches are moved here before the forward pass. Defaults to
            `model`'s own device.
        use_mean: Forwarded to `collectCrossModalReconstructions`.
        example_index: The sample shown, forwarded to `plotCrossModalReconstructionMatrix`
            and, as the only row, to `plotImageCrossModalReconstructionMatrix`.
        inverse_transforms: Decoder name -> inverse-preprocessing callable. Forwarded as
            `plotCrossModalReconstructionMatrix`'s own `inverse_transform` (already
            per-decoder-name, matching `exportEvaluationFigures`'s own parameter of the same
            name), and, one entry at a time, as `plotImageCrossModalReconstructionMatrix`'s.
        max_samples: Forwarded to `collectCrossModalReconstructions`.

    Returns:
        Paths of every PNG file written: `cross_modal_reconstructions.png` first (if any
        decoder produces 1D series), then one `cross_modal_reconstructions_<decoder>.png` per
        image decoder, sorted by name. `[]` if `model` has fewer than two input modalities.
    """
    if len(model.input_modalities) < 2:
        return []

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    collected = collectCrossModalReconstructions(
        model,
        dataloader,
        input_subsets=input_subsets,
        device=device,
        use_mean=use_mean,
        max_samples=max_samples,
    )
    image_decoders = sorted(
        {
            decoder_name
            for per_decoder in collected.values()
            for decoder_name, (originals, _) in per_decoder.items()
            if isImageBatch(originals)
        }
    )
    series_collected = {
        subset: series
        for subset, per_decoder in collected.items()
        if (series := {k: v for k, v in per_decoder.items() if k not in image_decoders})
    }

    written_paths: list[Path] = []
    if series_collected:
        fig = plotCrossModalReconstructionMatrix(
            series_collected,
            example_index=example_index,
            inverse_transform=inverse_transforms,
            title="Cross-modal reconstructions",
        )
        path = resolved_output_dir / "cross_modal_reconstructions.png"
        fig.savefig(path)
        plt.close(fig)
        written_paths.append(path)

    for decoder_name in image_decoders:
        image_fig = plotImageCrossModalReconstructionMatrix(
            collected,
            decoder_name,
            example_indices=[example_index],
            inverse_transform=(inverse_transforms or {}).get(decoder_name),
            title=f"Cross-modal reconstructions: {decoder_name}",
        )
        image_path = resolved_output_dir / f"cross_modal_reconstructions_{decoder_name}.png"
        image_fig.savefig(image_path)
        plt.close(image_fig)
        written_paths.append(image_path)
    return written_paths
