#!/usr/bin/env python3
r"""Standalone evaluation script (spec: "un script/mode d'éval distinct de l'entraînement").

Loads a checkpoint into a model built by a user-supplied factory function, evaluates it
against a user-supplied test-dataloader factory function, prints a summary, and
optionally saves a JSON report and reconstruction/latent-space figures, plus a
cross-modal reconstruction matrix (spec §5,
`docs/adr/0016-cross-modal-reconstruction-reporting.md`) for any model with more
than one modality.

Model construction and data loading stay the caller's own responsibility everywhere
else in this framework (data pipeline concerns are explicitly out of scope; no config
schema is defined yet, spec §11), so this script does not hardcode either: both are
dynamically imported from `module.path:function_name` strings you supply, rather than
this script assuming any particular encoder/decoder/routing-graph choice or dataset
format.

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/model.pt \
        --model-factory mypackage.models:build_model \
        --dataloader-factory mypackage.data:build_test_dataloader \
        --output-dir results/

Where, in your own code (anywhere importable on `PYTHONPATH`):
    def build_model() -> GlobalVae:
        # exactly the same architecture the checkpoint was saved from
        return GlobalVae.createSingleLatent(...)

    def build_test_dataloader() -> Iterable[dict[str, torch.Tensor]]:
        # a torch.utils.data.DataLoader, or any re-iterable object; see
        # evaluation.evaluate's and evaluation.visual_export's own docstrings
        return DataLoader(MyTestDataset(...), batch_size=32)

Two options serve a very large or a paired test set (`docs/adr/0038-paired-data-evaluation.md`):
`--streaming` computes the reconstruction metrics batch by batch instead of pooling every
reconstruction, and `--paired`, for a model with several modalities, adds retrieval and alignment
between the modalities' posterior means (`paired_results.json`) and the cross-modal reconstruction
metrics. With `--output-dir` the per-modality table is written as `metrics_table.md` and
`metrics_table.csv`.

Reconstruction figures are plotted in whatever space the batch itself is in: if your own
data pipeline applied a `data.transforms` step (`log`/`standardize`/`resample`, spec
§6.2) before training, the figures show *that* space, not the original physical units,
unless you also supply `--inverse-transform-factory`:

    def build_inverse_transforms() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        # decoder/modality name -> callable undoing that modality's own preprocessing.
        # global_vae.config.data.buildTransformPipeline builds this straight from a
        # DataConfig, if you have one: {name: pipeline.inverse for name, pipeline in
        # buildTransformPipeline(data_config).items()}
        return {"signal": my_signal_pipeline.inverse}

Run `python scripts/evaluate.py --help` for the full option list.
"""

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

from global_vae.config.training import (
    DEFAULT_RECONSTRUCTION_LOSS,
    DEFAULT_RECONSTRUCTION_WEIGHT,
    resolveReconstructionLoss,
    resolveReconstructionReduction,
    resolveReconstructionWeight,
    validateReconstructionLoss,
    validateReconstructionWeight,
)
from global_vae.evaluation.cross_modal import (
    computeCrossModalReconstructionMetrics,
    exportCrossModalFigures,
)
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.latent_diagnostics import DEFAULT_ACTIVE_UNIT_THRESHOLD
from global_vae.evaluation.metrics_table import buildMetricsTable
from global_vae.evaluation.paired_alignment import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_KS,
    SIMILARITIES,
    PairedAlignmentResults,
    evaluatePairedAlignment,
    validateRetrievalSettings,
)
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.losses.reconstruction import DEFAULT_REDUCTION, validateReduction
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.utils.imports import importCallable as _importCallable
from global_vae.visualization.reconstruction_plot import collectCrossModalReconstructions

logger = logging.getLogger("global_vae.scripts.evaluate")


def _splitPairs(text: str, kind: str) -> str | dict[str, str]:
    """Split the value of a per-modality flag into a bare value or `modality -> value` text.

    Args:
        text: A bare value (`"sum"`, `"0.5"`), or a comma-separated list of `modality=value`
            pairs (for example `"signal=sum,image=mean"`).
        kind: What the value is called in the error message (`"name"`, `"number"`).

    Returns:
        The stripped text, or a modality -> text dict.

    Raises:
        argparse.ArgumentTypeError: If a pair has no `=`, or no modality before it.
    """
    if "=" not in text:
        return text.strip()
    pairs: dict[str, str] = {}
    for pair in text.split(","):
        modality, separator, value = pair.partition("=")
        if not separator or not modality.strip():
            raise argparse.ArgumentTypeError(
                f"expected 'modality={kind}' pairs separated by commas, got '{pair}'."
            )
        pairs[modality.strip()] = value.strip()
    return pairs


def _parseReduction(text: str) -> str | dict[str, str]:
    """Parse the value of `--reconstruction-reduction`.

    Args:
        text: A reduction name (`"mean"`, `"sum"`), or a comma-separated list of
            `modality=name` pairs (for example `"signal=sum,image=mean"`).

    Returns:
        The name, or a modality -> name dict.

    Raises:
        argparse.ArgumentTypeError: If a pair has no `=`, or a name is not supported.
    """
    reduction = _splitPairs(text, "name")
    try:
        validateReduction(reduction, "--reconstruction-reduction")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return reduction


def _parseLoss(text: str) -> str | dict[str, str]:
    """Parse the value of `--reconstruction-loss`.

    Args:
        text: A loss name (`"mse"`, `"bce"`, ...), or a comma-separated list of `modality=name`
            pairs (for example `"signal=mse,mask=bce"`).

    Returns:
        The name, or a modality -> name dict.

    Raises:
        argparse.ArgumentTypeError: If a pair has no `=`, or a name is not supported.
    """
    loss = _splitPairs(text, "name")
    try:
        validateReconstructionLoss(loss, "--reconstruction-loss")
    except (KeyError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error.args[0])) from error
    return loss


def _parseWeight(text: str) -> float | dict[str, float]:
    """Parse the value of `--reconstruction-weight`.

    Args:
        text: A number, or a comma-separated list of `modality=number` pairs (for example
            `"signal=0.5,image=2"`).

    Returns:
        The number, or a modality -> number dict.

    Raises:
        argparse.ArgumentTypeError: If a pair has no `=`, a value is not a number, or a number
            is negative or not finite.
    """
    flag = "--reconstruction-weight"
    raw = _splitPairs(text, "number")
    weight: float | dict[str, float]
    try:
        if isinstance(raw, dict):
            weight = {modality: float(value) for modality, value in raw.items()}
        else:
            weight = float(raw)
        validateReconstructionWeight(weight, flag)
    except ValueError as error:
        message = str(error)
        raise argparse.ArgumentTypeError(
            message if message.startswith(flag) else f"{flag}: {message}"
        ) from error
    return weight


def _buildArgumentParser() -> argparse.ArgumentParser:
    """Build this script's `argparse.ArgumentParser`.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a checkpoint saved by training.checkpoint.saveCheckpoint.",
    )
    parser.add_argument(
        "--model-factory",
        required=True,
        help="'module.path:function_name' returning a freshly-constructed GlobalVae "
        "with the exact architecture the checkpoint was saved from.",
    )
    parser.add_argument(
        "--dataloader-factory",
        required=True,
        help="'module.path:function_name' returning an Iterable[dict[str, torch.Tensor]] "
        "test dataloader.",
    )
    parser.add_argument(
        "--device", default=None, help="'cpu', 'cuda', 'cuda:0', ... Defaults to auto-detect."
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Read the checkpoint with weights_only=False, which runs pickle code: use it only "
        "for a file you trust and that the safe mode refuses (one saved before ADR 0034). "
        "See upgradeCheckpoint in global_vae.training.checkpoint for the permanent fix.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Bound memory for a very large test set; see evaluation.evaluate's own docstring.",
    )
    parser.add_argument(
        "--reconstruction-reduction",
        type=_parseReduction,
        default=DEFAULT_REDUCTION,
        help="The reduction the model was trained with (training.reconstruction_reduction): "
        "'mean' (default) or 'sum' for every modality, or 'modality=name' pairs separated "
        "by commas, e.g. 'signal=sum,image=mean'. It sets the scale of the total "
        "reconstruction loss in the report, so that it matches the training curves "
        "(docs/adr/0029-loss-scale-convention.md); the per-modality metrics do not depend on it.",
    )
    parser.add_argument(
        "--reconstruction-loss",
        type=_parseLoss,
        default=DEFAULT_RECONSTRUCTION_LOSS,
        help="The loss the model was trained with (training.reconstruction_loss): 'mse' "
        "(default), 'l1', 'bce' or 'smooth_l1' for every modality, or 'modality=name' pairs "
        "separated by commas, e.g. 'signal=mse,mask=bce'; a modality left out keeps 'mse'. "
        "It only affects the total reconstruction loss of the report "
        "(docs/adr/0030-per-modality-reconstruction-loss-and-weight.md).",
    )
    parser.add_argument(
        "--reconstruction-weight",
        type=_parseWeight,
        default=DEFAULT_RECONSTRUCTION_WEIGHT,
        help="The weight the model was trained with (training.reconstruction_weight): a "
        "number for every modality (default: %(default)s), or 'modality=number' pairs "
        "separated by commas, e.g. 'signal=0.5,image=2'; a modality left out keeps 1.0. "
        "It only affects the total reconstruction loss of the report, which together with "
        "--reconstruction-loss and --reconstruction-reduction puts it on the scale of the "
        "training curves.",
    )
    parser.add_argument(
        "--active-unit-threshold",
        type=float,
        default=DEFAULT_ACTIVE_UNIT_THRESHOLD,
        help="A latent dimension counts as an active unit when the variance of its posterior "
        "mean over the evaluation set is above this value (default: %(default)s).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Compute the reconstruction metrics (mse, rmse, mae, r2, pearson_r) batch by "
        "batch instead of over the pooled test set, so that the memory they need does not grow "
        "with it. The values equal the pooled ones up to rounding.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="For a model with several modalities, also report retrieval and alignment between "
        "the posterior means each modality gives on its own ('paired_results.json'), and the "
        "reconstruction metrics of every input subset in the table. Needs batches that carry the "
        "modalities of one sample together. Does nothing for a single-modality model.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_RETRIEVAL_KS),
        metavar="K",
        help="With --paired, the k of each recall@k (default: %(default)s).",
    )
    parser.add_argument(
        "--retrieval-similarity",
        default="cosine",
        choices=SIMILARITIES,
        help="With --paired, what retrieval ranks by: 'cosine' (default) or 'l2' (the negative "
        "squared Euclidean distance).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If given, save a JSON report ('results.json') and the per-modality table "
        "('metrics_table.md', 'metrics_table.csv') here, and (unless --no-figures) "
        "reconstruction/latent-space figures too.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip exporting figures even if --output-dir is given.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Forwarded to exportEvaluationFigures's reconstruction grid.",
    )
    parser.add_argument(
        "--latent-projection-method",
        default="auto",
        choices=("auto", "pca", "tsne", "umap", "none"),
        help="Forwarded to exportEvaluationFigures's latent-space plot.",
    )
    parser.add_argument(
        "--inverse-transform-factory",
        default=None,
        help="Optional 'module.path:function_name' returning a "
        "dict[str, Callable[[Tensor], Tensor]] (decoder/modality name -> inverse "
        "preprocessing callable), forwarded to the reconstruction figures so they show "
        "original-scale values instead of whatever preprocessing (data.transforms, spec "
        "§6.2) was applied before training. Without this, reconstruction figures are "
        "plotted in raw model-space (possibly transformed) units. See this script's own "
        "module docstring for how to build this dict from a DataConfig.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation script.

    Args:
        argv: Command-line arguments (excluding the program name). `None`
            (default) uses `sys.argv[1:]`, the normal CLI entry point;
            passing an explicit list is mainly useful for testing.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _buildArgumentParser()
    args = parser.parse_args(argv)
    if args.paired:
        try:
            validateRetrievalSettings(
                args.retrieval_k, args.retrieval_similarity, DEFAULT_CHUNK_SIZE
            )
        except ValueError as error:
            parser.error(f"--retrieval-k: {error}")

    model_factory = _importCallable(args.model_factory)
    dataloader_factory = _importCallable(args.dataloader_factory)

    model: GlobalVae = model_factory()  # type: ignore[operator]
    resolved_device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device)
    loadCheckpoint(
        args.checkpoint,
        model=model,
        map_location=resolved_device,
        weights_only=not args.trust_checkpoint,
    )
    logger.info("Loaded checkpoint from '%s' onto device '%s'.", args.checkpoint, resolved_device)

    want_figures = args.output_dir is not None and not args.no_figures
    dataloader = list(dataloader_factory()) if want_figures else dataloader_factory()  # type: ignore[operator]

    inverse_transforms = None
    if args.inverse_transform_factory is not None:
        inverse_transform_factory = _importCallable(args.inverse_transform_factory)
        inverse_transforms = inverse_transform_factory()  # type: ignore[operator]

    # The same resolvers as `buildTrainerFromConfig`, so that a setting that names a decoder the
    # model does not have is refused here too, instead of silently keeping its default.
    try:
        reconstruction_loss_fn = resolveReconstructionLoss(
            args.reconstruction_loss, model, "--reconstruction-loss"
        )
        reconstruction_weights = resolveReconstructionWeight(
            args.reconstruction_weight, model, "--reconstruction-weight"
        )
        reconstruction_reduction = resolveReconstructionReduction(
            args.reconstruction_reduction, model, "--reconstruction-reduction"
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error.args[0]))

    results = evaluate(
        model,
        dataloader,
        reconstruction_weights=reconstruction_weights,
        reconstruction_loss_fn=reconstruction_loss_fn,
        device=resolved_device,
        max_samples=args.max_samples,
        reconstruction_reduction=reconstruction_reduction,
        active_unit_threshold=args.active_unit_threshold,
        streaming=args.streaming,
    )
    print(results.summary())

    # Each paired pass reads the data afresh: the factory returns a new iterable, which a
    # single-use iterator would not give a second time.
    def freshDataloader() -> Iterable[dict[str, torch.Tensor]]:
        return dataloader if isinstance(dataloader, list) else dataloader_factory()  # type: ignore[operator]

    paired_results: PairedAlignmentResults | None = None
    cross_modal_metrics = None
    if args.paired and len(model.input_modalities) >= 2:
        paired_results = evaluatePairedAlignment(
            model,
            freshDataloader(),
            ks=args.retrieval_k,
            similarity=args.retrieval_similarity,
            device=resolved_device,
            max_samples=args.max_samples,
        )
        print(paired_results.summary())
        cross_modal_metrics = computeCrossModalReconstructionMetrics(
            collectCrossModalReconstructions(
                model,
                freshDataloader(),
                device=resolved_device,
                max_samples=args.max_samples,
            )
        )
    elif args.paired:
        logger.info("--paired does nothing: the model has fewer than two input modalities.")

    table = buildMetricsTable(
        results, cross_modal_metrics, omit_subsets=[frozenset(model.input_modalities)]
    )
    if args.paired:
        print("\n" + table.toMarkdown())

    if args.output_dir is not None:
        report_path = args.output_dir / "results.json"
        results.save(report_path)
        logger.info("Saved evaluation report to '%s'.", report_path)
        for table_name in ("metrics_table.md", "metrics_table.csv"):
            table.save(args.output_dir / table_name)
        logger.info("Saved the per-modality table to '%s'.", args.output_dir)
        if paired_results is not None:
            paired_results.save(args.output_dir / "paired_results.json")
            logger.info("Saved the paired evaluation to '%s'.", args.output_dir)

        if want_figures:
            figure_paths = exportEvaluationFigures(
                model,
                dataloader,
                args.output_dir,
                device=resolved_device,
                max_examples=args.max_examples,
                latent_projection_method=args.latent_projection_method,
                inverse_transforms=inverse_transforms,
            )
            # No-op for a single-modality model (spec §6.1 milestone 1): see
            # exportCrossModalFigures's own docstring and
            # docs/adr/0016-cross-modal-reconstruction-reporting.md.
            figure_paths += exportCrossModalFigures(
                model,
                dataloader,
                args.output_dir,
                device=resolved_device,
                inverse_transforms=inverse_transforms,
            )
            logger.info("Saved %d figure(s) to '%s'.", len(figure_paths), args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
