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
from pathlib import Path

import torch

from global_vae.evaluation.cross_modal import exportCrossModalFigures
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.utils.imports import importCallable as _importCallable

logger = logging.getLogger("global_vae.scripts.evaluate")


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
        "--max-samples",
        type=int,
        default=None,
        help="Bound memory for a very large test set; see evaluation.evaluate's own docstring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If given, save a JSON report ('results.json') here, and (unless "
        "--no-figures) reconstruction/latent-space figures too.",
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
    args = _buildArgumentParser().parse_args(argv)

    model_factory = _importCallable(args.model_factory)
    dataloader_factory = _importCallable(args.dataloader_factory)

    model: GlobalVae = model_factory()  # type: ignore[operator]
    resolved_device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device)
    loadCheckpoint(args.checkpoint, model=model, map_location=resolved_device)
    logger.info("Loaded checkpoint from '%s' onto device '%s'.", args.checkpoint, resolved_device)

    want_figures = args.output_dir is not None and not args.no_figures
    dataloader = list(dataloader_factory()) if want_figures else dataloader_factory()  # type: ignore[operator]

    inverse_transforms = None
    if args.inverse_transform_factory is not None:
        inverse_transform_factory = _importCallable(args.inverse_transform_factory)
        inverse_transforms = inverse_transform_factory()  # type: ignore[operator]

    results = evaluate(model, dataloader, device=resolved_device, max_samples=args.max_samples)
    print(results.summary())

    if args.output_dir is not None:
        report_path = args.output_dir / "results.json"
        results.save(report_path)
        logger.info("Saved evaluation report to '%s'.", report_path)

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
