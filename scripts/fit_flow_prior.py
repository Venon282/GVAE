#!/usr/bin/env python3
r"""Fit a frozen flow prior to one latent space of a trained model (roadmap P6-8, ADR 0061).

The prior is what `flow_prior_alignment` pulls another space toward. It is fitted on the posterior
means that this space gives to its own unpaired samples (flow matching, as in Li et al.,
arXiv 2506.05240), written to one file, and never touched again:

    python scripts/fit_flow_prior.py \\
        --checkpoint runs/signal_solo/checkpoints/best.pt \\
        --model-factory my_project.models:buildPairModel \\
        --dataloader-factory my_project.data:buildUnpairedSignalLoader \\
        --space z_signal \\
        --dim 8 \\
        --output priors/z_signal.pt

`--dataloader-factory` returns the batches to encode: those that carry the modalities of the space
(`signal`); a batch that lacks one is skipped, so a mixed stream can be given as it is. `--dim` is
the width of the space that will be pulled, when it is not this one's: the points are then mapped
to it by a fixed random projection that the file keeps. The file is read with
`torch.load(weights_only=True)`. Next to it, `<output>.json` records what the fit did, so that the
loss it reached can be read against the reference loss that the term is judged by.

Run `python scripts/fit_flow_prior.py --help` for the full option list.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from global_vae.evaluation.space_means import collectSpaceMeans
from global_vae.losses.priors.flow_prior import fitFlowPrior, saveFlowPrior
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.utils.imports import importCallable as _importCallable

logger = logging.getLogger("fit_flow_prior")


def _buildArgumentParser() -> argparse.ArgumentParser:
    """Build this script's `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="A training checkpoint.")
    parser.add_argument(
        "--model-factory",
        required=True,
        help="'module.path:function_name' returning a GlobalVae of the checkpoint's architecture.",
    )
    parser.add_argument(
        "--dataloader-factory",
        required=True,
        help="'module.path:function_name' returning an iterable of batches; those that carry the "
        "modalities of the space are encoded.",
    )
    parser.add_argument(
        "--space",
        required=True,
        metavar="VIEW",
        help="The latent space to fit the prior to: a name (z_signal), or "
        "space@modality[+modality].",
    )
    parser.add_argument("--output", required=True, type=Path, help="The prior file to write.")
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="The width of the space that will be pulled, if not the width of --space: a fixed "
        "random projection maps the points to it (default: the width of --space).",
    )
    parser.add_argument("--hidden", type=int, default=128, help="Hidden width (default: 128).")
    parser.add_argument("--layers", type=int, default=3, help="Hidden layers (default: 3).")
    parser.add_argument(
        "--time-features", type=int, default=8, help="Fourier time features (default: 8)."
    )
    parser.add_argument("--steps", type=int, default=2000, help="Adam steps (default: 2000).")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the fit (default: 0).")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Stop after this many samples."
    )
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', ... (default: cuda if any).")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Read the checkpoint with weights_only=False. Only for a file you trust.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Fit and write the prior.

    Args:
        argv: Command-line arguments (excluding the program name). `None` uses `sys.argv[1:]`.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _buildArgumentParser()
    args = parser.parse_args(argv)
    model_factory = _importCallable(args.model_factory)
    dataloader_factory = _importCallable(args.dataloader_factory)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model: GlobalVae = model_factory()  # type: ignore[operator]
    model = model.to(device)
    loadCheckpoint(
        args.checkpoint,
        model=model,
        map_location=device,
        weights_only=not args.trust_checkpoint,
    )
    try:
        points = collectSpaceMeans(
            model,
            args.space,
            dataloader_factory(),  # type: ignore[operator]
            device=device,
            max_samples=args.max_samples,
        )
        fit = fitFlowPrior(
            points,
            dim=args.dim,
            hidden=args.hidden,
            layers=args.layers,
            time_features=args.time_features,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error.args[0]))
    saveFlowPrior(fit.prior, args.output, fit)
    summary = {
        "space": args.space,
        "checkpoint": str(args.checkpoint),
        "source_width": int(points.shape[1]),
        "dim": fit.prior.dim,
        "projected": fit.prior.projection is not None,
        "num_points": fit.num_points,
        "num_held_out": fit.num_held_out,
        "steps": fit.steps,
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "held_out_loss": fit.held_out_loss,
        "reference_loss": fit.reference_loss,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Fitted a flow prior to %d points of '%s' (width %d -> %d): flow matching loss %.4f -> "
        "%.4f, reference alignment loss %.4f%s. Written to '%s'.",
        fit.num_points,
        args.space,
        points.shape[1],
        fit.prior.dim,
        fit.initial_loss,
        fit.final_loss,
        fit.reference_loss,
        "" if fit.num_held_out else " (in sample: fewer than 100 points, none held out)",
        args.output,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
