#!/usr/bin/env python3
"""Standalone latent-space visualization script (spec §6.1 milestone 1: "the ability to
inspect training curves and visualize the latent space").

Loads a checkpoint into a model built by a user-supplied factory function, runs it over
a user-supplied dataloader factory function, and saves latent-space scatter plots, a
per-dimension KL bar chart, and (if the checkpoint carries one) a training-curve plot.
Reuses `visualization/`'s own plotting functions (`docs/adr/0009-visualization.md`);
no new plotting logic lives here.

Distinct from `scripts/evaluate.py`'s own figure export (`exportEvaluationFigures`,
which additionally produces reconstruction grids as part of a full metrics report):
this script is for quickly inspecting just the latent space and training curves,
without running a full evaluation pass, and adds a few capabilities
`exportEvaluationFigures` does not: coloring the scatter plot by an arbitrary batch
field (`--label-key`), restricting which latent spaces get plotted, and plotting
realized samples instead of the posterior mean (`--use-samples`).

Model construction and data loading stay the caller's own responsibility everywhere
else in this framework (data pipeline concerns are explicitly out of scope; no config
schema exists for it, spec §11), so this script does not hardcode either: both are
dynamically imported from `module.path:function_name` strings you supply, exactly like
`scripts/evaluate.py`.

Usage:
    python scripts/visualize_latent.py \\
        --checkpoint runs/model.pt \\
        --model-factory mypackage.models:build_model \\
        --dataloader-factory mypackage.data:build_dataloader \\
        --output-dir results/latent/

Where, in your own code (anywhere importable on `PYTHONPATH`):
    def build_model() -> GlobalVae:
        # exactly the same architecture the checkpoint was saved from
        return GlobalVae.createSingleLatent(...)

    def build_dataloader() -> Iterable[dict[str, torch.Tensor]]:
        # walked once per latent space plotted (plus once for label collection, if
        # --label-key is given), unlike scripts/evaluate.py's single pass; return
        # something re-iterable (a DataLoader, or a plain list of batches), not a
        # single-use generator.

Run `python scripts/visualize_latent.py --help` for the full option list.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only output: never opens an interactive window
import matplotlib.pyplot as plt
import torch

from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.utils.imports import importCallable as _importCallable
from global_vae.visualization.latent_plot import plotLatentSpace, plotPerDimensionKl
from global_vae.visualization.loss_curves import plotLossCurves

logger = logging.getLogger("global_vae.scripts.visualize_latent")


def _collectLatentParamsAndLabels(
    model: GlobalVae,
    dataloader: list[dict[str, torch.Tensor]],
    latent_name: str,
    label_key: str | None,
    device: torch.device,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Collect one latent space's `(mu, logvar)`, plus an optional label field, in lockstep.

    Not `visualization.latent_plot.collectLatentParams` plus a second pass over
    `label_key`: two separate passes could desynchronize sample ordering from label
    ordering the moment any batch is skipped (e.g. missing that latent space's
    modality, spec §5), so this collects both from the exact same forward passes
    instead.

    Args:
        model: The model to run.
        dataloader: Already materialized into a `list` (not just any `Iterable`):
            called once per latent space plotted, so a single-use iterator would
            silently only produce output for the first one.
        latent_name: Which latent space to collect (a key of `model.latent_spaces`).
        label_key: Optional batch key, not one of `model.encoders`' modality names,
            used purely to color the scatter plot (e.g. a class label your own
            dataloader includes alongside its modality tensors). `None` collects no
            labels.
        device: Batches are moved here before the forward pass.
        max_samples: Stop after collecting at least this many samples. `None`
            collects the entire dataloader.

    Returns:
        `(mu, logvar, labels)`, each shape `(N, ...)` on CPU; `labels` is `None` if
        `label_key` was `None`.

    Raises:
        ValueError: If no batch ever produced `latent_name`.
        KeyError: If `label_key` is given but missing from a batch that did produce
            `latent_name`.
    """
    collected_mu: list[torch.Tensor] = []
    collected_logvar: list[torch.Tensor] = []
    collected_labels: list[torch.Tensor] | None = [] if label_key is not None else None
    total = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            model_inputs = {
                name: tensor.to(device)
                for name, tensor in raw_batch.items()
                if name in model.encoders
            }
            outputs = model(model_inputs)
            if latent_name not in outputs["latent_params"]:
                continue

            mu, logvar = outputs["latent_params"][latent_name]
            collected_mu.append(mu.cpu())
            collected_logvar.append(logvar.cpu())
            if collected_labels is not None:
                if label_key not in raw_batch:
                    raise KeyError(
                        f"--label-key '{label_key}' is missing from a batch that did "
                        f"produce latent space '{latent_name}'."
                    )
                collected_labels.append(raw_batch[label_key])

            total += mu.shape[0]
            if max_samples is not None and total >= max_samples:
                break

    if not collected_mu:
        raise ValueError(
            f"Never observed latent space '{latent_name}' across the given dataloader."
        )

    mu_all = torch.cat(collected_mu, dim=0)
    logvar_all = torch.cat(collected_logvar, dim=0)
    labels_all = torch.cat(collected_labels, dim=0) if collected_labels is not None else None
    if max_samples is not None:
        mu_all, logvar_all = mu_all[:max_samples], logvar_all[:max_samples]
        if labels_all is not None:
            labels_all = labels_all[:max_samples]
    return mu_all, logvar_all, labels_all


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
        help="'module.path:function_name' returning an Iterable[dict[str, torch.Tensor]]; "
        "see this script's own module docstring for why it is walked more than once.",
    )
    parser.add_argument(
        "--device", default=None, help="'cpu', 'cuda', 'cuda:0', ... Defaults to auto-detect."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory PNG files are saved into. Defaults to "
        "'<checkpoint's parent directory>/visualizations'.",
    )
    parser.add_argument(
        "--latent-names",
        nargs="*",
        default=None,
        help="Which latent spaces to plot (model.latent_spaces keys). Defaults to "
        "every latent space in the model.",
    )
    parser.add_argument(
        "--latent-projection-method",
        default="auto",
        choices=("auto", "pca", "tsne", "umap", "none"),
        help="Forwarded to visualization.latent_plot.plotLatentSpace's method.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Scatter plot dimensionality.",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Plot a reparameterized sample from each latent space instead of the "
        "posterior mean (the default): shows the actual stochasticity the model "
        "samples from, at the cost of a different plot on every run.",
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="Optional batch key (not one of the model's modality names) used to color "
        "the scatter plot, e.g. a class label your own dataloader includes alongside "
        "its modality tensors.",
    )
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.01,
        help="Forwarded to visualization.latent_plot.plotPerDimensionKl.",
    )
    parser.add_argument(
        "--skip-kl", action="store_true", help="Skip the per-dimension KL bar chart."
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip the training-curve plot even if the checkpoint carries a non-empty history.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Bound memory/plot size for a very large dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Forwarded to stochastic projection methods (tsne/umap) for a reproducible layout.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the latent-visualization script.

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
    metadata = loadCheckpoint(args.checkpoint, model=model, map_location=resolved_device)
    logger.info("Loaded checkpoint from '%s' onto device '%s'.", args.checkpoint, resolved_device)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.checkpoint.parent / "visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = list(dataloader_factory())  # type: ignore[operator]

    latent_names = args.latent_names if args.latent_names else list(model.latent_spaces)
    unknown_latent_names = set(latent_names) - set(model.latent_spaces)
    if unknown_latent_names:
        raise ValueError(
            f"--latent-names references unknown latent space(s) {sorted(unknown_latent_names)}. "
            f"Available: {sorted(model.latent_spaces)}."
        )

    saved_paths: list[Path] = []
    for latent_name in latent_names:
        mu, logvar, labels = _collectLatentParamsAndLabels(
            model, dataloader, latent_name, args.label_key, resolved_device, args.max_samples
        )
        z = model.latent_spaces[latent_name].reparameterize(mu, logvar) if args.use_samples else mu

        latent_fig = plotLatentSpace(
            z,
            labels=labels,
            method=args.latent_projection_method,
            n_components=args.n_components,
            seed=args.seed,
            title=f"Latent space: {latent_name}",
            label_name=args.label_key or "label",
        )
        latent_path = output_dir / f"latent_{latent_name}.png"
        latent_fig.savefig(latent_path)
        plt.close(latent_fig)
        saved_paths.append(latent_path)
        logger.info("Saved '%s'.", latent_path)

        if not args.skip_kl:
            kl_fig = plotPerDimensionKl(
                mu,
                logvar,
                collapse_threshold=args.collapse_threshold,
                title=f"Per-dimension KL: {latent_name}",
            )
            kl_path = output_dir / f"kl_{latent_name}.png"
            kl_fig.savefig(kl_path)
            plt.close(kl_fig)
            saved_paths.append(kl_path)
            logger.info("Saved '%s'.", kl_path)

    if args.skip_history:
        pass
    elif metadata.history:
        history_fig = plotLossCurves(metadata.history)
        history_path = output_dir / "loss_curves.png"
        history_fig.savefig(history_path)
        plt.close(history_fig)
        saved_paths.append(history_path)
        logger.info("Saved '%s'.", history_path)
    else:
        logger.info("Checkpoint has no training history; skipping the loss-curve plot.")

    logger.info("Saved %d figure(s) to '%s'.", len(saved_paths), output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
