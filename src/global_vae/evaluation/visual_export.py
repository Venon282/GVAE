"""Export evaluation figures to disk for visual inspection (spec: "éventuellement export
des reconstructions pour inspection visuelle").

Pure wiring, no new plotting logic: reuses `visualization/`'s collection and plotting
functions (`docs/adr/0009-visualization.md`) and simply saves the resulting figures as
PNG files under `output_dir`.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.latent_plot import (
    collectLatentParams,
    plotLatentSpace,
    plotPerDimensionKl,
)
from global_vae.visualization.reconstruction_plot import (
    collectReconstructions,
    plotReconstructionGrid,
)

InverseTransform = Callable[[torch.Tensor], torch.Tensor]


def exportEvaluationFigures(
    model: GlobalVae,
    dataloader: list[dict[str, torch.Tensor]],
    output_dir: str | Path,
    device: str | torch.device | None = None,
    max_examples: int = 8,
    latent_projection_method: str = "auto",
    inverse_transforms: dict[str, InverseTransform] | None = None,
) -> list[Path]:
    """Save a reconstruction grid, a latent-space scatter plot, and a per-dimension KL
    bar chart for every modality/latent space `model` has, under `output_dir`.

    Args:
        model: The model to visualize.
        dataloader: Yields `dict[str, torch.Tensor]` batches. A plain
            `list` (not just any `Iterable`) since it is walked twice
            here (once per modality/latent space collected), unlike
            `evaluate`'s single pass; pass
            `list(your_dataloader)` if you have a single-use iterator.
        output_dir: Directory PNG files are saved into (created if
            missing).
        device: Batches are moved here before the forward pass.
            Defaults to `model`'s own device.
        max_examples: Forwarded to `plotReconstructionGrid`.
        latent_projection_method: Forwarded to `plotLatentSpace`'s
            `method` (`"auto"`/`"pca"`/`"tsne"`/`"umap"`/`"none"`).
        inverse_transforms: Modality name -> inverse-preprocessing
            callable, forwarded to `plotReconstructionGrid`'s own
            `inverse_transform` for that modality (spec §6: this
            framework has no built-in transforms of its own to invert,
            see `visualization.reconstruction_plot`'s module
            docstring). Modalities absent from this dict are plotted
            in raw model-space values, unchanged.

    Returns:
        Paths of every PNG file written, in the order written.
    """
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    inverse_transforms = inverse_transforms or {}
    written_paths: list[Path] = []

    for modality_name in model.decoders:
        originals, reconstructions = collectReconstructions(
            model, dataloader, modality_name, device=device
        )
        fig = plotReconstructionGrid(
            originals,
            reconstructions,
            inverse_transform=inverse_transforms.get(modality_name),
            max_examples=max_examples,
            title=f"Reconstructions: {modality_name}",
        )
        path = resolved_output_dir / f"reconstructions_{modality_name}.png"
        fig.savefig(path)
        plt.close(fig)
        written_paths.append(path)

    for latent_name in model.latent_spaces:
        mu, logvar = collectLatentParams(model, dataloader, latent_name, device=device)

        latent_fig = plotLatentSpace(
            mu, method=latent_projection_method, title=f"Latent space: {latent_name}"
        )
        latent_path = resolved_output_dir / f"latent_{latent_name}.png"
        latent_fig.savefig(latent_path)
        plt.close(latent_fig)
        written_paths.append(latent_path)

        kl_fig = plotPerDimensionKl(mu, logvar, title=f"Per-dimension KL: {latent_name}")
        kl_path = resolved_output_dir / f"kl_{latent_name}.png"
        kl_fig.savefig(kl_path)
        plt.close(kl_fig)
        written_paths.append(kl_path)

    return written_paths
