#!/usr/bin/env python3
"""End-to-end example: the full pipeline this framework currently supports (spec §6.1
milestone 1), on simple synthetic 1D signals, built directly through the Python API
(see `02_config_driven_pipeline.py` for the same pipeline built from the `configs/`
YAML files instead).

This is a **runnable, self-contained** walkthrough of every stage the framework
supports today, using only synthetic in-memory data (no external dataset, matching
this framework's own testing style throughout): no real dataset exists yet (spec §11's
pairing question is still open, and dataset loading is permanently the caller's own
responsibility, spec §6.2), so an example that used one could not be run as-is by
someone cloning this repository. Everything below is 1D signals, single modality, no
fusion — exactly spec §6.1 milestone 1, the only configuration this framework fully
supports end to end so far (image encoders/decoders, and the other 7 configurations of
spec §2.1, are not built yet).

Stages covered, in order:

1. **Synthetic data generation** (`_synthetic_signal_data.py`), deliberately on an
   **irregular, per-sample grid**: each curve is measured at its own,
   randomly-perturbed positions, of its own length. This is the scenario a naive
   "just change the point count" resample cannot handle correctly (two curves
   resampled to the same *count* are not on the same *grid* unless their measured
   positions are taken into account).
2. **Coordinate-aware resampling** onto one common grid
   (`ResampleTransform(interpolation="scipy")`, spec §6.2): applied per-sample, before
   batching, since each curve's own positions differ (this is exactly the case a single
   shared, construction-time grid cannot express; see `ResampleTransform`'s own module
   docstring). The common grid is chosen as the *intersection* of every curve's own
   range, so no extrapolation is ever needed.
3. **A shared preprocessing pipeline** (`log` + `standardize`, spec §6.2), computed from
   the training split only and applied via one composed, invertible
   `ComposeTransform`.
4. **Model assembly**: `GlobalVae.createSingleLatent` with the real
   `1d_cnn_encoder_v1`/`1d_cnn_decoder_v1` (`OneDCnnEncoder`/`OneDCnnDecoder`), no
   fusion strategy (single modality), regularized with `free_bits_kl` rather than the
   plain `kl_standard_normal` default — see "On regularization" below.
5. **Training**: `Trainer`, with a linear warm-up beta schedule, a CSV logger, and
   `BestCheckpointCallback`.
6. **Evaluation**: `evaluation.evaluate`, reported to the console and saved as JSON.
7. **Visualization**: a latent-space scatter plot, a per-dimension KL bar chart, a
   reconstruction grid (shown back in original, pre-transform units via
   `pipeline.inverse`), and a loss-curve plot with reconstruction/total and
   regularization on separate axes (`plotLossCurves`'s `twin_metrics`, since the
   regularization loss lives on a much smaller scale and would otherwise be visually
   flattened by the reconstruction/total curves next to it).

On regularization: an earlier version of this example used the default
`kl_standard_normal` regularizer with a short beta warm-up, and reconstruction quality
was poor (R^2 around 0.3) no matter how much model capacity or data was added. That is
the signature of posterior collapse, not underfitting: `beta` reaching its full weight
before the decoder has learned to rely on `z` lets the encoder cheaply satisfy the KL
term by pushing every dimension's posterior toward the prior, after which more
capacity cannot help, since the extra capacity is never used either. `free_bits_kl`
(spec §2.3) exists specifically for this: it gives every latent dimension a small KL
budget it is never penalized for using, which is enough here to raise R^2 from ~0.3 to
~0.97. This is not a hyperparameter this example happens to need; a real dataset can
hit the same failure mode, and `losses/regularizers/`'s alternatives to plain KL exist
for exactly this reason.

Run:
    pip install -e ".[dev]"   # or at least ".[interpolation]" for scipy
    python examples/01_signal_vae_pipeline.py

Everything this script writes goes under `examples/outputs/01_signal_vae_pipeline/`
(created if missing; already covered by `.gitignore`'s `outputs/` pattern).
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file-only output: this script never opens an interactive window
import matplotlib.pyplot as plt
import numpy as np
import torch
from _synthetic_signal_data import (
    buildResampleTransform,
    buildSyntheticDataset,
    computeCommonGrid,
    resampleOntoCommonGrid,
)

from global_vae.data.transforms.compose import ComposeTransform
from global_vae.data.transforms.log import LogTransform
from global_vae.data.transforms.standardize import StandardizeTransform
from global_vae.evaluation.evaluate import evaluate
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.checkpoint import BestCheckpointCallback, loadCheckpoint
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed
from global_vae.visualization.latent_plot import (
    collectLatentParams,
    plotLatentSpace,
    plotPerDimensionKl,
)
from global_vae.visualization.loss_curves import plotLossCurves
from global_vae.visualization.reconstruction_plot import (
    collectReconstructions,
    plotReconstructionGrid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("global_vae.examples.signal_vae_pipeline")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "01_signal_vae_pipeline"

SEED = 0
COMMON_GRID_LENGTH = 128  # the fixed length every curve is resampled to (step 2)
NUM_TRAIN, NUM_VAL, NUM_TEST = 24000, 3000, 3000

LATENT_DIM = 16
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 2e-3
FREE_BITS = 1.0  # per-dimension KL budget; see "On regularization" above
WARMUP_STEPS = 1500  # beta warm-up length, in optimizer steps, not epochs
ENCODER_CHANNELS = (32, 64, 128)
DECODER_CHANNELS = (128, 64, 32)

def _modelKwargs(latent_dim: int, output_length: int) -> dict[str, Any]:
    """Shared encoder/decoder construction arguments (used both for training and for
    rebuilding an identical, freshly-initialized model to load the best checkpoint
    into for evaluation).

    Args:
        latent_dim: Dimensionality of the single latent space.
        output_length: Length the decoder must reconstruct (must match the common
            grid length every training/eval sample is resampled to).

    Returns:
        `encoder_kwargs`/`decoder_kwargs` ready to pass to
        `GlobalVae.createSingleLatent`.
    """
    return {
        "encoder_kwargs": {"signal": {"latent_dim": latent_dim, "hidden_channels": ENCODER_CHANNELS}},
        "decoder_kwargs": {
            "signal": {
                "latent_dim": latent_dim,
                "output_length": output_length,
                "hidden_channels": DECODER_CHANNELS,
                "seed_length": 16,
                "upsample_modes": "conv_transpose",
            }
        },
    }


def buildModel(latent_dim: int = LATENT_DIM, output_length: int = COMMON_GRID_LENGTH) -> GlobalVae:
    """Build the spec §6.1 milestone 1 model: one real encoder, one latent space, one
    real decoder, no fusion (single modality).

    Args:
        latent_dim: Dimensionality of the single latent space.
        output_length: Length the decoder must reconstruct.

    Returns:
        A freshly-initialized `GlobalVae`.
    """
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {"encoder": "1d_cnn_encoder_v1", "decoder": "1d_cnn_decoder_v1"}
        },
        latent_dim=latent_dim,
        regularizer_strategy="free_bits_kl",
        regularizer_kwargs={"z_fused": {"free_bits": FREE_BITS}},
        **_modelKwargs(latent_dim, output_length),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setGlobalSeed(SEED)
    rng = np.random.default_rng(SEED)

    logger.info("Step 1/7: generating synthetic curves on irregular per-sample grids.")
    train_curves = buildSyntheticDataset(NUM_TRAIN, rng)
    val_curves = buildSyntheticDataset(NUM_VAL, rng)
    test_curves = buildSyntheticDataset(NUM_TEST, rng)
    all_curves = train_curves + val_curves + test_curves
    logger.info(
        "  %d curves total (%d/%d/%d train/val/test), point counts range %d-%d.",
        len(all_curves),
        NUM_TRAIN,
        NUM_VAL,
        NUM_TEST,
        min(len(p) for p, _ in all_curves),
        max(len(p) for p, _ in all_curves),
    )

    logger.info("Step 2/7: resampling every curve onto one common grid (spec §6.2).")
    common_grid = computeCommonGrid(all_curves, COMMON_GRID_LENGTH)
    resample_transform = buildResampleTransform(common_grid)

    train_values = resampleOntoCommonGrid(train_curves, resample_transform)
    val_values = resampleOntoCommonGrid(val_curves, resample_transform)
    test_values = resampleOntoCommonGrid(test_curves, resample_transform)
    logger.info("  every curve now shares one grid, tensor shape %s.", tuple(train_values.shape))

    logger.info("Step 3/7: building and applying the log + standardize pipeline.")
    log_transform = LogTransform(eps=1e-6)
    log_train_values = log_transform.apply(train_values)
    # Statistics computed from the training split only, never guessed by the
    # transform itself (see StandardizeTransform's own docstring).
    train_mean = log_train_values.mean().item()
    train_std = log_train_values.std().item()
    pipeline = ComposeTransform(
        [LogTransform(eps=1e-6), StandardizeTransform(mean=train_mean, std=train_std)]
    )
    logger.info("  log-space training statistics: mean=%.4f, std=%.4f.", train_mean, train_std)

    def toBatches(values: torch.Tensor, batch_size: int) -> list[dict[str, torch.Tensor]]:
        preprocessed = pipeline.apply(values)
        return [
            {"signal": preprocessed[start : start + batch_size]}
            for start in range(0, preprocessed.shape[0], batch_size)
        ]

    train_batches = toBatches(train_values, BATCH_SIZE)
    val_batches = toBatches(val_values, BATCH_SIZE)
    test_batches = toBatches(test_values, BATCH_SIZE)

    logger.info(
        "Step 4/7: assembling the model (real OneDCnnEncoder/OneDCnnDecoder, no fusion, "
        "free_bits_kl regularizer)."
    )
    model = buildModel()
    logger.info(
        "  model built: %d encoder param(s), %d decoder param(s).",
        sum(p.numel() for p in model.encoders["signal"].parameters()),
        sum(p.numel() for p in model.decoders["signal"].parameters()),
    )

    logger.info("Step 5/7: training.")
    trainer = Trainer(
        model,
        device="cpu",
        optimizer_kwargs={"lr": LEARNING_RATE},
        beta_schedules={
            "z_fused": LinearWarmupBetaSchedule(
                warmup_steps=WARMUP_STEPS, start_value=0.0, end_value=1.0
            )
        },
        callbacks=[
            CsvLogger(OUTPUT_DIR / "metrics.csv"),
            # Monitors reconstruction, not "val/loss/total": free_bits_kl keeps the
            # regularization term close to a constant per-dimension floor
            # (latent_dim * free_bits), which dominates the *magnitude* of
            # "val/loss/total" without being the thing that actually distinguishes a
            # better epoch from a worse one here. Monitoring the dominated total would
            # pick checkpoints essentially at random with respect to reconstruction
            # quality -- the same reconstruction/regularization scale mismatch that
            # motivates plotLossCurves's own twin_metrics below, in a different guise.
            BestCheckpointCallback(OUTPUT_DIR / "best.pt", monitor="val/loss/reconstruction"),
        ],
        log_every_n_steps=500,
    )
    history = trainer.fit(train_batches, num_epochs=NUM_EPOCHS, val_dataloader=val_batches)
    logger.info(
        "  training complete: final train/val total loss = %.4f / %.4f.",
        history[-1]["train/loss/total"],
        history[-1]["val/loss/total"],
    )

    logger.info("Step 6/7: evaluating the best checkpoint on the held-out test split.")
    best_model = buildModel()
    loadCheckpoint(OUTPUT_DIR / "best.pt", model=best_model)
    results = evaluate(best_model, test_batches, device="cpu")
    print("\n" + results.summary() + "\n")
    results.save(OUTPUT_DIR / "evaluation.json")

    logger.info(
        "Step 7/7: saving figures (latent space, per-dimension KL, reconstructions, loss curves)."
    )
    mu, logvar = collectLatentParams(best_model, test_batches, "z_fused", device="cpu")
    latent_fig = plotLatentSpace(mu, title="Latent space (test split)")
    latent_fig.savefig(OUTPUT_DIR / "latent_space.png")
    plt.close(latent_fig)

    kl_fig = plotPerDimensionKl(mu, logvar, title="Per-dimension KL (test split)")
    kl_fig.savefig(OUTPUT_DIR / "per_dimension_kl.png")
    plt.close(kl_fig)

    originals, reconstructions = collectReconstructions(
        best_model, test_batches, "signal", device="cpu"
    )
    # inverse_transform brings both series back from (log, standardize)-space to the
    # original, physical intensity units on the common grid -- exactly the composed
    # inverse callable global_vae.config.data.buildTransformPipeline also produces.
    recon_fig = plotReconstructionGrid(
        originals,
        reconstructions,
        inverse_transform=pipeline.inverse,
        x_values=common_grid,
        max_examples=6,
        ncols=3,
        title="Test-set reconstructions (original units)",
        xlabel="position (common grid)",
        ylabel="intensity",
    )
    recon_fig.savefig(OUTPUT_DIR / "reconstructions.png")
    plt.close(recon_fig)

    # Reconstruction/total loss and the regularization loss live on very different
    # scales (regularization is kept small on purpose by free_bits_kl); twin_metrics
    # puts the latter on its own right-hand axis so neither curve visually flattens
    # the other.
    # Reconstruction and regularization live on very different scales here
    # (free_bits_kl keeps regularization near a roughly constant per-dimension floor,
    # so "total" behaves like regularization, not like reconstruction, and is left
    # out of this plot for that reason: it would not add a visually distinct curve).
    # twin_metrics puts regularization on its own right-hand axis so neither curve
    # flattens the other; log_scale on both axes since each still spans more than an
    # order of magnitude within its own group.
    loss_fig = plotLossCurves(
        history,
        metrics=["train/loss/reconstruction", "val/loss/reconstruction"],
        twin_metrics=["train/loss/regularization", "val/loss/regularization"],
        log_scale=True,
        twin_log_scale=True,
        title="Training curves (reconstruction left, regularization right)",
        ylabel="reconstruction loss",
        twin_ylabel="regularization loss",
    )
    loss_fig.savefig(OUTPUT_DIR / "loss_curves.png")
    plt.close(loss_fig)

    logger.info("Done. Every output was written to '%s'.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
