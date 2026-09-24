#!/usr/bin/env python3
"""End-to-end example: (signal, image) -> image, with an output that is not the input.

Covers spec §2.1 `EN-L1-DN`, spec §4 (fusion), spec §5 (missing-modality robustness) and
spec §6 (image modality).

Every other example reconstructs what it was given (`signal -> z -> signal`). This one
shows the multimodal case the framework is built for: two encoders feed one fused latent
space, and a decoder produces a target that differs from every input. Synthetic data, no
external dataset, so it runs as-is after cloning.

**The task.** Each sample is a Gaussian blob with a hidden centre `(cx, cy)` and width
`sigma`. Three views of it exist:

- `signal`: a 1D profile along the x axis (a detector that integrates over y). It sees
  where the blob is horizontally and how wide it is, and is **blind to its vertical
  position**.
- `image_in`: a *degraded* picture of the blob (heavy noise plus a random occluded
  square). It carries the vertical position, but noisily and possibly hidden.
- `image_out`: the clean picture of the blob. This is the target the decoder must
  produce. It is different from `image_in`: returning the input would score poorly.

Neither input is enough alone, and they are complementary, which is the situation
Product-of-Experts fusion (spec §4) is for.

**Model.** `GlobalVae` built from an explicit `RoutingGraph`, not `createSingleLatent`:
that helper ties one decoder to every encoder name, whereas here the decoder is named
`image_out` and has no encoder of the same name (`GlobalVae.__init__` keeps
`encoder_configs` and `decoder_configs` independent for exactly this reason):

    signal   -> 1d_cnn_resnet_encoder_v1 -+
                                          +-> PoE -> z_fused -> 2d_cnn_resnet_decoder_v1
                                                                    -> image_out
    image_in -> 2d_cnn_resnet_encoder_v1 -+

**Training with a target that is not an input.** `Trainer` feeds the whole batch to
`GlobalVae.forward` and also uses it as the reconstruction target, and `forward` raises
`KeyError` on a key that has no encoder (`image_out` here). `TranslationTrainer` below
is the smallest fix: it overrides `_applyModalityDropout`, the one place `computeLosses`
selects what the encoders see, to keep only encoder inputs. The full batch is still the
reconstruction target, so nothing else changes. Modality dropout (spec §5) then randomly
hides `signal` or `image_in` during training, so the single fused latent learns to work
from either alone.

Stages, in order:

1. Synthetic data generation.
2. Model assembly (explicit routing graph, PoE fusion, `free_bits_kl` regularizer).
3. Training (`TranslationTrainer`, beta warm-up, CSV log, best checkpoint).
4. Evaluation by input subset (`collectCrossModalReconstructions`,
   `computeCrossModalReconstructionMetrics`, ADR 0016): what does the decoder produce
   from `signal` alone, `image_in` alone, and both? Compared against the trivial
   "return the degraded input" baseline.
5. Figures: a translation grid and the training curves.

Note on the figure: `visualization.reconstruction_plot.plotCrossModalReconstructionMatrix`
draws 1D series only (an image comparison plot is still listed as deferred in
`visualization/NOTE.md`), so the image grid below is example-local.

Run:
    pip install -e ".[dev]"
    python examples/03_signal_image_to_image.py
    python examples/03_signal_image_to_image.py --num-epochs 5 --num-train 256   # quick check

Everything this script writes goes under `examples/outputs/03_signal_image_to_image/`
(created if missing; already covered by `.gitignore`'s `outputs/` pattern).
"""

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only output: this script never opens an interactive window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from global_vae.evaluation.cross_modal import computeCrossModalReconstructionMetrics
from global_vae.evaluation.metrics import computeMse, computePearsonR, computeR2
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.checkpoint import BestCheckpointCallback, loadCheckpoint
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed
from global_vae.visualization.loss_curves import plotLossCurves
from global_vae.visualization.reconstruction_plot import collectCrossModalReconstructions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("global_vae.examples.signal_image_to_image")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "03_signal_image_to_image"

# Names are independent on the encoder and decoder sides: "image_in" is an encoder input,
# "image_out" a decoder target, and neither shares a name with the other.
SIGNAL = "signal"
IMAGE_IN = "image_in"
IMAGE_OUT = "image_out"
LATENT_NAME = "z_fused"

IMAGE_SIZE = 32
SIGNAL_LENGTH = 64
NOISE_STD = 0.5  # noise added to the degraded input image
OCCLUSION_SIZE = 14  # side of the square erased from the degraded input image
SIGNAL_NOISE_STD = 0.02

LATENT_DIM = 8
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
FREE_BITS = 0.5  # per-dimension KL budget, see 01_signal_vae_pipeline.py "On regularization"
WARMUP_EPOCHS = 8  # beta warm-up length, converted to optimizer steps at run time
MODALITY_DROPOUT_P = 0.3  # spec §5: per-modality probability of hiding an encoder input
# `mse_loss` averages over pixels, while the KL term is a sum over latent dimensions: left
# as is, the KL term would dwarf the reconstruction term. Weighting the mean squared error
# by the pixel count makes the reconstruction term a sum over pixels, like the KL term.
RECONSTRUCTION_WEIGHT = float(IMAGE_SIZE * IMAGE_SIZE)


# --- Stage 1: synthetic data ---------------------------------------------------------------


def generateBlobParameters(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the hidden parameters of `num_samples` blobs.

    Args:
        num_samples: Number of samples.
        rng: A seeded NumPy random generator.

    Returns:
        Array of shape `(num_samples, 3)`, columns `(cx, cy, sigma)`, all in image-relative
        units (`0` is one edge of the image, `1` the opposite edge).
    """
    centres = rng.uniform(0.2, 0.8, size=(num_samples, 2))
    sigmas = rng.uniform(0.06, 0.14, size=(num_samples, 1))
    return np.concatenate([centres, sigmas], axis=1)


def renderBlobImages(params: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Render the clean blob image of every sample.

    Args:
        params: `(N, 3)` array of `(cx, cy, sigma)`, as from `generateBlobParameters`.
        size: Side of the square image, in pixels.

    Returns:
        Array of shape `(N, size, size)`, values in `[0, 1]`. Axis 1 is the vertical (y)
        axis and axis 2 the horizontal (x) axis.
    """
    coords = (np.arange(size) + 0.5) / size
    cx, cy, sigma = (params[:, index, None, None] for index in range(3))
    squared_distance = (coords[None, None, :] - cx) ** 2 + (coords[None, :, None] - cy) ** 2
    return np.exp(-squared_distance / (2.0 * sigma**2)).astype(np.float32)


def renderXProfiles(
    params: np.ndarray, rng: np.random.Generator, length: int = SIGNAL_LENGTH
) -> np.ndarray:
    """Render the 1D signal of every sample: the blob's profile along x.

    The profile is the blob integrated over y, so it depends on `cx` and `sigma` but not
    on `cy`. That blindness to the vertical position is what makes the signal and the
    degraded image complementary.

    Args:
        params: `(N, 3)` array of `(cx, cy, sigma)`.
        rng: A seeded NumPy random generator (measurement noise).
        length: Number of points of the profile.

    Returns:
        Array of shape `(N, length)`.
    """
    positions = (np.arange(length) + 0.5) / length
    cx, sigma = params[:, 0, None], params[:, 2, None]
    profile = np.exp(-((positions[None, :] - cx) ** 2) / (2.0 * sigma**2))
    noise = rng.normal(0.0, SIGNAL_NOISE_STD, size=profile.shape)
    return (profile + noise).astype(np.float32)


def degradeImages(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn clean images into the degraded `image_in` view: occlusion plus noise.

    Args:
        clean: `(N, H, W)` clean images.
        rng: A seeded NumPy random generator.

    Returns:
        Array of the same shape: one randomly placed square erased per image, then
        Gaussian noise added everywhere.
    """
    degraded = clean.copy()
    num_samples, height, width = clean.shape
    tops = rng.integers(0, height - OCCLUSION_SIZE + 1, size=num_samples)
    lefts = rng.integers(0, width - OCCLUSION_SIZE + 1, size=num_samples)
    for index in range(num_samples):
        top, left = tops[index], lefts[index]
        degraded[index, top : top + OCCLUSION_SIZE, left : left + OCCLUSION_SIZE] = 0.0
    degraded += rng.normal(0.0, NOISE_STD, size=degraded.shape).astype(np.float32)
    return degraded


def buildSplit(num_samples: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Generate one data split: every sample's signal, degraded image, and clean image.

    Args:
        num_samples: Number of samples in the split.
        rng: A seeded NumPy random generator.

    Returns:
        `{"signal": (N, SIGNAL_LENGTH), "image_in": (N, H, W), "image_out": (N, H, W)}`.
    """
    params = generateBlobParameters(num_samples, rng)
    clean = renderBlobImages(params)
    return {
        SIGNAL: torch.from_numpy(renderXProfiles(params, rng)),
        IMAGE_IN: torch.from_numpy(degradeImages(clean, rng)),
        IMAGE_OUT: torch.from_numpy(clean),
    }


def toBatches(split: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
    """Cut a split into batches, in order (the `Trainer` batch convention).

    Args:
        split: As returned by `buildSplit`.
        batch_size: Samples per batch.

    Returns:
        A list of `dict[str, torch.Tensor]` batches, all three keys in every batch.
    """
    total = next(iter(split.values())).shape[0]
    return [
        {name: tensor[start : start + batch_size] for name, tensor in split.items()}
        for start in range(0, total, batch_size)
    ]


# --- Stage 2: model ------------------------------------------------------------------------


def buildModel(latent_dim: int = LATENT_DIM) -> GlobalVae:
    """Build the `(signal, image_in) -> image_out` model from an explicit routing graph.

    Args:
        latent_dim: Dimensionality of the single fused latent space.

    Returns:
        A freshly initialized `GlobalVae`: two encoders feeding `z_fused` through PoE
        fusion, one decoder (`image_out`) consuming it.
    """
    routing_graph = buildSingleLatentRoutingGraph(
        encoder_names=[SIGNAL, IMAGE_IN],
        decoder_names=[IMAGE_OUT],
        latent_dim=latent_dim,
        latent_name=LATENT_NAME,
    )
    return GlobalVae(
        encoder_configs={SIGNAL: "1d_cnn_resnet_encoder_v1", IMAGE_IN: "2d_cnn_resnet_encoder_v1"},
        decoder_configs={IMAGE_OUT: "2d_cnn_resnet_decoder_v1"},
        routing_graph=routing_graph,
        # Two encoders feed z_fused, so it needs a fusion strategy (spec §4). PoE is
        # natively subset-tolerant (spec §5), which is what allows querying with either
        # input alone.
        fusion_strategies={LATENT_NAME: "poe"},
        regularizer_strategies={LATENT_NAME: "free_bits_kl"},
        regularizer_kwargs={LATENT_NAME: {"free_bits": FREE_BITS}},
        encoder_kwargs={
            SIGNAL: {
                "latent_dim": latent_dim,
                "hidden_channels": (16, 32, 64),
                "modality_name": SIGNAL,
            },
            IMAGE_IN: {
                "latent_dim": latent_dim,
                "hidden_channels": (16, 32, 64),
                "modality_name": IMAGE_IN,
            },
        },
        decoder_kwargs={
            IMAGE_OUT: {
                "latent_dim": latent_dim,
                # 4 -> 8 -> 16 -> 32 through three doubling transitions (the residual
                # decoder's bare defaults double each axis, see
                # TwoDCnnResidualDecoder.computeOutputShape to check any variation).
                "output_shape": (IMAGE_SIZE, IMAGE_SIZE),
                "hidden_channels": (64, 32, 16),
                "seed_shape": (4, 4),
                "modality_name": IMAGE_OUT,
            }
        },
    )


# --- Stage 3: training ---------------------------------------------------------------------


class TranslationTrainer(Trainer):
    """`Trainer` for a model whose decoder target is not one of its encoder inputs.

    `Trainer.computeLosses` passes the whole batch through `_applyModalityDropout` and
    on to `GlobalVae.forward`, then uses the whole batch again as the reconstruction
    target. That is right when every batch key is both an encoder input and a decoder
    target (plain autoencoding), and fails with `KeyError` when a key such as
    `image_out` is a target only. Restricting what reaches the encoders to the keys that
    have an encoder fixes it and leaves the reconstruction target untouched.

    Also turns modality dropout off during validation: `Trainer.evaluate` goes through
    the same `computeLosses`, so with dropout on, the validation loss the best-checkpoint
    callback monitors would depend on which modalities happened to be hidden that pass.
    """

    def _applyModalityDropout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep only encoder inputs, then apply the usual modality dropout to them.

        Args:
            batch: Every key of the batch, targets included.

        Returns:
            The encoder inputs that survive dropout (at least one).
        """
        encoder_inputs = {
            name: tensor for name, tensor in batch.items() if name in self.model.encoders
        }
        return super()._applyModalityDropout(encoder_inputs)

    def evaluate(self, dataloader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Run the validation pass with every input modality present.

        Args:
            dataloader: Validation batches.

        Returns:
            Metrics averaged over every batch, keys prefixed `"val/"`.
        """
        training_dropout = self.modality_dropout_p
        self.modality_dropout_p = 0.0
        try:
            return super().evaluate(dataloader)
        finally:
            self.modality_dropout_p = training_dropout


# --- Stage 4: evaluation -------------------------------------------------------------------


def subsetLabel(subset: frozenset[str]) -> str:
    """Human-readable label of an input subset, e.g. `"image_in + signal"`.

    Args:
        subset: A non-empty set of modality names.

    Returns:
        The names, sorted and joined with `" + "`.
    """
    return " + ".join(sorted(subset))


def summarizeMetrics(
    metrics: dict[frozenset[str], dict[str, dict[str, float]]],
    baseline_inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Flatten the per-input-subset metrics for `image_out`, plus the copy-the-input baseline.

    Args:
        metrics: As from `computeCrossModalReconstructionMetrics`.
        baseline_inputs: The degraded input images, returned as-is by the baseline.
        targets: The clean target images.

    Returns:
        Label -> `{"mse", "r2", "pearson_r"}`, one row per input subset, plus a
        `"baseline: return image_in"` row.
    """
    rows = {
        subsetLabel(subset): {
            name: per_decoder[IMAGE_OUT][name] for name in ("mse", "r2", "pearson_r")
        }
        for subset, per_decoder in metrics.items()
    }
    rows["baseline: return image_in"] = {
        "mse": computeMse(baseline_inputs, targets),
        "r2": computeR2(baseline_inputs, targets),
        "pearson_r": computePearsonR(baseline_inputs, targets),
    }
    return rows


def formatMetricsTable(rows: dict[str, dict[str, float]]) -> str:
    """Render the metric rows as an aligned plain-text table.

    Args:
        rows: As from `summarizeMetrics`.

    Returns:
        A multi-line string.
    """
    width = max(len(label) for label in rows)
    lines = [f"{'input given to the model':<{width}}   {'mse':>9} {'r2':>8} {'pearson_r':>10}"]
    for label, row in rows.items():
        lines.append(
            f"{label:<{width}}   {row['mse']:>9.5f} {row['r2']:>8.3f} {row['pearson_r']:>10.3f}"
        )
    return "\n".join(lines)


# --- Stage 5: figures ----------------------------------------------------------------------


def plotTranslationGrid(
    test_split: dict[str, torch.Tensor],
    collected: dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]],
    example_indices: list[int],
) -> Figure:
    """Lay out, for a few test samples, every input and what each input subset produces.

    Columns: the 1D signal, the degraded input image, the clean target, then the model's
    `image_out` from `signal` alone, from `image_in` alone, and from both.

    Args:
        test_split: The test split, as from `buildSplit` (aligned with `collected`, since
            the test batches were not shuffled).
        collected: As from `collectCrossModalReconstructions`.
        example_indices: Which test samples to show, one row each.

    Returns:
        The matplotlib `Figure`.
    """
    subsets = [frozenset({SIGNAL}), frozenset({IMAGE_IN}), frozenset({SIGNAL, IMAGE_IN})]
    titles = [
        "signal (input)",
        "image_in (input, degraded)",
        "image_out (target)",
        "output from\nsignal only",
        "output from\nimage_in only",
        "output from\nboth",
    ]
    fig, axes = plt.subplots(
        len(example_indices),
        len(titles),
        figsize=(2.6 * len(titles), 2.4 * len(example_indices)),
        squeeze=False,
    )
    for row, sample in enumerate(example_indices):
        axes[row][0].plot(test_split[SIGNAL][sample].numpy(), linewidth=1.2)
        axes[row][0].set_ylim(-0.2, 1.2)
        images = [
            test_split[IMAGE_IN][sample],
            test_split[IMAGE_OUT][sample],
            *(collected[subset][IMAGE_OUT][1][sample] for subset in subsets),
        ]
        for column, image in enumerate(images, start=1):
            axes[row][column].imshow(image.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
        if row == 0:
            for column, title in enumerate(titles):
                axes[row][column].set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


# --- Entry point ---------------------------------------------------------------------------


def _buildArgumentParser() -> argparse.ArgumentParser:
    """Build this script's `argparse.ArgumentParser`.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-train", type=int, default=2048)
    parser.add_argument("--num-val", type=int, default=256)
    parser.add_argument("--num-test", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the example.

    Args:
        argv: Command-line arguments (excluding the program name). `None` (default) uses
            `sys.argv[1:]`.
    """
    args = _buildArgumentParser().parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setGlobalSeed(args.seed)
    rng = np.random.default_rng(args.seed)

    logger.info("Step 1/5: generating synthetic (signal, degraded image, clean image) triplets.")
    train_split = buildSplit(args.num_train, rng)
    val_split = buildSplit(args.num_val, rng)
    test_split = buildSplit(args.num_test, rng)
    train_batches = toBatches(train_split, BATCH_SIZE)
    val_batches = toBatches(val_split, BATCH_SIZE)
    test_batches = toBatches(test_split, BATCH_SIZE)
    logger.info(
        "  %d/%d/%d train/val/test samples; signal %s, image_in %s, image_out %s.",
        args.num_train,
        args.num_val,
        args.num_test,
        tuple(train_split[SIGNAL].shape[1:]),
        tuple(train_split[IMAGE_IN].shape[1:]),
        tuple(train_split[IMAGE_OUT].shape[1:]),
    )

    logger.info("Step 2/5: assembling the model (two encoders -> PoE -> z_fused -> one decoder).")
    model = buildModel()
    logger.info(
        "  encoders %s, decoders %s, fusions %s, latent spaces %s.",
        list(model.encoders),
        list(model.decoders),
        list(model.fusions),
        list(model.latent_spaces),
    )

    logger.info("Step 3/5: training with modality dropout p=%.2f.", MODALITY_DROPOUT_P)
    warmup_steps = WARMUP_EPOCHS * len(train_batches)
    trainer = TranslationTrainer(
        model,
        device="cpu",
        optimizer_kwargs={"lr": LEARNING_RATE},
        reconstruction_weights=RECONSTRUCTION_WEIGHT,
        beta_schedules={
            LATENT_NAME: LinearWarmupBetaSchedule(
                warmup_steps=warmup_steps, start_value=0.0, end_value=1.0
            )
        },
        modality_dropout_p=MODALITY_DROPOUT_P,
        callbacks=[
            CsvLogger(output_dir / "metrics.csv"),
            # Monitors reconstruction, not "val/loss/total": see 01_signal_vae_pipeline.py.
            BestCheckpointCallback(output_dir / "best.pt", monitor="val/loss/reconstruction"),
        ],
        log_every_n_steps=200,
    )
    history = trainer.fit(train_batches, num_epochs=args.num_epochs, val_dataloader=val_batches)
    logger.info(
        "  training complete: final val reconstruction loss = %.4f.",
        history[-1]["val/loss/reconstruction"],
    )

    logger.info("Step 4/5: evaluating the best checkpoint on the test split, per input subset.")
    best_model = buildModel()
    loadCheckpoint(output_dir / "best.pt", model=best_model)
    best_model.eval()
    # Default subsets (ADR 0016): each encoder alone, and every encoder together.
    collected = collectCrossModalReconstructions(best_model, test_batches, device="cpu")
    rows = summarizeMetrics(
        computeCrossModalReconstructionMetrics(collected),
        baseline_inputs=test_split[IMAGE_IN],
        targets=test_split[IMAGE_OUT],
    )
    logger.info("\n%s", formatMetricsTable(rows))
    (output_dir / "metrics_by_input_subset.json").write_text(json.dumps(rows, indent=2))

    logger.info("Step 5/5: saving figures.")
    grid_fig = plotTranslationGrid(test_split, collected, example_indices=[0, 1, 2, 3, 4])
    grid_fig.savefig(output_dir / "translation_grid.png", dpi=110)
    plt.close(grid_fig)

    loss_fig = plotLossCurves(
        history,
        metrics=["train/loss/reconstruction", "val/loss/reconstruction"],
        twin_metrics=["train/loss/regularization", "val/loss/regularization"],
        log_scale=True,
        title="Training curves (reconstruction left, regularization right)",
        ylabel="reconstruction loss",
        twin_ylabel="regularization loss",
    )
    loss_fig.savefig(output_dir / "loss_curves.png")
    plt.close(loss_fig)

    logger.info("Done. Every output was written to '%s'.", output_dir)


if __name__ == "__main__":
    main()
