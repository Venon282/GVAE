#!/usr/bin/env python3
"""The same spec §6.1 milestone 1 pipeline as `01_signal_vae_pipeline.py`, but
assembled entirely from the `configs/` YAML files (spec §9, §10 "Config management")
instead of hand-written Python kwargs: the same files `scripts/train.py` composes by
default (`configs/experiment/signal_vae.yaml`), with a synthetic, in-memory
`loader_factory` (`_synthetic_signal_data.buildSyntheticSignalDataloaders`) standing in
for a real dataset, so this still runs with no external data.

To be explicit about what "config-driven" means here, since it is easy to miss just by
skimming the code below: `loadExperimentConfig()` is called with no `config_name`
argument, so it defaults to `"experiment/signal_vae"` and genuinely reads and merges
`configs/experiment/signal_vae.yaml` (plus, via that file's own `defaults:` list,
`configs/model/signal_single_latent.yaml`, `configs/data/signal.yaml`, and
`configs/training/default.yaml`) from disk through Hydra, exactly like
`scripts/train.py` does. `EXPERIMENT_VARIANTS`'s dotlist strings below are overrides
layered on top of those YAML files, not a second, Python-side definition of the
config; `runVariant` also logs the exact override list and a few of the values it
resolves to, so this is visible at run time too, not only in this docstring.

Besides "the same pipeline, config-driven", this script demonstrates something
`01_signal_vae_pipeline.py` cannot show at all: **versioned, comparable experiment
runs.** Two named variants of the shipped config are composed and trained back to
back, each through nothing but a small list of Hydra dotlist overrides on top of the
same `configs/experiment/signal_vae.yaml`:

- `"baseline"`: the config exactly as shipped, no overrides beyond wiring in the
  synthetic loader factory. This is the plain `kl_standard_normal` regularizer
  (`configs/model/signal_single_latent.yaml`) with a beta warm-up tuned for a generic
  first attempt (`configs/training/default.yaml`).
- `"tuned"`: the same config, with a handful of overrides switching to `free_bits_kl`
  and a slower beta warm-up: the exact fix `01_signal_vae_pipeline.py`'s own module
  docstring documents for the posterior-collapse failure mode plain KL-to-standard-
  normal is prone to.

Each variant gets its own `output_dir` (so its checkpoint, config snapshot, CSV/
TensorBoard logs, and figures never overwrite the other's: a plain, filesystem-level
form of run versioning any config-driven workflow gets close to for free) and its own
evaluation report. The two are compared side by side at the end, so the value of
driving hyperparameters from config (easy to vary, easy to keep every variant's exact
settings on record) is visible directly in the numbers, not just asserted.

Run:
    pip install -e ".[dev]"
    python examples/02_config_driven_pipeline.py

Everything this script writes goes under
`examples/outputs/02_config_driven_pipeline/<variant>/` (created if missing; already
covered by `.gitignore`'s `outputs/` pattern).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _synthetic_signal_data import _buildSyntheticSignalArtifacts

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildDataloadersFromConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig
from global_vae.evaluation.evaluate import EvaluationResults, evaluate
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.utils.seed import setGlobalSeed
from global_vae.visualization.latent_plot import collectLatentParams, plotLatentSpace
from global_vae.visualization.loss_curves import plotLossCurves
from global_vae.visualization.reconstruction_plot import (
    collectReconstructions,
    plotReconstructionGrid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("global_vae.examples.config_driven_pipeline")

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "02_config_driven_pipeline"
LOADER_FACTORY = "_synthetic_signal_data:buildSyntheticSignalDataloaders"
NUM_EPOCHS = 100

# Every variant starts from the exact same shipped configs/experiment/signal_vae.yaml
# (spec §9); each list below is *only* the overrides that make this run this variant's
# own version -- the "versioning" this script exists to demonstrate is entirely
# expressed as small, readable diffs on top of one shared baseline, not as separate,
# hand-duplicated config files.
EXPERIMENT_VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "tuned": [
        "model.single_latent.regularizer.strategy=free_bits_kl",
        "+model.single_latent.regularizer.kwargs.free_bits=1.0",
        "training.beta_schedules.z_fused.kwargs.warmup_steps=1500",
        "training.checkpoint.best_monitor=val/loss/reconstruction",
        "training.optimizer.kwargs.lr=0.002",
    ],
}


def _saveVariantFigures(
    variant_dir: Path,
    best_model: GlobalVae,
    cfg: ExperimentConfig,
    history: list[dict[str, float]],
) -> None:
    """Save this variant's own reconstruction/latent/loss figures, mirroring
    `01_signal_vae_pipeline.py`'s own step 7, so both examples produce directly
    comparable output artifacts.

    Args:
        variant_dir: This variant's own output directory.
        best_model: The best checkpoint's model, already loaded.
        cfg: This variant's composed `ExperimentConfig`; rebuilds the same
            preprocessing pipeline and common grid (deterministic in `cfg.seed`) for
            the reconstruction plot's `inverse_transform`/`x_values`.
        history: `Trainer.history` for this variant's run.
    """
    bundle, pipeline, common_grid = _buildSyntheticSignalArtifacts(cfg.data)
    assert bundle.test is not None  # _buildSyntheticSignalArtifacts always populates it
    test_batches = list(bundle.test)

    mu, logvar = collectLatentParams(best_model, test_batches, "z_fused", device="cpu")
    latent_fig = plotLatentSpace(mu, title=f"Latent space ({variant_dir.name})")
    latent_fig.savefig(variant_dir / "latent_space.png")
    plt.close(latent_fig)

    originals, reconstructions = collectReconstructions(
        best_model, test_batches, "signal", device="cpu"
    )
    recon_fig = plotReconstructionGrid(
        originals,
        reconstructions,
        inverse_transform=pipeline.inverse,
        x_values=common_grid,
        max_examples=6,
        ncols=3,
        title=f"Test-set reconstructions ({variant_dir.name})",
        xlabel="position (common grid)",
        ylabel="intensity",
    )
    recon_fig.savefig(variant_dir / "reconstructions.png")
    plt.close(recon_fig)

    # Same reasoning as 01_signal_vae_pipeline.py: reconstruction and regularization
    # live on very different scales, especially for the "tuned" variant where
    # free_bits_kl holds regularization near a roughly constant floor; a twin axis
    # keeps both readable regardless of which variant this is.
    loss_fig = plotLossCurves(
        history,
        metrics=["train/loss/reconstruction", "val/loss/reconstruction"],
        twin_metrics=["train/loss/regularization", "val/loss/regularization"],
        log_scale=True,
        twin_log_scale=True,
        title=f"Training curves ({variant_dir.name})",
        ylabel="reconstruction loss",
        twin_ylabel="regularization loss",
    )
    loss_fig.savefig(variant_dir / "loss_curves.png")
    plt.close(loss_fig)


def runVariant(name: str, extra_overrides: list[str]) -> EvaluationResults:
    """Compose, train, checkpoint, and evaluate one named experiment variant.

    Args:
        name: Variant name, used as its own `output_dir` subfolder.
        extra_overrides: Hydra dotlist overrides applied on top of the shared base
            overrides (the synthetic loader factory, epoch count, and this variant's
            own `output_dir`).

    Returns:
        This variant's `EvaluationResults` on its own held-out test split.
    """
    variant_dir = OUTPUT_ROOT / name
    logger.info("--- Variant '%s' ---", name)
    logger.info(
        "  composing configs/experiment/signal_vae.yaml (itself pulling in "
        "configs/model/signal_single_latent.yaml, configs/data/signal.yaml, and "
        "configs/training/default.yaml, per that file's own `defaults:` list) with "
        "%d Hydra override(s) on top: %s",
        len(extra_overrides),
        extra_overrides if extra_overrides else "(none: this is the baseline, exactly as shipped)",
    )

    cfg = loadExperimentConfig(
        overrides=[
            f"data.loader_factory={LOADER_FACTORY}",
            "data.train_path=synthetic",  # unused by this loader_factory; required by DataConfig
            f"output_dir={variant_dir}",
            f"training.num_epochs={NUM_EPOCHS}",
            *extra_overrides,
        ]
    )
    regularizer_strategy = (
        cfg.model.single_latent.regularizer.strategy
        if cfg.model.single_latent is not None
        else None
    )
    logger.info(
        "  resolved from that composed config: regularizer=%s, best-checkpoint monitor=%s, lr=%s.",
        regularizer_strategy,
        cfg.training.checkpoint.best_monitor,
        cfg.training.optimizer.kwargs.get("lr"),
    )
    setGlobalSeed(cfg.seed, deterministic=cfg.deterministic)

    model = buildModelFromConfig(cfg.model)
    dataloaders = buildDataloadersFromConfig(cfg.data)
    trainer = buildTrainerFromConfig(model, cfg.training, config_snapshot=cfg)

    history = trainer.fit(
        dataloaders.train, num_epochs=cfg.training.num_epochs, val_dataloader=dataloaders.val
    )
    logger.info(
        "  final train/val reconstruction loss = %.4f / %.4f (regularization %.4f / %.4f).",
        history[-1]["train/loss/reconstruction"],
        history[-1]["val/loss/reconstruction"],
        history[-1]["train/loss/regularization"],
        history[-1]["val/loss/regularization"],
    )

    best_model = buildModelFromConfig(cfg.model)
    assert cfg.training.checkpoint.best_path is not None  # every variant sets one
    loadCheckpoint(Path(cfg.training.checkpoint.best_path), model=best_model)

    assert dataloaders.test is not None  # this example's loader_factory always populates it
    results = evaluate(best_model, dataloaders.test, device=trainer.device)
    print(f"\n[{name}]\n{results.summary()}\n")
    results.save(variant_dir / "evaluation.json")

    _saveVariantFigures(variant_dir, best_model, cfg, history)
    return results


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = {
        name: runVariant(name, overrides) for name, overrides in EXPERIMENT_VARIANTS.items()
    }

    print("=" * 72)
    print("Variant comparison (test split, signal reconstruction metrics)")
    print("=" * 72)
    header = f"{'variant':<12}{'mse':>10}{'r2':>10}{'pearson_r':>12}{'reg (kl_std_normal)':>22}"
    print(header)
    for name, results in all_results.items():
        recon = results.reconstruction_metrics["signal"]
        reg = results.regularization_metrics["z_fused"]["kl_standard_normal"]
        print(
            f"{name:<12}{recon['mse']:>10.4f}{recon['r2']:>10.4f}{recon['pearson_r']:>12.4f}"
            f"{reg:>22.4f}"
        )
    print(
        "\nEach variant's full config was snapshotted alongside its checkpoint "
        f"(see '{OUTPUT_ROOT}/<variant>/best.pt', loadable via "
        "training.checkpoint.loadCheckpoint's returned config); every figure and "
        "the JSON evaluation report live under that same variant's own output_dir, "
        "never overwriting another variant's."
    )


if __name__ == "__main__":
    main()
