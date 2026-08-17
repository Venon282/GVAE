#!/usr/bin/env python3
"""Training entry point (spec §10 "Config management", §6.1 milestone 1: a
single-modality signal VAE trained end to end).

Hydra-driven: composes `configs/model/*.yaml` + `configs/data/*.yaml` +
`configs/training/*.yaml` (via `configs/experiment/*.yaml`'s own `defaults` list) into
one `global_vae.config.experiment.ExperimentConfig`, then builds a real `GlobalVae`
(`global_vae.config.model.buildModelFromConfig`), a real `Trainer`
(`global_vae.config.training.buildTrainerFromConfig`), and the actual dataloaders
(`global_vae.config.data.buildDataloadersFromConfig`, which only ever resolves and
calls *your own* `data.loader_factory` callable, spec: data pipeline concerns stay
outside this framework).

Usage:
    python scripts/train.py

Runs `configs/experiment/signal_vae.yaml` by default. Override anything from the
command line with Hydra's dotlist syntax, e.g.:

    python scripts/train.py \\
        data.train_path=/path/to/data \\
        data.loader_factory=my_project.data:buildSignalDataloaders \\
        training.num_epochs=50 \\
        training.optimizer.kwargs.lr=0.0003

Or run an entirely different experiment file:

    python scripts/train.py --config-name experiment/other_experiment

Where, in your own code (anywhere importable on `PYTHONPATH`):
    def buildSignalDataloaders(config: DataConfig) -> DataloaderBundle:
        # your own dataset/transform/split logic (spec §6); config.train_path,
        # config.batch_size, config.transforms, etc. are yours to interpret however
        # your data actually needs.
        return DataloaderBundle(train=..., val=..., test=...)

See `global_vae/config/data.py` for the exact `DataConfig`/`DataloaderBundle` contract,
and `configs/experiment/signal_vae.yaml` for the full default config this script runs.
"""

import logging

import hydra
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildDataloadersFromConfig
from global_vae.config.experiment import ExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig
from global_vae.utils.seed import setGlobalSeed

logger = logging.getLogger("global_vae.scripts.train")


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/signal_vae")
def main(cfg: ExperimentConfig) -> None:
    """Compose, validate, and run one training experiment.

    Args:
        cfg: Hydra's own composed config (a `DictConfig` at runtime,
            despite the `ExperimentConfig` type hint used here for
            readability). Re-validated and materialized into a real
            `ExperimentConfig` instance via the exact same
            `OmegaConf.merge(OmegaConf.structured(...), ...)` step
            `global_vae.config.experiment.loadExperimentConfig` uses,
            so this entry point and any programmatic caller of
            `loadExperimentConfig` enjoy identical guarantees (clear
            errors on missing/mistyped fields, a plain dataclass
            instance downstream code can use directly). Hydra's own
            `@hydra.main` composition (this decorator) and
            `loadExperimentConfig`'s manual `compose()` call are two
            separate entry points into the same schema; a script only
            ever uses one of the two at a time, never both in the same
            process.
    """
    resolved_config: ExperimentConfig = OmegaConf.to_object(  # type: ignore[assignment]
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig), cfg)
    )

    setGlobalSeed(resolved_config.seed, deterministic=resolved_config.deterministic)

    model = buildModelFromConfig(resolved_config.model)
    dataloaders = buildDataloadersFromConfig(resolved_config.data)
    trainer = buildTrainerFromConfig(
        model, resolved_config.training, config_snapshot=OmegaConf.to_container(cfg)
    )

    logger.info(
        "Starting training for %d epoch(s) on device '%s'.",
        resolved_config.training.num_epochs,
        trainer.device,
    )
    trainer.fit(
        dataloaders.train,
        num_epochs=resolved_config.training.num_epochs,
        val_dataloader=dataloaders.val,
    )
    logger.info("Training complete. Final metrics: %s", trainer.history[-1])


if __name__ == "__main__":
    main()
