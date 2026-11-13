#!/usr/bin/env python3
r"""Training entry point (spec §10 "Config management", §6.1 milestone 1).

Trains the model a `configs/experiment/*.yaml` file describes end to end: by default the
single-modality signal VAE (spec §6.1 milestone 1), or any model of the config schema
(several latent spaces, separate `encoders` and `decoders`, ADR 0025).

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

    python scripts/train.py \
        data.train_path=/path/to/data \
        data.loader_factory=my_project.data:buildSignalDataloaders \
        training.num_epochs=50 \
        training.optimizer.kwargs.lr=0.0003

Or run an entirely different experiment file:

    python scripts/train.py --config-name experiment/other_experiment

Resume an interrupted run from one of its checkpoints. `training.num_epochs` is the total of
the run, so give the same value as before and only the missing epochs are trained:

    python scripts/train.py \
        training.resume_from=runs/signal_vae/checkpoints/checkpoint_epoch_0039.pt

Where, in your own code (anywhere importable on `PYTHONPATH`):
    def buildSignalDataloaders(config: DataConfig) -> DataloaderBundle:
        # your own dataset/transform/split logic (spec §6); config.train_path,
        # config.batch_size, config.transforms, etc. are yours to interpret however
        # your data actually needs.
        return DataloaderBundle(train=..., val=..., test=...)

See `global_vae/config/data.py` for the exact `DataConfig`/`DataloaderBundle` contract,
and `configs/experiment/signal_vae.yaml` for the full default config this script runs.

A run with `training.stage` set (a free label, `null` by default) writes a stage manifest into
every checkpoint: the label, the modalities, the latent spaces with their widths and a digest of
the model config. It is how the stages of a staged recipe tell one another apart
(`configs/experiment/pair_solo_signal.yaml`, `docs/how-to/train-in-stages.md`, ADR 0051).

A run with `training.warm_start` set (a list of `{checkpoint, modules}` entries, empty by default)
starts from the encoders and decoders of solo checkpoints: the pair stage of a staged recipe copies
the modules of two solo models into one pair model, after checking each file against the stage
manifest, the keys and the shapes, and that the solo encoder has not collapsed. Only the weights
are used: the run keeps its own optimizer, seed and step counter. A resumed run
(`training.resume_from`) does not apply it (`configs/model/signal_image_pair.yaml`,
`docs/how-to/train-in-stages.md`, ADR 0052).

A run stops before the model is built when `data.loader_facts` (the sample rate of an audio
modality, the vocabulary size of a text modality, the categorical block sizes of a table)
contradicts what the model declares; the message starts with the path of the field to change
(`global_vae/config/data_contract.py`, ADR 0049).
"""

import logging

import hydra
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildDataloadersFromConfig
from global_vae.config.data_contract import checkDataContract
from global_vae.config.experiment import ExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    applyWarmStartFromConfig,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
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

    # The facts the loader owns (`data.loader_facts`) against what the model declares: a model
    # written for 16 kHz and a loader at 22.05 kHz stop here, before anything is built (ADR 0049).
    checkDataContract(resolved_config.model, resolved_config.data)

    setGlobalSeed(resolved_config.seed, deterministic=resolved_config.deterministic)

    model = buildModelFromConfig(resolved_config.model)
    dataloaders = buildDataloadersFromConfig(resolved_config.data)

    # A run that lists `training.warm_start` starts from the modules of solo checkpoints (the pair
    # stage of a staged recipe, ADR 0052). It is applied before the trainer is built, so that the
    # optimizer is created over the weights the run starts from, and not when the run resumes.
    applyWarmStartFromConfig(
        model,
        resolved_config.training,
        model_config=resolved_config.model,
        probe_dataloader=dataloaders.train,
    )
    trainer = buildTrainerFromConfig(
        model,
        resolved_config.training,
        config_snapshot=OmegaConf.to_container(cfg),
        model_config=resolved_config.model,
    )

    # `training.num_epochs` is the total of the run: with `training.resume_from` only the
    # epochs after the checkpoint's are left (ADR 0032).
    epochs_to_run = resumeTrainerFromConfig(trainer, resolved_config.training)
    if epochs_to_run == 0:
        logger.info(
            "Nothing to train: the checkpoint holds %d epoch(s) and training.num_epochs is %d.",
            trainer.start_epoch,
            resolved_config.training.num_epochs,
        )
        return

    logger.info("Starting training for %d epoch(s) on device '%s'.", epochs_to_run, trainer.device)
    trainer.fit(dataloaders.train, num_epochs=epochs_to_run, val_dataloader=dataloaders.val)
    logger.info("Training complete. Final metrics: %s", trainer.history[-1])


if __name__ == "__main__":
    main()
