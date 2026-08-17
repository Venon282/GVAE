"""Top-level experiment configuration (spec §9, §10 "Config management"): combines
`ModelConfig`, `DataConfig`, and `TrainingConfig` into one composed, validated config,
plus `loadExperimentConfig`, the one function that turns a Hydra config directory and a
config name into a real, typed `ExperimentConfig` instance.

Composition (which YAML files get merged) is Hydra's job, driven by each
`configs/experiment/*.yaml`'s own `defaults` list (spec §9's illustrative examples,
now backed by a real schema); **validation** (does the merged result actually match
`ExperimentConfig`'s shape, are all required fields present) is done explicitly here via
`OmegaConf.merge(OmegaConf.structured(ExperimentConfig), composed)`, rather than relying
on Hydra's `ConfigStore`-group-name-matching to catch every mismatch implicitly. This
two-step "compose, then validate/cast" split is deliberate: it keeps the *set of files
Hydra merges* (a `defaults` list concern) and *whether the result is well-typed* (an
`OmegaConf` concern) as two separately-reasoned-about steps, rather than depending on
exact `ConfigStore` group/name-matching semantics lining up with every concrete YAML
file's name.
"""

from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import MISSING, OmegaConf

from global_vae.config.data import DataConfig
from global_vae.config.model import ModelConfig
from global_vae.config.training import TrainingConfig

# `config/experiment.py` -> `config/` -> `global_vae/` -> `src/` -> repo root -> `configs/`.
# Assumes the standard repo layout (spec §8): `configs/` alongside `src/`. Callers working
# outside that layout (or wanting a different config directory entirely) pass their own
# `config_dir` to `loadExperimentConfig` instead of relying on this default.
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class ExperimentConfig:
    """Everything one training run needs, composed from the three config groups (spec §9).

    Attributes:
        model: See `global_vae.config.model.ModelConfig`. Required:
            there is no sensible default architecture.
        data: See `global_vae.config.data.DataConfig`. Required: there
            is no sensible default dataset.
        training: See `global_vae.config.training.TrainingConfig`.
            Every field has its own sensible default, so omitting this
            entirely is valid and trains with defaults throughout.
        seed: Global RNG seed (spec §10 "global seed management"),
            forwarded to `utils.seed.setGlobalSeed` by
            `scripts/train.py` before any model is constructed.
        deterministic: Forwarded to `utils.seed.setGlobalSeed`'s own
            `deterministic` flag (spec §10's "documented
            deterministic-mode flag").
        output_dir: Where a run's own outputs (checkpoints, logs) are
            expected to live; not itself consumed by this framework
            (checkpoint/logger paths in `TrainingConfig` are set
            independently), but useful as a single value config groups
            can interpolate against (e.g.
            `training.checkpoint.directory: ${output_dir}/checkpoints`)
            and as part of the snapshotted config (spec §10).
    """

    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 0
    deterministic: bool = False
    output_dir: str = "outputs/experiment"


def loadExperimentConfig(
    config_name: str = "experiment/signal_vae",
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Compose, validate, and materialize an `ExperimentConfig`.

    Args:
        config_name: Name passed to Hydra's `compose`, relative to
            `config_dir`, without the `.yaml` extension (e.g.
            `"experiment/signal_vae"` for
            `configs/experiment/signal_vae.yaml`). Every
            `configs/experiment/*.yaml` file must start with
            `# @package _global_` for its `defaults` list (pulling in
            `model`/`data`/`training`) to land at the config root
            instead of being nested under an `experiment:` key; see
            `configs/experiment/signal_vae.yaml` for the concrete
            example.
        config_dir: Absolute path to the `configs/` directory. `None`
            (default) resolves to this repository's own `configs/`
            directory (see this module's own top-of-file comment).
        overrides: Hydra dotlist overrides (e.g.
            `["training.num_epochs=5", "data.batch_size=64"]`), the
            same syntax used on the command line.

    Returns:
        A fully materialized, plain (no more `OmegaConf` wrapper)
        `ExperimentConfig` instance (`OmegaConf.to_object`), so
        downstream code (`buildModelFromConfig`, `buildTrainerFromConfig`,
        `buildDataloadersFromConfig`) works with ordinary dataclass
        attribute access.

    Raises:
        omegaconf.errors.MissingMandatoryValue: If a required field
            (e.g. `model`, `data`, or any field within them marked
            with Hydra's `MISSING`) is not supplied by the composed
            YAML files or `overrides`.
        hydra.errors.MissingConfigException: If `config_name` does not
            resolve to an existing YAML file under `config_dir`.
    """
    resolved_config_dir = (
        str(Path(config_dir).resolve()) if config_dir is not None else str(_DEFAULT_CONFIG_DIR)
    )
    with initialize_config_dir(config_dir=resolved_config_dir, version_base=None):
        composed = compose(config_name=config_name, overrides=overrides or [])

    validated = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), composed)
    config: ExperimentConfig = OmegaConf.to_object(validated)  # type: ignore[assignment]
    return config
