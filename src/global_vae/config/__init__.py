"""config subpackage of global_vae: Hydra-driven, dataclass-validated configuration
(spec §9, §10 "Config management").

Importing this package registers `ModelConfig`, `DataConfig`, `TrainingConfig`, and
`ExperimentConfig` with Hydra's `ConfigStore` under the `model`/`data`/`training`
groups and the config root, mirroring the self-registration side-effect pattern used
everywhere else in this codebase (`assemblers/__init__.py`,
`losses/regularizers/__init__.py`, `training/beta_schedules/__init__.py`,
`training/loggers/__init__.py`): this makes the schemas discoverable to Hydra's own
tooling (`--info`, `--help`, config-group listing) and available for any config file
that wants to inherit from one of these base schemas directly via its own `defaults`
list, in addition to the explicit `OmegaConf.merge`-based validation
`config.experiment.loadExperimentConfig` already performs on every call regardless of
this registration.

This is a genuinely optional side effect for `loadExperimentConfig` itself (that
function validates via `OmegaConf.merge` unconditionally, not by relying on
`ConfigStore` group-name matching), but importing `global_vae.config` (rather than only
`global_vae.config.experiment`) is still the recommended entry point, exactly like
importing `global_vae.assemblers` rather than only the one assembler class you happen
to need right now.
"""

from hydra.core.config_store import ConfigStore

from global_vae.config.data import DataConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig  # noqa: F401
from global_vae.config.model import ModelConfig, buildModelFromConfig  # noqa: F401
from global_vae.config.training import TrainingConfig, buildTrainerFromConfig  # noqa: F401

_config_store = ConfigStore.instance()
_config_store.store(group="model", name="base_model", node=ModelConfig)
_config_store.store(group="data", name="base_data", node=DataConfig)
_config_store.store(group="training", name="base_training", node=TrainingConfig)
_config_store.store(name="base_experiment", node=ExperimentConfig)
