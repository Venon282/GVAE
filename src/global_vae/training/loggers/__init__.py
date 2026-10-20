"""loggers subpackage of global_vae.training.

Importing this package registers every built-in experiment logger
(`csv`, `mlflow`, `tensorboard`) via their `@registerLogger` decorators. A
`@registerX(...)` decorator only runs once its module is imported;
without these imports, `getLoggerClass("csv")` would raise `KeyError`
even though `csv_logger.py` exists on disk.

Note: this import never requires the `tensorboard` or `mlflow` PyPI
packages, even though `tensorboard_logger.py` and `mlflow_logger.py` are
imported here. `TensorBoardLogger` only imports
`torch.utils.tensorboard.SummaryWriter` inside its own `__init__`, and
`MlflowLogger` only imports `mlflow` inside its own `__init__`, not at
module load time, precisely so that an environment without them
installed can still use `getLoggerClass("csv")` (see the module
docstrings of `tensorboard_logger.py` and `mlflow_logger.py`).
"""

import global_vae.training.loggers.csv_logger  # noqa: F401
import global_vae.training.loggers.mlflow_logger  # noqa: F401
import global_vae.training.loggers.tensorboard_logger  # noqa: F401
