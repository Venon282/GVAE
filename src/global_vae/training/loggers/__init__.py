"""loggers subpackage of global_vae.training.

Importing this package registers every built-in experiment logger
(`csv`, `tensorboard`) via their `@registerLogger` decorators. A
`@registerX(...)` decorator only runs once its module is imported;
without these imports, `getLoggerClass("csv")` would raise `KeyError`
even though `csv_logger.py` exists on disk.

Note: this import never requires the `tensorboard` PyPI package, even
though `tensorboard_logger.py` is imported here. `TensorBoardLogger`
only imports `torch.utils.tensorboard.SummaryWriter` inside its own
`__init__`, not at module load time, precisely so that an environment
without `tensorboard` installed can still use `getLoggerClass("csv")`
(see `tensorboard_logger.py`'s module docstring).
"""

import global_vae.training.loggers.csv_logger  # noqa: F401
import global_vae.training.loggers.tensorboard_logger  # noqa: F401
