"""encoders subpackage of global_vae.

Importing this package registers every built-in encoder implementation
(`1d_cnn_encoder_v1`, `1d_cnn_resnet_encoder_v1`, `2d_cnn_encoder_v1`)
via each module's own `@registerEncoder` decorator. A `@registerX(...)`
decorator only runs once its module is imported; without these
imports, `getEncoderClass(...)` would raise `KeyError` even though the
corresponding file exists on disk. This mirrors the pattern already
used by `assemblers/__init__.py`, `losses/regularizers/__init__.py`,
`training/beta_schedules/__init__.py`, and `training/loggers/__init__.py`
(spec §10: "each registry-based subpackage's `__init__.py` must import
every concrete implementation for that side effect").
"""

import global_vae.encoders.OneDCnnEncoder  # noqa: F401
import global_vae.encoders.OneDCnnResidualEncoder  # noqa: F401
import global_vae.encoders.TwoDCnnEncoder  # noqa: F401
