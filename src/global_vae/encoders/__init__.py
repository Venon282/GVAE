"""encoders subpackage of global_vae.

Importing this package registers every built-in encoder implementation
(`1d_cnn_encoder_v1`) via its `@registerEncoder` decorator. A
`@registerX(...)` decorator only runs once its module is imported;
without this import, `getEncoderClass("1d_cnn_encoder_v1")` would raise
`KeyError` even though `OneDCnnEncoder.py` exists on disk. This mirrors
the pattern already used by `assemblers/__init__.py`,
`losses/regularizers/__init__.py`, `training/beta_schedules/__init__.py`,
and `training/loggers/__init__.py` (spec §10: "each registry-based
subpackage's `__init__.py` must import every concrete implementation
for that side effect").
"""

import global_vae.encoders.OneDCnnEncoder  # noqa: F401
