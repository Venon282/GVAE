"""Deprecated alias of `global_vae.encoders.two_d_cnn_residual_encoder`, kept until 2.0.

The module was renamed to snake_case like every other module of the package (roadmap D-9,
`docs/adr/0023-snake-case-module-names.md`). Importing it from this path still works, returns
the very same class, and emits a `DeprecationWarning`. Import `TwoDCnnResidualEncoder` from
`global_vae.encoders.two_d_cnn_residual_encoder` instead.
"""

# ruff: noqa: N999  (the CamelCase file name is the old import path this shim exists to keep)

import warnings

from global_vae.encoders.two_d_cnn_residual_encoder import TwoDCnnResidualEncoder

__all__ = ["TwoDCnnResidualEncoder"]

warnings.warn(
    "global_vae.encoders.TwoDCnnResidualEncoder is deprecated and will be removed in 2.0; "
    "import from global_vae.encoders.two_d_cnn_residual_encoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
