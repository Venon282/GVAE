"""Deprecated alias of `global_vae.decoders.two_d_cnn_decoder`, kept until 2.0.

The module was renamed to snake_case like every other module of the package (roadmap D-9,
`docs/adr/0023-snake-case-module-names.md`). Importing it from this path still works, returns
the very same class, and emits a `DeprecationWarning`. Import `TwoDCnnDecoder` from
`global_vae.decoders.two_d_cnn_decoder` instead.
"""

# ruff: noqa: N999  (the CamelCase file name is the old import path this shim exists to keep)

import warnings

from global_vae.decoders.two_d_cnn_decoder import TwoDCnnDecoder

__all__ = ["TwoDCnnDecoder"]

warnings.warn(
    "global_vae.decoders.TwoDCnnDecoder is deprecated and will be removed in 2.0; "
    "import from global_vae.decoders.two_d_cnn_decoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
