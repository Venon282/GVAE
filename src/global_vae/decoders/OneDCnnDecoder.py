"""Deprecated alias of `global_vae.decoders.one_d_cnn_decoder`, kept until 2.0.

The module was renamed to snake_case like every other module of the package (roadmap D-9,
`docs/adr/0023-snake-case-module-names.md`). Importing it from this path still works, returns
the very same class, and emits a `DeprecationWarning`. Import `OneDCnnDecoder` from
`global_vae.decoders.one_d_cnn_decoder` instead.
"""

# ruff: noqa: N999  (the CamelCase file name is the old import path this shim exists to keep)

import warnings

from global_vae.decoders.one_d_cnn_decoder import OneDCnnDecoder

__all__ = ["OneDCnnDecoder"]

warnings.warn(
    "global_vae.decoders.OneDCnnDecoder is deprecated and will be removed in 2.0; "
    "import from global_vae.decoders.one_d_cnn_decoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
