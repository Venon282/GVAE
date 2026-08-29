"""transforms subpackage of global_vae.data (spec §6.2).

A small library of generic, invertible data transforms
(`AbstractTransform` + registry, mirroring every other pluggable strategy
in this codebase), plus `ComposeTransform` for chaining several into one
pipeline. Every concrete transform here works on a tensor of *any* shape
or dimensionality; none is written for, or aware of, one specific dataset
or modality (see `base.py`'s module docstring for the exact rule).

Importing this package registers every built-in transform (`log`,
`standardize`, `resample`) via their `@registerTransform` decorators. A
`@registerX(...)` decorator only runs once its module is imported; without
these imports, `getTransformClass("log")` would raise `KeyError` even
though `log.py` exists on disk (spec §10, the same convention every other
registry-based subpackage in this codebase follows).
"""

import global_vae.data.transforms.log  # noqa: F401
import global_vae.data.transforms.resample  # noqa: F401
import global_vae.data.transforms.standardize  # noqa: F401
