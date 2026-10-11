"""heads subpackage of global_vae.

Importing this package registers every built-in latent head (`identity`, `linear`) via
each module's own `@registerHead` decorator, mirroring `fusion/__init__.py` and
`assemblers/__init__.py`. A `@registerX(...)` decorator only runs once its module is
imported; without these imports, `getHeadClass("identity")` would raise `KeyError` even
though `identity.py` exists on disk.
"""

import global_vae.heads.identity  # noqa: F401
import global_vae.heads.linear  # noqa: F401
