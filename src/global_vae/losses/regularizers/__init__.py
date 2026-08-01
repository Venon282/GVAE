"""regularizers subpackage of global_vae.losses.

Importing this package registers every built-in latent regularizer
strategy (`kl_standard_normal`) via its `@registerRegularizer`
decorator. A `@registerX(...)` decorator only runs once its module is
imported; without this import, `getRegularizerClass("kl_standard_normal")`
would raise `KeyError` even though `kl_standard_normal.py` exists on disk.
"""

import global_vae.losses.regularizers.kl_standard_normal  # noqa: F401
