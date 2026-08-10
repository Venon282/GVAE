"""regularizers subpackage of global_vae.losses.

Importing this package registers every built-in latent regularizer
strategy (`kl_standard_normal`, `free_bits_kl`, `mmd`) via their
`@registerRegularizer` decorators. A `@registerX(...)` decorator only
runs once its module is imported; without these imports,
`getRegularizerClass(...)` would raise `KeyError` even though the
corresponding file exists on disk.
"""

import global_vae.losses.regularizers.free_bits_kl  # noqa: F401
import global_vae.losses.regularizers.kl_standard_normal  # noqa: F401
import global_vae.losses.regularizers.mmd  # noqa: F401
