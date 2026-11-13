"""distances subpackage of global_vae.losses.

Importing this package registers every built-in latent distance (`mean_mse`, `wasserstein2_diag`,
`symmetric_kl`, `cosine`, `mixture_jsd`, `info_nce`) via their `@registerLatentDistance`
decorators. A `@registerX(...)` decorator only runs once its module is imported; without these
imports, `getLatentDistanceClass(...)` would raise `KeyError` even though the corresponding file
exists on disk. The contract, the six distances and the scale they share are in
`docs/adr/0053-latent-distance-registry.md`.
"""

import global_vae.losses.distances.cosine  # noqa: F401
import global_vae.losses.distances.info_nce  # noqa: F401
import global_vae.losses.distances.mean_mse  # noqa: F401
import global_vae.losses.distances.mixture_jsd  # noqa: F401
import global_vae.losses.distances.symmetric_kl  # noqa: F401
import global_vae.losses.distances.wasserstein2_diag  # noqa: F401
