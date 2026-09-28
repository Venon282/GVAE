"""fusion subpackage of global_vae.

Importing this package registers every built-in fusion strategy
(`poe`, `moe`, `concat_mlp`, `cross_attention`) via each module's own
`@registerFusion` decorator, mirroring `encoders/__init__.py` (see that
module's docstring for why this import is required).
"""

import global_vae.fusion.concat_mlp  # noqa: F401
import global_vae.fusion.cross_attention  # noqa: F401
import global_vae.fusion.moe  # noqa: F401
import global_vae.fusion.poe  # noqa: F401
