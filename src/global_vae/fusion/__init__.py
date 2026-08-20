"""fusion subpackage of global_vae.

Importing this package registers every built-in fusion strategy
(`poe`) via its `@registerFusion` decorator, mirroring
`encoders/__init__.py` (see that module's docstring for why this
import is required).
"""

import global_vae.fusion.poe  # noqa: F401
