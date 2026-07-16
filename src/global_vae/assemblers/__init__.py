"""assemblers subpackage of global_vae.

Importing this package registers every built-in assembler strategy
(`concat`, `sum`, `average`) via their `@registerAssembler` decorators.
A `@registerX(...)` decorator only runs once its module is imported;
without these imports, `getAssemblerClass("concat")` would raise
`KeyError` even though `concat.py` exists on disk.
"""

import global_vae.assemblers.average  # noqa: F401
import global_vae.assemblers.concat  # noqa: F401
import global_vae.assemblers.sum  # noqa: F401
