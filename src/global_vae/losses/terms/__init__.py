"""terms subpackage of global_vae.losses.

Importing this package registers every built-in loss term through its `@registerLossTerm`
decorator: the two terms of the pair stage (roadmap P6-4, `docs/adr/0054-pair-stage-loss-terms.md`),
`latent_alignment` and `cross_reconstruction`, and `flow_prior_alignment`, which pulls one space
toward a frozen flow prior fitted on another (roadmap P6-8,
`docs/adr/0061-few-pairs-many-unpaired.md`). The terms of Phase 7 (a latent reconstruction, a
reference alignment) land with their own ADRs. The registry, the base class and the way `Trainer`
uses them are in `docs/adr/0047-extra-loss-terms.md`; `pairs.py` holds the checks the two pair
terms share and is not a term.
"""

import global_vae.losses.terms.cross_reconstruction  # noqa: F401
import global_vae.losses.terms.flow_prior_alignment  # noqa: F401
import global_vae.losses.terms.latent_alignment  # noqa: F401
