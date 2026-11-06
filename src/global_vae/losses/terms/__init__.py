"""terms subpackage of global_vae.losses.

Importing this package registers every built-in loss term through its `@registerLossTerm`
decorator. There is none yet: the terms are the objectives of Phases 6 and 7 of the roadmap (a
latent alignment, a cross reconstruction, a latent reconstruction, a reference alignment), and each
lands with its own ADR. The registry, the base class and the way `Trainer` uses them are in
`docs/adr/0047-extra-loss-terms.md`.
"""
