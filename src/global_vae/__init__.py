"""Global Multimodal VAE — a modular, extensible multimodal Variational
Autoencoder framework.

See the project specification (`global-vae-project-specification.md`) for
the architectural vision: three independent axes (encoder cardinality,
latent cardinality, decoder cardinality), a configurable latent routing
graph, and a registry-based extension mechanism so that new modalities
never require touching the core.
"""

__version__ = "0.1.0"
