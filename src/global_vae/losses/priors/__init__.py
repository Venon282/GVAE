"""priors subpackage of global_vae.losses: frozen priors over a latent space (ADR 0061).

`flow_prior.py` holds the flow prior of roadmap P6-8 (fit, save, load, and the alignment loss that
pulls a latent toward it). The term that uses it is `losses/terms/flow_prior_alignment.py`.
"""

from global_vae.losses.priors.flow_prior import (
    FlowPrior,
    FlowPriorFit,
    fitFlowPrior,
    loadFlowPrior,
    saveFlowPrior,
)

__all__ = ["FlowPrior", "FlowPriorFit", "fitFlowPrior", "loadFlowPrior", "saveFlowPrior"]
