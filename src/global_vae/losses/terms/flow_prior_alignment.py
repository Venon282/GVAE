"""Pull a latent space toward a frozen flow prior fitted on another space (P6-8, ADR 0061).

With few paired samples and many unpaired ones, the unpaired samples of one modality say nothing
about the other modality's space through a pair loss. A frozen flow prior does. It is fitted once,
on the posterior means that the **other** space gives to its own unpaired samples
(`scripts/fit_flow_prior.py`, or `fitFlowPrior`), and this term then asks the latent space of the
modality at hand to be distributed like them, using the alignment loss of Li et al. ([arXiv
2506.05240](https://arxiv.org/abs/2506.05240)): the flow matching loss, with the flow frozen and
the latent as its data (`losses/priors/flow_prior.py`).

```yaml
training:
  loss_terms:
    - name: flow_prior_alignment
      kwargs:
        space: z_image                 # the space that is pulled
        prior: priors/z_signal.pt      # fitted on the z_signal means of the signal samples
        on_missing: skip               # a stream with batches that do not carry the image
      weight: 0.01
```

The term acts on every batch that carries the modalities of `space`, paired or not, and needs no
partner: that is the point. What it does **not** do is pair the samples. A distribution has
symmetries (a rotation of a round cloud is the same cloud) that a few pairs have to break: the
prior says where the latents may lie, and the pair loss says which latent goes with which.

The prior is a submodule of the term with frozen parameters, so it is in the term's checkpoint
state and a resumed run uses the prior it started with, whatever the file now holds. The optimizer
holds its parameters and never updates them (they have no gradient).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from global_vae.losses.priors.flow_prior import loadFlowPrior
from global_vae.losses.terms.base import AbstractLossTerm
from global_vae.losses.terms.registry import registerLossTerm
from global_vae.models.global_vae import GlobalVae

TERM_NAME = "flow_prior_alignment"
READ_CHOICES: tuple[str, ...] = ("mu", "sample")
ON_MISSING_CHOICES: tuple[str, ...] = ("raise", "skip")


@registerLossTerm(TERM_NAME)
class FlowPriorAlignmentTerm(AbstractLossTerm):
    """The alignment loss of a latent space against a frozen flow prior.

    Attributes:
        space: The latent space that is pulled.
        read: `"mu"` (the posterior mean) or `"sample"` (the reparameterized sample of the step).
        num_draws: How many `(x0, t)` draws the loss averages.
        on_missing: `"raise"` or `"skip"`: what the term does with a batch whose forward pass did
            not reach `space`.
        prior: The frozen `FlowPrior`.
        num_applied: How many batches the term has been computed on.
        num_skipped: How many batches it skipped under `on_missing: skip`.
    """

    def __init__(
        self,
        space: str,
        prior: str | Path,
        *,
        read: str = "mu",
        num_draws: int = 1,
        on_missing: str = "raise",
    ) -> None:
        """Configure the term and read the prior file.

        Args:
            space: The name of the latent space to pull.
            prior: The path of a prior file (`saveFlowPrior`, `scripts/fit_flow_prior.py`).
            read: `"mu"` (default) or `"sample"`: which latent of `space` is pulled. The prior was
                fitted on posterior means, which `"mu"` matches; `"sample"` also moves the
                variance.
            num_draws: The number of draws of the loss per batch, at least 1.
            on_missing: `"raise"` (default) or `"skip"`. A batch that does not reach `space` (an
                unpaired batch of the other modality, in a mixed stream) is refused by name, or
                skipped with a zero under `"skip"`.

        Raises:
            ValueError: If `space` is not a non-empty string, `read` or `on_missing` is not one of
                its choices, `num_draws` is below 1, or the prior file is not a flow prior.
            FileNotFoundError: If the prior file does not exist.
        """
        super().__init__()
        if not isinstance(space, str) or not space:
            raise ValueError(f"{TERM_NAME}: `space` must be a latent space name, got {space!r}.")
        if read not in READ_CHOICES:
            raise ValueError(
                f"{TERM_NAME}: read must be one of {list(READ_CHOICES)}, got {read!r}."
            )
        if on_missing not in ON_MISSING_CHOICES:
            raise ValueError(
                f"{TERM_NAME}: on_missing must be one of {list(ON_MISSING_CHOICES)}, "
                f"got {on_missing!r}."
            )
        if isinstance(num_draws, bool) or not isinstance(num_draws, int) or num_draws < 1:
            raise ValueError(
                f"{TERM_NAME}: num_draws must be an integer of 1 or more, got {num_draws!r}."
            )
        self.space = space
        self.read = read
        self.num_draws = num_draws
        self.on_missing = on_missing
        self.prior_path = str(prior)
        self.prior = loadFlowPrior(prior)
        self.num_applied = 0
        self.num_skipped = 0

    def validateModel(self, model: GlobalVae) -> None:
        """Check that `space` is a latent space of `model` and has the prior's width.

        Args:
            model: The model the trainer trains.

        Raises:
            ValueError: If the space is not one of the model's, or its width is not the prior's
                (the message says how to fit a prior of that width).
        """
        if self.space not in model.latent_spaces:
            raise ValueError(
                f"{TERM_NAME}: '{self.space}' is not a latent space of this model (latent spaces: "
                f"{sorted(model.latent_spaces)})."
            )
        width = model.latent_spaces[self.space].dim
        if width != self.prior.dim:
            raise ValueError(
                f"{TERM_NAME}: the latent space '{self.space}' has width {width} and the prior "
                f"'{self.prior_path}' is over width {self.prior.dim}. Fit the prior for this width "
                f"(`--dim {width}` in scripts/fit_flow_prior.py, or `dim={width}` in "
                f"fitFlowPrior): when the target space has another width, a fixed random "
                f"projection maps its points to it."
            )

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        """The alignment loss of the latents of `space` against the prior.

        Args:
            model: The model the trainer trains.
            batch: Modality name -> tensor (not read: the space is looked up in `outputs`).
            outputs: What `GlobalVae.forward` returned for the step.
            step: The global step (not read: the weight and its ramp are the trainer's).

        Returns:
            A scalar attached to the graph of the latents, or a zero when the batch did not reach
            the space and `on_missing` is `"skip"`.

        Raises:
            ValueError: If the pass did not reach `space` and `on_missing` is `"raise"`.
        """
        latent_params: Mapping[str, Any] = outputs["latent_params"]
        if self.space not in latent_params:
            if self.on_missing == "skip":
                self.num_skipped += 1
                return torch.zeros((), device=next(model.parameters()).device)
            raise ValueError(
                f"{TERM_NAME}: this batch did not reach the latent space '{self.space}' (it has "
                f"{sorted(batch)}, and the pass reached {sorted(latent_params)}). A stream that "
                f"mixes batches of several modalities says `on_missing: skip` (ADR 0061)."
            )
        latents = (
            latent_params[self.space][0]
            if self.read == "mu"
            else outputs["latent_samples"][self.space]
        )
        self.num_applied += 1
        return self.prior.alignmentLoss(latents.float(), self.num_draws)
