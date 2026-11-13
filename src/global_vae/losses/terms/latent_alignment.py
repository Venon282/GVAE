"""The latent alignment term: pull the latent spaces of two modalities together (P6-4, ADR 0054).

Stage 3 of the staged recipe trains two modalities that each have a latent space of their own and
asks the two spaces to agree on the same sample. This term is that request. For every pair
`[first, second]` of latent space names it applies a latent distance (`losses/distances/`, ADR
0053) to the posteriors `(mu, logvar)` that the two spaces gave to the same batch, averages the
per-sample value over the batch, and sums the pairs. Its weight, and a ramp from zero that keeps
the solo solutions intact at the start, are the `weight` and `schedule` of its entry in
`training.loss_terms` (ADR 0047).

The distance acts on the posterior parameters of one sample from each modality, not on two
independent samples: it reads `outputs["latent_params"]` of the step, where row `i` of one space
and row `i` of the other describe the same sample. Nothing is encoded again, so the term costs one
distance per pair.

```yaml
training:
  modality_dropout_p: 0.0
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]
        distance: {name: wasserstein2_diag, kwargs: {squared: true}}
      schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 200, end_value: 1.0}}
```
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import getLatentDistanceClass
from global_vae.losses.terms.base import AbstractLossTerm, TrainerSettings
from global_vae.losses.terms.pairs import (
    Pair,
    checkStopGradient,
    parsePairs,
    refuseModalityDropout,
    requireModalitiesInBatch,
    validatePairsAgainstModel,
)
from global_vae.losses.terms.registry import registerLossTerm
from global_vae.models.global_vae import GlobalVae

TERM_NAME = "latent_alignment"


def _buildDistance(distance: Any) -> AbstractLatentDistance:
    """Build the latent distance a term was configured with.

    Args:
        distance: A registered distance name, or a mapping with the key `name` and optionally
            `kwargs` (the constructor arguments of the distance).

    Returns:
        The distance.

    Raises:
        ValueError: If `distance` is neither a name nor a mapping, if a mapping has no `name` or
            a key other than `name` and `kwargs`.
        KeyError: If the name is not a registered distance (the message lists them).
    """
    if isinstance(distance, str):
        return getLatentDistanceClass(distance)()
    if not isinstance(distance, Mapping):
        raise ValueError(
            f"{TERM_NAME}: `distance` must be the name of a latent distance or a mapping "
            f"{{name: ..., kwargs: {{...}}}}, got {distance!r}. The distance is required: which "
            f"one stage 3 trains with is a choice that the config should state."
        )
    unknown = sorted(set(distance) - {"name", "kwargs"})
    if "name" not in distance or unknown:
        raise ValueError(
            f"{TERM_NAME}: `distance` takes the keys `name` and `kwargs`, got {sorted(distance)}."
        )
    return getLatentDistanceClass(distance["name"])(**dict(distance.get("kwargs") or {}))


@registerLossTerm(TERM_NAME)
class LatentAlignmentTerm(AbstractLossTerm):
    """The latent distance between the spaces of each pair, averaged over the batch and summed.

    The value is on the footing of the KL term (ADR 0029) when the distance sums over the latent
    dimensions, which every built-in does by default, so a weight reads next to `beta`.

    Attributes:
        pairs: The pairs of latent space names.
        distance: The latent distance, a module with no parameters.
        stop_gradient: `"none"`, or `"source"` / `"target"` to cut the gradient into the first /
            second space of every pair, which makes it an anchor that the other space moves
            toward.
    """

    def __init__(
        self,
        pairs: Sequence[Sequence[str]],
        distance: str | Mapping[str, Any],
        stop_gradient: str = "none",
    ) -> None:
        """Configure the term.

        Args:
            pairs: A non-empty list of `[first, second]` latent space names. The two orders of a
                pair are one pair when the distance is symmetric.
            distance: The name of a registered latent distance, or `{name, kwargs}`. Required: D-13
                (which distance stage 3 uses) is open, so no default makes the choice silently.
            stop_gradient: `"none"` (default), `"source"` or `"target"`: the gradient does not
                reach the first, or the second, space of every pair, so the other one is trained
                toward it and it is left as it is.

        Raises:
            ValueError: If `pairs`, `distance` or `stop_gradient` is malformed (the message names
                the entry), or a pair is listed twice.
            KeyError: If `distance` is not a registered latent distance.
        """
        super().__init__()
        self.distance = _buildDistance(distance)
        self.pairs: tuple[Pair, ...] = parsePairs(
            pairs, TERM_NAME, allow_reversed_duplicates=not self.distance.is_symmetric
        )
        self.stop_gradient = checkStopGradient(stop_gradient, TERM_NAME)
        self._needs: tuple[tuple[str, str], ...] = ()

    def validateModel(self, model: GlobalVae) -> None:
        """Check that every pair names two latent spaces of `model` that have the same width.

        It also works out which modalities each batch must carry: those of the encoders that
        feed the two spaces of every pair.

        Args:
            model: The model the trainer trains.

        Raises:
            ValueError: If a name is not a latent space of `model`, or two spaces of a pair have
                different widths (roadmap D-16).
        """
        validatePairsAgainstModel(model, self.pairs, TERM_NAME)
        needs: dict[str, str] = {}
        for first, second in self.pairs:
            for name in (first, second):
                for modality in model.modalitiesEncodedInto(name):
                    needs.setdefault(
                        modality,
                        f"it feeds the latent space '{name}' of the pair [{first}, {second}]",
                    )
        self._needs = tuple(needs.items())

    def bindTrainerSettings(self, settings: TrainerSettings) -> None:
        """Refuse a trainer that hides modalities from the encoders.

        Args:
            settings: The trainer's settings.

        Raises:
            ValueError: If `modality_dropout_p` is above zero.
        """
        refuseModalityDropout(settings, TERM_NAME)

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        """Compute the alignment of every pair on the posteriors of this step's pass.

        Args:
            model: The model the trainer trains (not read).
            batch: Modality name -> tensor for the whole batch.
            outputs: What `GlobalVae.forward` returned for the step.
            step: The global step (not read: the weight, and its ramp, are the trainer's).

        Returns:
            A scalar: the mean over the batch of the distance, summed over the pairs.

        Raises:
            ValueError: If the batch lacks a modality that a pair needs (the stage needs paired
                batches), or by the distance itself (spaces of different widths, a batch too
                small for the distance).
        """
        requireModalitiesInBatch(batch, self._needs, TERM_NAME)
        latent_params: Mapping[str, LatentParams] = outputs["latent_params"]
        values = []
        for first, second in self.pairs:
            posterior_a = self._side(latent_params[first], self.stop_gradient == "source")
            posterior_b = self._side(latent_params[second], self.stop_gradient == "target")
            values.append(self.distance(posterior_a, posterior_b).mean())
        return torch.stack(values).sum()

    @staticmethod
    def _side(posterior: LatentParams, detached: bool) -> LatentParams:
        """Return a posterior as it is, or with the gradient cut."""
        if not detached:
            return posterior
        mu, logvar = posterior
        return mu.detach(), logvar.detach()
