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

Spaces of different widths are read through **adapters** (`adapters`, roadmap P6-8, ADR 0060):
latent heads that this term alone puts between a space and the distance, trained with it, so that
the spaces keep the widths they have (`losses/terms/adapters.py`).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import getLatentDistanceClass
from global_vae.losses.terms.adapters import (
    AdapterSpec,
    applyAdapter,
    buildAdapters,
    parseAdapterSpecs,
)
from global_vae.losses.terms.base import AbstractLossTerm, TrainerSettings
from global_vae.losses.terms.pairs import (
    Pair,
    batchHasModalities,
    checkOnUnpaired,
    checkStopGradient,
    refuseModalityDropout,
    requireModalitiesInBatch,
    requireSameSamples,
    resolvePairs,
    validatePairsAgainstModel,
    zeroLoss,
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
        adapters: Space name -> the latent head this term reads that space through, a module with
            parameters that the trainer trains with the term. Empty until the term is checked
            against the model (`validateModel`), since the width a head reads is the space's.
        on_unpaired: `"raise"` or `"skip"`: what the term does with a batch that lacks a modality
            of a pair (ADR 0061).
        num_applied: How many batches the term has been computed on.
        num_skipped: How many batches it skipped under `on_unpaired: skip`.
    """

    def __init__(
        self,
        pairs: Sequence[Sequence[str]] | None = None,
        *,
        distance: str | Mapping[str, Any],
        stop_gradient: str = "none",
        check_sample_ids: bool = True,
        anchor: str | None = None,
        others: Sequence[str] | None = None,
        adapters: Mapping[str, Any] | None = None,
        on_unpaired: str = "raise",
    ) -> None:
        """Configure the term.

        Args:
            pairs: A non-empty list of `[first, second]` latent space names. The two orders of a
                pair are one pair when the distance is symmetric. Give this, or `anchor` with
                `others`.
            distance: The name of a registered latent distance, or `{name, kwargs}`. Required: D-13
                (which distance stage 3 uses) is open, so no default makes the choice silently.
            stop_gradient: `"none"` (default), `"source"` or `"target"`: the gradient does not
                reach the first, or the second, space of every pair, so the other one is trained
                toward it and it is left as it is.
            check_sample_ids: Compare the `sample_id.<modality>` entries of a batch, when the loader
                gives them, and raise if the modalities are not the same samples (ADR 0058).
                `True` by default; a batch without the entries is trusted either way.
            anchor: The latent space that every space of `others` is aligned to (anchor mode,
                roadmap P6-8, ADR 0059). The pairs are `[other, anchor]` for each of `others`, so
                the distance is the one from each space to the anchor, and `stop_gradient:
                target` keeps the anchor where it is.
            others: The latent spaces aligned to `anchor`.
            adapters: Latent space name -> a width (a `linear` head from the space's width to it)
                or `{name, out_dim, kwargs}` (another registered head): the term compares the
                adapted posterior of that space, so spaces of different widths can be aligned and
                keep their own (roadmap P6-8, ADR 0060). Only this term reads an adapter, and it
                is trained with the term and saved in its checkpoint state. Every space named
                must be in a pair. Adapt one space of a pair toward the other, with
                `stop_gradient` on the other if it should stay: two trainable adapters can make
                the distance small by shrinking both outputs.
            on_unpaired: `"raise"` (default) or `"skip"`. A batch that lacks a modality of a pair
                is refused by name, since a skipped term hides a loader that does not pair its
                samples. A stage that feeds unpaired batches on purpose, next to paired ones
                (roadmap P6-8, ADR 0061), says `"skip"`: the term returns zero on such a batch and
                counts it in `num_skipped`.

        Raises:
            ValueError: If `pairs`, `anchor`, `others`, `adapters`, `distance` or `stop_gradient`
                is malformed (the message names the entry), a pair is listed twice, or an adapter
                names a space that is in no pair.
            KeyError: If `distance` is not a registered latent distance.
        """
        super().__init__()
        self.distance = _buildDistance(distance)
        self.pairs: tuple[Pair, ...] = resolvePairs(
            pairs,
            anchor,
            others,
            TERM_NAME,
            allow_reversed_duplicates=not self.distance.is_symmetric,
        )
        self.anchor = anchor
        self.stop_gradient = checkStopGradient(stop_gradient, TERM_NAME)
        self.check_sample_ids = bool(check_sample_ids)
        self.on_unpaired = checkOnUnpaired(on_unpaired, TERM_NAME)
        self.num_applied = 0
        self.num_skipped = 0
        self._adapter_specs: dict[str, AdapterSpec] = parseAdapterSpecs(adapters, TERM_NAME)
        in_pairs = {name for pair in self.pairs for name in pair}
        stray = [space for space in self._adapter_specs if space not in in_pairs]
        if stray:
            raise ValueError(
                f"{TERM_NAME}: `adapters` names {stray}, which is in no pair of this term "
                f"(pairs: {[list(pair) for pair in self.pairs]}). An adapter reads a space that "
                f"the term compares."
            )
        self.adapters = nn.ModuleDict()
        self._needs: tuple[tuple[str, str], ...] = ()

    def validateModel(self, model: GlobalVae) -> None:
        """Check that every pair names two latent spaces of `model` that are read at one width.

        It builds the adapters, once, since the width a head reads is the width of its space
        (their parameters then exist before the trainer builds its optimizer, on the model's
        device). It also works out which modalities each batch must carry: those of the encoders
        that feed the two spaces of every pair.

        Args:
            model: The model the trainer trains.

        Raises:
            ValueError: If a name is not a latent space of `model`, an adapter cannot be built for
                its space (a head that refuses the widths), or two spaces of a pair are read at
                different widths (roadmap D-16: the widths of the spaces, or of their adapters).
            KeyError: If an adapter names a head that is not registered.
        """
        if self._adapter_specs and len(self.adapters) == 0:
            for space in self._adapter_specs:
                if space not in model.latent_spaces:
                    raise ValueError(
                        f"{TERM_NAME}: the adapter names '{space}', which is not a latent space "
                        f"of this model (latent spaces: {sorted(model.latent_spaces)})."
                    )
            self.adapters = buildAdapters(model, self._adapter_specs, TERM_NAME)
        validatePairsAgainstModel(
            model,
            self.pairs,
            TERM_NAME,
            adapted_widths={space: spec.out_dim for space, spec in self._adapter_specs.items()},
        )
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
                batches; not with `on_unpaired: skip`), if its sample ids say that the
                modalities are not the same samples, or by the distance itself (spaces of
                different widths, a batch too small for the distance).
        """
        if self.on_unpaired == "skip" and not batchHasModalities(batch, self._needs):
            self.num_skipped += 1
            return zeroLoss(model)
        requireModalitiesInBatch(batch, self._needs, TERM_NAME)
        self.num_applied += 1
        if self.check_sample_ids:
            requireSameSamples(batch, self._needs, TERM_NAME)
        if self._adapter_specs and len(self.adapters) == 0:
            raise RuntimeError(
                f"{TERM_NAME} has adapters that are not built: they are built when the term is "
                f"checked against the model (`validateModel`), which `Trainer` does when it is "
                f"built."
            )
        latent_params: Mapping[str, LatentParams] = outputs["latent_params"]
        values = []
        for first, second in self.pairs:
            posterior_a = self._read(first, latent_params[first], self.stop_gradient == "source")
            posterior_b = self._read(second, latent_params[second], self.stop_gradient == "target")
            values.append(self.distance(posterior_a, posterior_b).mean())
        return torch.stack(values).sum()

    def _read(self, space: str, posterior: LatentParams, detached: bool) -> LatentParams:
        """The posterior of a space as the distance reads it: detached if asked, then adapted.

        The gradient is cut **before** the adapter, so a detached space is not moved by this term
        and its adapter, which is the term's own, still learns.
        """
        adapter = self.adapters[space] if space in self.adapters else None
        return applyAdapter(adapter, self._side(posterior, detached))

    @staticmethod
    def _side(posterior: LatentParams, detached: bool) -> LatentParams:
        """Return a posterior as it is, or with the gradient cut."""
        if not detached:
            return posterior
        mu, logvar = posterior
        return mu.detach(), logvar.detach()
