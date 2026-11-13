"""The cross reconstruction term: decode one modality's latent sample with another's decoder.

(Roadmap P6-4, ADR 0054.) For every ordered pair `[source, target]` of latent space names, the
sample the step drew in the `source` space is passed to `GlobalVae.decode` under the name of the
`target` space (ADR 0046), so the decoders that read `target` decode it, and what they produce is
compared with the batch tensors of the modalities they reconstruct. A signal is rebuilt as an image
and an image as a signal, which is what makes the two spaces say the same thing and not only sit
close to each other.

The comparison is the trainer's own: the loss, the reduction and the weight that the trainer
applies to that modality's self reconstruction (`computeTotalReconstructionLoss`, ADR 0045 and ADR
0048), so a modality is scored the same way whether it was rebuilt from its own space or from
another's. The weight of the whole term, and a ramp, are the `weight` and `schedule` of its entry
in `training.loss_terms` (ADR 0047).

```yaml
training:
  modality_dropout_p: 0.0
  loss_terms:
    - name: cross_reconstruction
      kwargs:
        pairs: [[z_signal, z_image], [z_image, z_signal]]   # both directions, listed
        stop_gradient: none                                  # none, source or target
```
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from global_vae.losses.reconstruction.aggregate import computeTotalReconstructionLoss
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

TERM_NAME = "cross_reconstruction"


@contextmanager
def _decodingStopsGradient(model: GlobalVae, stopped: bool) -> Iterator[None]:
    """Keep the gradient of this decoding out of the parameters of the decoding side.

    With `stopped`, every parameter of the decoders and of the assemblers is switched to not
    requiring a gradient for the duration, and put back as it was. The gradient still flows to the
    input of the decoding, so the encoder that produced it is trained, and the decoders are not
    moved by this term. (Detaching the sample cuts the other side.)

    Args:
        model: The model whose decoding side is kept out.
        stopped: Whether to do it. `False` changes nothing.

    Yields:
        Nothing: the block is the decoding.
    """
    if not stopped:
        yield
        return
    flags = [
        (parameter, parameter.requires_grad)
        for module in (model.decoders, model.assemblers)
        for parameter in module.parameters()
    ]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


@registerLossTerm(TERM_NAME)
class CrossReconstructionTerm(AbstractLossTerm):
    """The reconstruction of each target modality from the latent sample of another modality.

    The value is, for every ordered pair, the trainer's reconstruction loss of what the target's
    decoders produce from the source's sample (summed over the modalities they reconstruct, each
    with its own loss, reduction and weight), and the pairs are summed.

    Attributes:
        pairs: The ordered pairs `(source, target)` of latent space names.
        stop_gradient: `"none"`, or `"source"` to detach the source's sample (only the decoders
            learn to read the other space) or `"target"` to keep the decoders' parameters out of
            this term (only the encoders learn to write what the decoders read).
    """

    def __init__(self, pairs: Sequence[Sequence[str]], stop_gradient: str = "none") -> None:
        """Configure the term.

        Args:
            pairs: A non-empty list of ordered `[source, target]` latent space names. List both
                orders for both directions: a pair listed once is decoded one way.
            stop_gradient: `"none"` (default), `"source"` or `"target"`, as in the class.

        Raises:
            ValueError: If `pairs` or `stop_gradient` is malformed (the message names the entry),
                or a pair is listed twice in the same order.
        """
        super().__init__()
        self.pairs: tuple[Pair, ...] = parsePairs(pairs, TERM_NAME, allow_reversed_duplicates=True)
        self.stop_gradient = checkStopGradient(stop_gradient, TERM_NAME)
        self._needs: tuple[tuple[str, str], ...] = ()
        # A plain attribute: the loss it holds may be a module that the trainer already owns.
        self._settings: TrainerSettings | None = None

    def validateModel(self, model: GlobalVae) -> None:
        """Check the pairs against `model`, and work out which modalities each batch must carry.

        Args:
            model: The model the trainer trains.

        Raises:
            ValueError: If a name is not a latent space of `model`, if two spaces of a pair have
                different widths (roadmap D-16), or if no decoder can decode from the target space
                alone.
        """
        validatePairsAgainstModel(model, self.pairs, TERM_NAME)
        needs: dict[str, str] = {}
        for source, target in self.pairs:
            decoded = model.modalitiesDecodedFrom([target])
            if not decoded:
                raise ValueError(
                    f"{TERM_NAME}: the pair [{source}, {target}] cannot be decoded: no decoder can "
                    f"run from the latent space '{target}' alone (it is read by no decoder, or "
                    f"only through an assembler that needs the other spaces too)."
                )
            pair = f"[{source}, {target}]"
            for modality in model.modalitiesEncodedInto(source):
                needs.setdefault(modality, f"it feeds the source space '{source}' of {pair}")
            for modality in decoded:
                needs.setdefault(modality, f"it is what the target space '{target}' decodes to")
        self._needs = tuple(needs.items())

    def bindTrainerSettings(self, settings: TrainerSettings) -> None:
        """Keep the trainer's per-modality reconstruction setup, and refuse a modality dropout.

        Args:
            settings: The trainer's settings.

        Raises:
            ValueError: If `modality_dropout_p` is above zero.
        """
        refuseModalityDropout(settings, TERM_NAME)
        self._settings = settings

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        """Decode every source sample with its target's decoders and score the result.

        Args:
            model: The model the trainer trains.
            batch: Modality name -> tensor for the whole batch; the targets are in it.
            outputs: What `GlobalVae.forward` returned for the step; the samples are read from it.
            step: The global step (not read: the weight, and its ramp, are the trainer's).

        Returns:
            A scalar: the reconstruction loss of every pair, summed.

        Raises:
            ValueError: If the batch lacks a modality that a pair needs (the stage needs paired
                batches).
            RuntimeError: If the term was never bound to a trainer, so it does not know which
                loss to score a modality with.
        """
        if self._settings is None:
            raise RuntimeError(
                f"{TERM_NAME} is not bound to a trainer: it scores each modality with the "
                f"trainer's loss, which `Trainer` hands over when it is built "
                f"(`bindTrainerSettings`)."
            )
        settings = self._settings
        requireModalitiesInBatch(batch, self._needs, TERM_NAME)
        samples: Mapping[str, torch.Tensor] = outputs["latent_samples"]
        values = []
        for source, target in self.pairs:
            sample = samples[source].detach() if self.stop_gradient == "source" else samples[source]
            with _decodingStopsGradient(model, self.stop_gradient == "target"):
                decoded = model.decode({target: sample})
            values.append(
                computeTotalReconstructionLoss(
                    decoded,
                    dict(batch),
                    weights=settings.reconstruction_weights,
                    loss_fn=settings.reconstruction_loss_fn,
                    reduction=settings.reconstruction_reduction,
                )
            )
        return torch.stack(values).sum()
