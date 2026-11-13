"""Abstract interface for the extra terms of the training loss (roadmap P5-6, ADR 0047).

`Trainer.computeLosses` is the one place where the loss of a step is computed, for training and
for validation. It adds a reconstruction term and a regularization term, and every objective that
the later phases bring (a latent alignment, a cross reconstruction, a latent reconstruction, a
reference alignment) is one more term added there, chosen by name from `training.loss_terms` and
registered like the regularizers and the beta schedules.

A term is an `nn.Module`, so that one with parameters (a projection head in front of an alignment,
say) is held by the `Trainer`: in its optimizer, in its checkpoint, and restored when a run
resumes. Its weight is not its own business: it is a number or a beta schedule that the trainer
applies (`training.loss_terms[...].weight` and `.schedule`).

Concrete subclasses must self-register via `@registerLossTerm(name)` (see `registry.py`).

Two things that a term may need to know when it is built are given to it once, by the trainer that
holds it, and never kept as the trainer itself: the model (`validateModel`, ADR 0047) and the
settings of the trainer that bear on the term (`bindTrainerSettings`, ADR 0054). A term that
compares a decoder's output with a batch tensor, as a cross reconstruction does, needs the loss,
the reduction and the weight that the trainer uses for that modality; one that needs every
modality of every batch needs to know that modality dropout is off.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from global_vae.losses.reconstruction.aggregate import LossFn
from global_vae.models.global_vae import GlobalVae


@dataclass(frozen=True)
class TrainerSettings:
    """The settings of a `Trainer` that a loss term may read when it is bound (ADR 0054).

    Attributes:
        reconstruction_loss_fn: The reconstruction loss of the trainer: one for every modality, or
            a mapping reconstruction name -> loss. The same object the trainer uses, so a loss
            that is a module is shared with it and not copied. A term keeps it as a plain
            attribute, never as a submodule, or its parameters would be counted twice.
        reconstruction_weights: The trainer's reconstruction weight: one number for every
            modality, or a mapping reconstruction name -> weight.
        reconstruction_reduction: The trainer's reconstruction reduction: one name for every
            modality, or a mapping reconstruction name -> name.
        modality_dropout_p: The trainer's probability of hiding a modality from the encoders in a
            training step.
    """

    reconstruction_loss_fn: LossFn | dict[str, LossFn]
    reconstruction_weights: dict[str, float] | float
    reconstruction_reduction: str | Mapping[str, str]
    modality_dropout_p: float


class AbstractLossTerm(nn.Module, ABC):
    """Base class for every extra term of the training loss.

    A term turns what a training step produced into one scalar. The trainer multiplies it by its
    weight, adds it to the total, logs it as `loss/<name>`, and differentiates the total.
    """

    def validateModel(self, model: GlobalVae) -> None:
        """Check, once, that this term can run on `model`.

        `Trainer` calls it for every term when it is built, so that a term whose constructor
        arguments name latent spaces or modalities fails on a mistyped name at build time and not
        at the first step. The default accepts every model.

        Args:
            model: The model the trainer trains.

        Raises:
            ValueError: By a subclass that cannot run on `model`.
        """
        return None

    def bindTrainerSettings(self, settings: TrainerSettings) -> None:
        """Receive, once, the settings of the trainer that holds this term.

        `Trainer` calls it for every term when it is built, right after `validateModel`, so that a
        term that depends on the trainer's setup fails then and not at the first step, and so that
        it can use the trainer's per-modality reconstruction loss (ADR 0054). The default keeps
        nothing and accepts every setup.

        Args:
            settings: The trainer's reconstruction loss, weights and reduction, and its modality
                dropout.

        Raises:
            ValueError: By a subclass that cannot run with these settings (a term that needs every
                modality in every batch refuses a modality dropout above zero).
        """
        return None

    @abstractmethod
    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        """Compute this term for one batch.

        The term runs outside the `torch.autocast` context of a mixed precision run, so what it
        computes is in `float32`, like the other losses (ADR 0031). A term that has to encode or
        decode again calls `model.encode` and `model.decode` (ADR 0046).

        Args:
            model: The `GlobalVae` the trainer trains (never its compiled wrapper). It is an
                argument and not something to store: keeping it as an attribute would make it a
                submodule of the term and count its parameters twice.
            batch: Modality name -> tensor for the whole batch, on the trainer's device, targets
                included (a key that is not an encoder input is there too).
            outputs: What `GlobalVae.forward` returned for this step, with the keys
                `"reconstructions"`, `"latent_params"` and `"latent_samples"`. It is the pass of
                the training step, after modality dropout if that is on, and its tensors are
                `float32` when `mixed_precision` is set.
            step: The global step, the one the beta schedules are resolved at.

        Returns:
            A scalar tensor, still attached to the autograd graph (a tensor with no dimension:
            a vector is refused by the trainer, since reducing it is a choice about the scale
            that belongs to the term).

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError
