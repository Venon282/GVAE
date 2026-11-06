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
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from global_vae.models.global_vae import GlobalVae


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
