"""Cross-entropy over the tokens of a text (roadmap P5-3, ADR 0048)."""

from typing import ClassVar

import torch
import torch.nn.functional as functional

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss
from global_vae.utils.text_blocks import computeTokenCrossEntropy

_ID_DTYPES = (torch.int32, torch.int64)


@registerReconstructionLoss("token_cross_entropy")
class TokenCrossEntropyLoss(AbstractReconstructionLoss):
    """The negative log-likelihood of a text under the decoder's scores, per token.

    It wraps `computeTokenCrossEntropy` and divides it by the number of tokens of a text, so it is
    the mean cross-entropy of a token and `reduction: sum` gives the cross-entropy of the text, the
    reconstruction term of the ELBO of a categorical decoder.

    The reconstruction is the raw scores `(batch, length, vocabulary)` of a decoder built with
    `output_activation=None` (probabilities would be softmaxed twice). The target is the **token
    ids** `(batch, length)`, or a one-hot or soft tensor of the shape of the scores. Padding is a
    token like any other, which the decoder learns to predict (`utils/text_blocks.py`).
    """

    requires_matching_shapes: ClassVar[bool] = False
    default_metrics: ClassVar[tuple[str, ...] | None] = ("token_accuracy",)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: Raw scores, `(batch, length, vocabulary)`.
            target: Integer ids `(batch, length)`, or a float one-hot or soft tensor of the shape
                of `reconstruction`.

        Returns:
            A scalar tensor: the mean cross-entropy of a token, averaged over the batch.

        Raises:
            ValueError: If `reconstruction` is not 3D, if a float target does not have its shape,
                if ids are not `(batch, length)` integers of dtype `int32` or `int64`, or if an id
                is outside the vocabulary.
        """
        if reconstruction.dim() != 3:
            raise ValueError(
                "token_cross_entropy needs scores (batch, length, vocabulary), got "
                f"{tuple(reconstruction.shape)}."
            )
        vocabulary = reconstruction.shape[2]
        if target.is_floating_point():
            if target.shape != reconstruction.shape:
                raise ValueError(
                    "A one-hot or soft target must have the shape of the scores "
                    f"{tuple(reconstruction.shape)}, got {tuple(target.shape)}. Pass token ids "
                    "(batch, length) instead if the batch holds ids."
                )
            one_hot = target
        else:
            if target.dtype not in _ID_DTYPES or target.shape != reconstruction.shape[:2]:
                raise ValueError(
                    f"Token ids must be an int32 or int64 tensor of shape "
                    f"{tuple(reconstruction.shape[:2])}, got dtype {target.dtype} and shape "
                    f"{tuple(target.shape)}."
                )
            if target.numel() > 0 and (target.min() < 0 or target.max() >= vocabulary):
                raise ValueError(
                    f"Token ids must be in [0, {vocabulary}) (the last axis of the scores), "
                    f"got a range of [{int(target.min())}, {int(target.max())}]."
                )
            one_hot = functional.one_hot(target.long(), vocabulary).to(reconstruction.dtype)
        loss: torch.Tensor = (
            computeTokenCrossEntropy(reconstruction, one_hot) / (reconstruction.shape[1])
        )
        return loss

    def countTerms(self, reconstruction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of tokens of a text (`length`).

        Args:
            reconstruction: The scores, `(batch, length, vocabulary)`.
            target: The ids or the one-hot tensor.

        Returns:
            `length`.
        """
        return int(reconstruction.shape[1])
