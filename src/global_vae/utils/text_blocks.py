"""Building blocks of the text backbones: token embedding, padding mask and masked mean.

See spec §6 and §10 (the new-modality checklist) and roadmap P3-8, ADR 0041. A text is a sequence
of token ids, and the batch layout of the framework is fixed by one rule of the loss: a decoder's
reconstruction and its target (the batch tensor of the same name) must have the same shape
(`losses.reconstruction.computeTotalReconstructionLoss`). So a text reaches the model as a one-hot
tensor `(batch, length, vocab_size)`, and the decoder returns a tensor of that shape, a
distribution over the vocabulary at every position.

`TokenEmbedding` reads either form of a token sequence with one table of vectors. Integer ids
`(batch, length)` are looked up. A one-hot or soft tensor `(batch, length, vocab_size)` is
multiplied by the table, which for a one-hot row is the same lookup, so both give identical
embeddings and only the memory they need differs.

Padding is a token like any other, with an id the loader chooses (`pad_token_id`). The decoder
predicts it, which is how it learns the length of a text. The encoder is the one that must not read
it: `computePaddingMask` finds the padded positions, the attention layers skip them as keys and
`computeMaskedMean` leaves them out of the mean.

`computeTokenCrossEntropy` is the exact likelihood of a text, for a decoder built with
`output_activation=None`. It is a plain function of `(reconstruction, target)`, so it goes to
`Trainer(reconstruction_loss_fn=...)`, where the loss names of a YAML file (`mse`, `bce`, ...) have
no cross-entropy (ADR 0041).
"""

import torch
import torch.nn.functional as functional
from torch import nn

_PAD_MASS_THRESHOLD = 0.5
_ID_DTYPES = (torch.int32, torch.int64)


class TokenEmbedding(nn.Module):
    """One table of `d_model` vectors, one per token of the vocabulary."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Build the table.

        Args:
            vocab_size: Number of distinct tokens, padding included, at least `2`.
            d_model: Width of a token vector, at least `1`.

        Raises:
            ValueError: If `vocab_size` is below `2` or `d_model` is below `1`.
        """
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2, got {vocab_size}.")
        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}.")
        self.vocab_size = vocab_size
        self.table = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed a batch of token sequences.

        Args:
            tokens: Integer ids `(batch, length)`, or a one-hot (or soft) float tensor
                `(batch, length, vocab_size)`.

        Returns:
            `(batch, length, d_model)`.

        Raises:
            ValueError: If `tokens` is neither of the two layouts above: a float `(batch,
                length)` tensor has no vocabulary axis and is rejected on purpose, since it would
                be read as ids by accident, and a float tensor whose last axis is not
                `vocab_size` was built for another vocabulary.
        """
        if not tokens.is_floating_point():
            if tokens.dtype not in _ID_DTYPES:
                raise ValueError(
                    f"Token ids must be an int32 or int64 tensor, got dtype {tokens.dtype}."
                )
            if tokens.dim() != 2:
                raise ValueError(
                    f"Integer token ids must have shape (batch, length), got {tuple(tokens.shape)}."
                )
            ids: torch.Tensor = self.table(tokens)
            return ids
        if tokens.dim() != 3:
            raise ValueError(
                "A float token tensor must be one-hot with shape (batch, length, vocab_size), "
                f"got shape {tuple(tokens.shape)}. For token ids give an integer tensor "
                "(batch, length)."
            )
        if tokens.shape[-1] != self.vocab_size:
            raise ValueError(
                f"The model was built for vocab_size={self.vocab_size}, but the one-hot tensor "
                f"has {tokens.shape[-1]} entries on its last axis (shape {tuple(tokens.shape)})."
            )
        weight = self.table.weight
        embedded: torch.Tensor = tokens.to(weight.dtype) @ weight
        return embedded


def computePaddingMask(
    tokens: torch.Tensor, pad_token_id: int | None, vocab_size: int
) -> torch.Tensor | None:
    """Mark the padded positions of a batch of token sequences.

    Ids are padding when they equal `pad_token_id`. A one-hot or soft row is padding when its mass
    outside the pad class is below one half, which covers the rows of the pad class and a row of
    zeros, and a decoder's probabilities fed back to an encoder. A row that would be padding
    everywhere (an empty text) keeps its first position, so attention never has every key masked,
    which would produce `NaN`.

    Args:
        tokens: Integer ids `(batch, length)` or a float tensor `(batch, length, vocab_size)`.
        pad_token_id: The id of the padding token, or `None` for a batch with no padding.
        vocab_size: Number of tokens of the vocabulary.

    Returns:
        A boolean `(batch, length)` tensor, `True` at a padded position, or `None` if
        `pad_token_id` is `None`.

    Raises:
        ValueError: If `pad_token_id` is not in `[0, vocab_size)`.
    """
    if pad_token_id is None:
        return None
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"pad_token_id must be in [0, {vocab_size}), got {pad_token_id}.")
    if tokens.is_floating_point():
        real_mass = tokens.sum(dim=-1) - tokens[..., pad_token_id]
        padded = real_mass < _PAD_MASS_THRESHOLD
    else:
        padded = tokens == pad_token_id
    empty = padded.all(dim=1)
    keep_first = torch.zeros_like(padded)
    keep_first[:, 0] = True
    return padded & ~(empty.unsqueeze(1) & keep_first)


def computeMaskedMean(encoded: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Average the tokens of each sequence, leaving the padded ones out.

    Args:
        encoded: `(batch, length, d_model)`.
        padding_mask: Boolean `(batch, length)`, `True` at a padded position, with at least one
            `False` per row (as `computePaddingMask` guarantees), or `None` for no padding.

    Returns:
        `(batch, d_model)`.
    """
    if padding_mask is None:
        mean: torch.Tensor = encoded.mean(dim=1)
        return mean
    weights = (~padding_mask).to(encoded.dtype).unsqueeze(-1)
    pooled: torch.Tensor = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
    return pooled


def computeTokenCrossEntropy(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the negative log-likelihood of a batch of texts under the decoder's scores.

    It is the cross-entropy between the one-hot (or soft) target and the softmax of `scores`,
    summed over the tokens of a text and averaged over the batch, so it is the per-text
    reconstruction term of the ELBO of a categorical decoder. Pass it as the
    `reconstruction_loss_fn` of a `Trainer` with the default `"mean"` reduction (the function
    already sums over the tokens, which the `"sum"` reduction would do a second time), and weight
    `1.0` for the exact ELBO or another weight to trade it against the KL term (`beta`).

    A position whose target row is all zeros adds nothing, so pad with the pad class, which a
    decoder then learns to predict, and not with zeros, if the length of a text is to be learned.

    Args:
        scores: Raw scores `(batch, length, vocab_size)`, from a decoder built with
            `output_activation=None`. Probabilities (the default `"softmax"` output) would be
            softmaxed a second time and give a meaningless loss.
        target: One-hot (or soft) tensor of the same shape.

    Returns:
        A scalar tensor.

    Raises:
        ValueError: If `scores` and `target` differ in shape or are not 3D.
    """
    if scores.dim() != 3 or scores.shape != target.shape:
        raise ValueError(
            "scores and target must both be (batch, length, vocab_size), got "
            f"{tuple(scores.shape)} and {tuple(target.shape)}."
        )
    log_probabilities = functional.log_softmax(scores, dim=-1)
    loss: torch.Tensor = -(target * log_probabilities).sum() / scores.shape[0]
    return loss
