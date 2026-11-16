"""Model and data factories for the `--latent-pair` tests of `scripts/evaluate.py` (roadmap P6-6).

Named without a `test_` prefix so that pytest never collects it, and importable as
`tests.integration._latent_alignment_fixtures`, the form `module:function` that the script's
`--model-factory`, `--dataloader-factory` and `--train-dataloader-factory` take. The models come
from `tests.unit._tiny_pair_model` (importing it registers its two dummy modules once).

The pair model has two separate latent spaces, `z_signal` and `z_image`, of width 3. Two states of
it are built on purpose:

- **aligned**: the image encoder is a copy of the signal encoder and the image is the signal, so
  that row `i` has the same code in both spaces and the Procrustes map recovers the partner;
- **collapsed**: the heads of both encoders are zero, so that every sample has the same posterior:
  the two spaces are at distance 0 from each other, and from themselves, which proves nothing.
"""

from typing import cast

import torch
from torch import nn

from global_vae.models.global_vae import GlobalVae
from tests.unit._tiny_pair_model import buildTinyModel, tinyBatches

HELDOUT_BATCHES = 3
TRAIN_BATCHES = 6


def buildAlignedPairModel() -> GlobalVae:
    """The tiny pair model whose two encoders are the same function."""
    model = buildTinyModel(0)
    model.encoders["image"].load_state_dict(model.encoders["signal"].state_dict())
    return model


def buildCollapsedPairModel() -> GlobalVae:
    """The aligned pair model with both encoders' heads at zero: every posterior is the same."""
    model = buildAlignedPairModel()
    for modality in ("signal", "image"):
        for name in ("to_mu", "to_logvar"):
            head = cast(nn.Linear, model.encoders[modality].get_submodule(name))
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
    return model


def _linked(count: int, seed: int) -> list[dict[str, torch.Tensor]]:
    """Batches whose image is the signal: the same sample twice."""
    return [{"signal": b["signal"], "image": b["signal"].clone()} for b in tinyBatches(count, seed)]


def buildHeldoutPairs() -> list[dict[str, torch.Tensor]]:
    """Held-out pairs for the aligned model: 24 samples."""
    return _linked(HELDOUT_BATCHES, seed=1)


def buildTrainPairs() -> list[dict[str, torch.Tensor]]:
    """Training pairs for the aligned model: 48 samples, the image of row `i` is its signal."""
    return _linked(TRAIN_BATCHES, seed=2)


def buildShuffledTrainPairs() -> list[dict[str, torch.Tensor]]:
    """Training pairs whose image rows are in another order than the signal rows (a bad join)."""
    batches = _linked(TRAIN_BATCHES, seed=2)
    signal = torch.cat([b["signal"] for b in batches])
    image = signal[torch.randperm(signal.shape[0], generator=torch.Generator().manual_seed(11))]
    size = batches[0]["signal"].shape[0]
    return [
        {"signal": signal[start : start + size], "image": image[start : start + size]}
        for start in range(0, signal.shape[0], size)
    ]


def buildSignalOnlyBatches() -> list[dict[str, torch.Tensor]]:
    """Batches that lack the image, which a pair evaluation cannot use."""
    return [{"signal": batch["signal"]} for batch in tinyBatches(2, seed=1)]
