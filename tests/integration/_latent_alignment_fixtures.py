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

The `--latent-set` tests and the multi-way unit tests (roadmap P6-8) use a three-modality model,
`signal`, `image` and `text`, with a latent space each: in the **rotated** state the three
encoders are one function followed by an orthogonal map of the posterior mean, so that the three
spaces are one cloud turned three ways, which no raw comparison reads as aligned and which every
post-hoc orthogonal map recovers.

The **embedded** state (roadmap P6-8, ADR 0060) gives `text` a latent space of width 5 that holds an
embedding of `z_signal`, which has width 3: `mu_text = mu_signal E` with `E` of orthonormal rows. An
adapter whose weight is `E` reads the two as one posterior, which is what the adapter tests use.
"""

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.models.global_vae import GlobalVae
from tests.unit._tiny_pair_model import BATCH, IN_DIM, LATENT, buildTinyModel, tinyBatches

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


TRIPLE = ("signal", "image", "text")
TRIPLE_BATCH = BATCH * 4


def buildTripleModel(widths: Mapping[str, int] | None = None, seed: int = 0) -> GlobalVae:
    """Three modalities, each with a latent space `z_<modality>` (width 3 unless `widths` says)."""
    widths = widths or {name: LATENT for name in TRIPLE}
    torch.manual_seed(seed)
    graph = RoutingGraph(
        latent_specs={f"z_{m}": LatentSpace(f"z_{m}", w) for m, w in widths.items()},
        encoder_to_latents={m: [f"z_{m}"] for m in widths},
        latent_to_decoders={f"z_{m}": [m] for m in widths},
    )
    return GlobalVae(
        encoder_configs={m: "tiny_bn_encoder_p65_test" for m in widths},
        decoder_configs={m: "tiny_bn_decoder_p65_test" for m in widths},
        routing_graph=graph,
        encoder_kwargs={
            m: {"input_dim": IN_DIM, "latent_dim": w, "modality": m} for m, w in widths.items()
        },
        decoder_kwargs={
            m: {"output_dim": IN_DIM, "latent_dim": w, "modality": m} for m, w in widths.items()
        },
    )


def _orthogonal(size: int, seed: int, reflection: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix: torch.Tensor = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float64)
    ).Q
    if reflection:
        matrix[:, 0] = -matrix[:, 0]
    return matrix.float()


def buildRotatedTripleModel() -> GlobalVae:
    """The three encoders are one function, each followed by an orthogonal map of the mean.

    Fed the same input, `signal` gives `mu`, `image` gives `mu R_1` and `text` gives `mu R_2` (the
    second map is a reflection), so the three spaces are one cloud turned.
    """
    model = buildTripleModel()
    base = model.encoders["signal"].state_dict()
    for index, name in enumerate(("image", "text"), start=1):
        encoder = model.encoders[name]
        encoder.load_state_dict(base)
        head = cast(nn.Linear, encoder.get_submodule("to_mu"))
        rotation = _orthogonal(LATENT, index + 10, reflection=index == 2)
        with torch.no_grad():
            head.weight.copy_(rotation.T @ base["to_mu.weight"])
            head.bias.copy_(rotation.T @ base["to_mu.bias"])
    return model


def buildTripleBatches(count: int, seed: int, same: bool = True) -> list[dict[str, torch.Tensor]]:
    """`count` batches of `TRIPLE_BATCH` samples: one tensor in every modality, or strangers."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        x = torch.randn(TRIPLE_BATCH, IN_DIM, generator=generator)
        batches.append(
            {
                name: x.clone() if same else torch.randn(x.shape, generator=generator)
                for name in TRIPLE
            }
        )
    return batches


def buildTripleHeldout() -> list[dict[str, torch.Tensor]]:
    """Held-out samples for the rotated triple model: 96 samples."""
    return buildTripleBatches(3, seed=2)


def buildTripleTrain() -> list[dict[str, torch.Tensor]]:
    """Training samples for the rotated triple model: 128 samples."""
    return buildTripleBatches(4, seed=1)


EMBEDDED_WIDTHS = {"signal": 3, "image": 3, "text": 5}
"""The widths of the embedded model: `z_text` is wider than the others."""


def embeddingMatrix(seed: int = 0) -> torch.Tensor:
    """`E` of shape (3, 5) with orthonormal rows: the map from the wide space to the narrow one."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(5, 3, generator=generator, dtype=torch.float64)).Q
    return cast(torch.Tensor, q.T.float())


def buildEmbeddedModel() -> GlobalVae:
    """`z_text` (width 5) holds `mu_signal E`, and the log-variance likewise (`embeddingMatrix`)."""
    model = buildTripleModel(EMBEDDED_WIDTHS)
    signal, text = model.encoders["signal"], model.encoders["text"]
    embedding = embeddingMatrix()
    state = signal.state_dict()
    for key, value in state.items():
        if key.startswith("to_mu") or key.startswith("to_logvar"):
            continue
        text.state_dict()[key].copy_(value)
    for head in ("to_mu", "to_logvar"):
        target = cast(nn.Linear, text.get_submodule(head))
        with torch.no_grad():
            target.weight.copy_(embedding.T @ state[f"{head}.weight"])
            target.bias.copy_(embedding.T @ state[f"{head}.bias"])
    return model
