"""A tiny two-modality model with `BatchNorm`, for the freeze and learning-rate group tests.

Named without a `test_` prefix so that pytest never collects it, and importable as
`tests.unit._tiny_pair_model`. Importing it registers the two dummy strategies once (a second
module that registered the same names would fail), which is why it is shared by
`test_freeze_modules.py` and `test_param_groups.py` (roadmap P6-5, ADR 0055).

The model has a latent space and an encoder and decoder per modality (`signal`, `image`), each a
linear layer, a `BatchNorm1d` and the output layer, so every module has parameters (weights and
biases) **and** buffers (a running mean, a running variance and a batch counter) that move in
training mode and stand still in evaluation mode. That is what a frozen module must not change,
and it is cheap enough for dozens of tests.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer

MODALITIES = ("signal", "image")
IN_DIM = 8
LATENT = 3
HIDDEN = 10
BATCH = 8


@registerEncoder("tiny_bn_encoder_p65_test")
class TinyEncoder(AbstractEncoder):
    """Linear, `BatchNorm1d`, tanh, then the two heads of a posterior."""

    def __init__(self, input_dim: int, latent_dim: int, modality: str) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._modality = modality
        self.hidden = nn.Linear(input_dim, HIDDEN)
        self.norm = nn.BatchNorm1d(HIDDEN)
        self.to_mu = nn.Linear(HIDDEN, latent_dim)
        self.to_logvar = nn.Linear(HIDDEN, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.norm(self.hidden(x)))
        return self.to_mu(h), self.to_logvar(h)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return self._modality

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("tiny_bn_decoder_p65_test")
class TinyDecoder(AbstractDecoder):
    """Linear, `BatchNorm1d`, tanh, then the output layer."""

    def __init__(self, output_dim: int, latent_dim: int, modality: str) -> None:
        super().__init__()
        self._modality = modality
        self.hidden = nn.Linear(latent_dim, HIDDEN)
        self.norm = nn.BatchNorm1d(HIDDEN)
        self.project = nn.Linear(HIDDEN, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(torch.tanh(self.norm(self.hidden(z))))
        return reconstruction

    @property
    def modality_name(self) -> str:
        return self._modality


def buildTinyModel(seed: int = 0) -> GlobalVae:
    """Two modalities, `signal` and `image`, each with its own latent space `z_<modality>`."""
    torch.manual_seed(seed)
    graph = RoutingGraph(
        latent_specs={f"z_{m}": LatentSpace(f"z_{m}", LATENT) for m in MODALITIES},
        encoder_to_latents={m: [f"z_{m}"] for m in MODALITIES},
        latent_to_decoders={f"z_{m}": [m] for m in MODALITIES},
    )
    return GlobalVae(
        encoder_configs={m: "tiny_bn_encoder_p65_test" for m in MODALITIES},
        decoder_configs={m: "tiny_bn_decoder_p65_test" for m in MODALITIES},
        routing_graph=graph,
        encoder_kwargs={
            m: {"input_dim": IN_DIM, "latent_dim": LATENT, "modality": m} for m in MODALITIES
        },
        decoder_kwargs={
            m: {"output_dim": IN_DIM, "latent_dim": LATENT, "modality": m} for m in MODALITIES
        },
    )


def tinyBatches(count: int = 3, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """`count` batches of `BATCH` samples of both modalities, the same on every call."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {m: torch.randn(BATCH, IN_DIM, generator=generator) for m in MODALITIES}
        for _ in range(count)
    ]


def buildTinyTrainer(model: GlobalVae | None = None, **kwargs: Any) -> Trainer:
    """A trainer on CPU with Adam at `lr=0.01`; `kwargs` go to `Trainer` and win."""
    options: dict[str, Any] = {"device": "cpu", "optimizer_kwargs": {"lr": 0.01}}
    options.update(kwargs)
    return Trainer(model if model is not None else buildTinyModel(), **options)


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """A copy of every parameter and buffer, by name (`state_dict`, which holds both)."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def changedNames(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], prefix: str = ""
) -> list[str]:
    """Names under `prefix` whose tensor is not bit-identical between the two snapshots."""
    return [
        name
        for name in before
        if name.startswith(prefix) and not torch.equal(before[name], after[name])
    ]


def namesUnder(model: nn.Module, prefix: str) -> list[str]:
    """Every parameter and buffer name of `model` that starts with `prefix`."""
    return [name for name in model.state_dict() if name.startswith(prefix)]
