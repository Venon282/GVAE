"""Model/dataloader factory functions for `test_evaluate_script.py`'s end-to-end CLI test.

Named without a `test_`/`_test` pattern so pytest's own test discovery never imports
this file directly (only `scripts/evaluate.py`'s dynamic `importlib.import_module`,
and this test module's own explicit import, do; both resolve to the same
`sys.modules` cache entry since they use the identical dotted path). Importing this
file as a collected test module under a *different* name (as pytest's own discovery
would) would re-run its `@registerEncoder`/`@registerDecoder` decorators a second
time and raise "already registered" (see `test_checkpoint.py`'s module docstring for
the same concern with a sibling test file).
"""

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_script_fixtures")
class _DummySignalEncoder(AbstractEncoder):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_mu(x), self.to_logvar(x)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return "signal"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_signal_decoder_script_fixtures")
class _DummySignalDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


def buildModelForScript() -> GlobalVae:
    """Model factory referenced by `--model-factory` in the CLI test."""
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_script_fixtures",
                "decoder": "dummy_signal_decoder_script_fixtures",
            },
        },
        latent_dim=LATENT_DIM,
    )


def buildDataloaderForScript() -> list[dict[str, torch.Tensor]]:
    """Dataloader factory referenced by `--dataloader-factory` in the CLI test."""
    torch.manual_seed(0)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(2)]


def buildLabeledDataloaderForScript() -> list[dict[str, torch.Tensor]]:
    """Like `buildDataloaderForScript`, plus a "label" batch key the model never
    consumes (not one of its modality names): exercises
    `scripts/visualize_latent.py`'s `--label-key` coloring option.
    """
    torch.manual_seed(0)
    return [
        {"signal": torch.randn(BATCH_SIZE, INPUT_DIM), "label": torch.arange(BATCH_SIZE) % 2}
        for _ in range(2)
    ]
