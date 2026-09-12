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
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion
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


@registerEncoder("dummy_x_encoder_script_fixtures")
class _DummyEncoderX(AbstractEncoder):
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
        return "x"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerEncoder("dummy_y_encoder_script_fixtures")
class _DummyEncoderY(AbstractEncoder):
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
        return "y"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_x_decoder_script_fixtures")
class _DummyDecoderX(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "x"


@registerDecoder("dummy_y_decoder_script_fixtures")
class _DummyDecoderY(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "y"


@registerFusion("dummy_poe_script_fixtures")
class _DummyProductOfExperts(AbstractFusion):
    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_mu_sum: torch.Tensor | None = None
        total_precision: torch.Tensor | None = None
        for mu, logvar in params.values():
            precision = torch.exp(-logvar)
            weighted_mu_sum = (
                precision * mu if weighted_mu_sum is None else weighted_mu_sum + precision * mu
            )
            total_precision = precision if total_precision is None else total_precision + precision
        assert weighted_mu_sum is not None
        assert total_precision is not None
        return weighted_mu_sum / total_precision, -torch.log(total_precision)

    @property
    def handlesMissingModalities(self) -> bool:
        return True


def buildTwoModalityModelForScript() -> GlobalVae:
    """Model factory for the cross-modal-figures CLI test: two modalities ("x", "y")
    fused via a dummy Product-of-Experts, so `scripts/evaluate.py`'s opt-in
    `exportCrossModalFigures` call actually has something to report (spec §5,
    `docs/adr/0016-cross-modal-reconstruction-reporting.md`).
    """
    return GlobalVae.createSingleLatent(
        modality_configs={
            "x": {
                "encoder": "dummy_x_encoder_script_fixtures",
                "decoder": "dummy_x_decoder_script_fixtures",
            },
            "y": {
                "encoder": "dummy_y_encoder_script_fixtures",
                "decoder": "dummy_y_decoder_script_fixtures",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="dummy_poe_script_fixtures",
    )


def buildTwoModalityDataloaderForScript() -> list[dict[str, torch.Tensor]]:
    """Dataloader factory paired with `buildTwoModalityModelForScript`."""
    torch.manual_seed(0)
    return [
        {"x": torch.randn(BATCH_SIZE, INPUT_DIM), "y": torch.randn(BATCH_SIZE, INPUT_DIM)}
        for _ in range(2)
    ]
