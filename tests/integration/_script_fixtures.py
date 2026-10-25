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

from collections.abc import Callable, Iterator

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


def buildInverseTransformsForScript() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Factory referenced by `--inverse-transform-factory` in the CLI test.

    A deliberately large, easily-distinguished rescaling (`* 100`), so a
    reconstruction figure exported with this inverse transform is
    trivially distinguishable from one exported without it (a
    completely different data range, hence a different y-axis and a
    different rendered PNG), without needing to inspect pixel values
    directly.
    """
    return {"signal": lambda tensor: tensor * 100.0}


def buildLabeledDataloaderForScript() -> list[dict[str, torch.Tensor]]:
    """Like `buildDataloaderForScript`, plus a "label" batch key the model never consumes.

    The key is not one of its modality names; it exercises `scripts/visualize_latent.py`'s
    `--label-key` coloring option.
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
    """Model factory for the cross-modal-figures CLI test.

    Two modalities ("x", "y") fused via a dummy Product-of-Experts, so `scripts/evaluate.py`'s
    opt-in `exportCrossModalFigures` call actually has something to report (spec §5,
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


def buildTwoModalityGeneratorForScript() -> Iterator[dict[str, torch.Tensor]]:
    """Like `buildTwoModalityDataloaderForScript`, but a single-use generator.

    `scripts/evaluate.py --paired` makes several passes over the data, and a generator cannot
    give a second one: the script has to call the factory again for each pass (roadmap P3-6).
    """
    return iter(buildTwoModalityDataloaderForScript())


SIGNAL_LENGTH = 32
IMAGE_SIDE = 16


def buildSignalImageModelForScript() -> GlobalVae:
    """Model factory for the image-figures CLI test: a signal and a 16 x 16 image, fused by PoE.

    Built from the framework's own 1D and 2D CNN pairs (tiny ones), not from dummies, so the
    figures are drawn from tensors with the shapes a real image model produces: the decoder of
    `signal` yields `(N, 32)`, the decoder of `image` yields `(N, 16, 16)`.
    """
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {"encoder": "1d_cnn_encoder_v1", "decoder": "1d_cnn_decoder_v1"},
            "image": {"encoder": "2d_cnn_encoder_v1", "decoder": "2d_cnn_decoder_v1"},
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="poe",
        encoder_kwargs={
            "signal": {
                "latent_dim": LATENT_DIM,
                "hidden_channels": (4, 8),
                "modality_name": "signal",
            },
            "image": {
                "latent_dim": LATENT_DIM,
                "hidden_channels": (4, 8),
                "modality_name": "image",
            },
        },
        decoder_kwargs={
            "signal": {
                "latent_dim": LATENT_DIM,
                "output_length": SIGNAL_LENGTH,
                "hidden_channels": (8, 4),
                "upsample_modes": "conv_transpose",
                "modality_name": "signal",
            },
            "image": {
                "latent_dim": LATENT_DIM,
                "output_shape": (IMAGE_SIDE, IMAGE_SIDE),
                "hidden_channels": (8, 4),
                "seed_shape": (4, 4),
                "upsample_modes": "conv_transpose",
                "modality_name": "image",
            },
        },
    )


def buildSignalImageDataloaderForScript() -> list[dict[str, torch.Tensor]]:
    """Dataloader factory paired with `buildSignalImageModelForScript`."""
    torch.manual_seed(0)
    return [
        {
            "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
            "image": torch.rand(BATCH_SIZE, IMAGE_SIDE, IMAGE_SIDE),
        }
        for _ in range(2)
    ]
