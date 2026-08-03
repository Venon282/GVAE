"""End-to-end integration test for the `EN-L1-DN` configuration (spec §2.1).

`EN-L1-DN` (per-modality encoders, one fused latent, per-modality
decoders) is the Phase-1 default chosen in
`docs/adr/0001-phase1-default-configuration.md`. This test builds it
through `GlobalVae.createSingleLatent(...)`, the convenience
constructor introduced in
`docs/adr/0002-generalize-global-vae-to-routing-graph.md`, which is why
this file exists separately from that ADR: ADR 0002 flagged the
previous version of this test as stale against the new constructor
signature, since its prior content was not available for that review.

Encoders, decoder, and fusion strategy are trivial dummies (linear
layers plus a minimal Product-of-Experts fusion): this test exercises
the *assembly and routing* machinery (registries, `RoutingGraph`,
`GlobalVae.forward`, KL aggregation), not real modality architectures,
which are still deferred (spec §11, `encoders/`, `decoders/` NOTE.md).
Dummy registry names are suffixed `_en_l1_dn` so they cannot collide
with dummy fixtures in sibling integration tests for the other 7
configurations in spec §2.1.
"""

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion
from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer
from global_vae.losses.regularizers.registry import registerRegularizer
from global_vae.models.global_vae import GlobalVae

SIGNAL_INPUT_DIM = 64
IMAGE_INPUT_DIM = 32
LATENT_DIM = 16
BATCH_SIZE = 4


@registerEncoder("dummy_signal_encoder_en_l1_dn")
class _DummySignalEncoder(AbstractEncoder):
    """Trivial linear stand-in for a real 1D-signal encoder (spec §6)."""

    def __init__(self, input_dim: int = SIGNAL_INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
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
    def modality_name(self):
        return "dummy"

    @property
    def minimal_input_length(self):
        return 10


@registerEncoder("dummy_image_encoder_en_l1_dn")
class _DummyImageEncoder(AbstractEncoder):
    """Trivial linear stand-in for a real image encoder (spec §6)."""

    def __init__(self, input_dim: int = IMAGE_INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
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
    def modality_name(self):
        return "dummy"

    @property
    def minimal_input_length(self):
        return 10


@registerDecoder("dummy_signal_decoder_en_l1_dn")
class _DummySignalDecoder(AbstractDecoder):
    """Trivial linear stand-in for a real 1D-signal decoder (spec §6)."""

    def __init__(self, output_dim: int = SIGNAL_INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


@registerDecoder("dummy_image_decoder_en_l1_dn")
class _DummyImageDecoder(AbstractDecoder):
    """Trivial linear stand-in for a real image decoder (spec §6)."""

    def __init__(self, output_dim: int = IMAGE_INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "image"


@registerFusion("dummy_poe_en_l1_dn")
class _DummyProductOfExperts(AbstractFusion):
    """Minimal Product-of-Experts fusion (spec §4), for this test only.

    Combines per-modality Gaussian experts into a precision-weighted
    posterior. Natively subset-tolerant: a missing expert is simply
    absent from `params` (spec §5). Not the production `poe.py`
    implementation (not yet built); this only exercises the "several
    encoders feed one latent space" path of `GlobalVae`.
    """

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not params:
            raise ValueError("_DummyProductOfExperts received an empty `params` dict.")

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
        fused_mu = weighted_mu_sum / total_precision
        fused_logvar = -torch.log(total_precision)
        return fused_mu, fused_logvar

    @property
    def handlesMissingModalities(self) -> bool:
        return True


@registerRegularizer("dummy_zero_regularizer_en_l1_dn")
class _DummyZeroRegularizer(AbstractLatentRegularizer):
    """Always-zero penalty, for this test only: proves a non-default
    `regularizer_strategy` is actually wired in and used, rather than
    silently falling back to `kl_standard_normal`."""

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(mu.shape[0])


@pytest.fixture
def model() -> GlobalVae:
    """Build the `EN-L1-DN` model via the Phase-1 default convenience constructor."""
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_en_l1_dn",
                "decoder": "dummy_signal_decoder_en_l1_dn",
            },
            "image": {
                "encoder": "dummy_image_encoder_en_l1_dn",
                "decoder": "dummy_image_decoder_en_l1_dn",
            },
        },
        fusion_strategy="dummy_poe_en_l1_dn",
        latent_dim=LATENT_DIM,
    )


def _dummyInputs() -> dict[str, torch.Tensor]:
    return {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_INPUT_DIM),
        "image": torch.randn(BATCH_SIZE, IMAGE_INPUT_DIM),
    }


class TestEnL1DnDefault:
    """End-to-end shape and gradient sanity checks (spec §10) for `EN-L1-DN`."""

    def test_forward_shapes_all_modalities_present(self, model: GlobalVae) -> None:
        output = model(_dummyInputs())

        assert set(output["reconstructions"]) == {"signal", "image"}
        assert output["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_INPUT_DIM)
        assert output["reconstructions"]["image"].shape == (BATCH_SIZE, IMAGE_INPUT_DIM)

        assert set(output["latent_params"]) == {"z_fused"}
        mu, logvar = output["latent_params"]["z_fused"]
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)
        assert output["latent_samples"]["z_fused"].shape == (BATCH_SIZE, LATENT_DIM)

    def test_forward_with_missing_modality(self, model: GlobalVae) -> None:
        """The model must still run and decode every modality with a partial input (spec §5)."""
        output = model({"signal": torch.randn(BATCH_SIZE, SIGNAL_INPUT_DIM)})

        assert set(output["reconstructions"]) == {"signal", "image"}
        assert "z_fused" in output["latent_params"]

    def test_forward_rejects_empty_inputs(self, model: GlobalVae) -> None:
        with pytest.raises(ValueError):
            model({})

    def test_regularization_loss_is_finite_scalar(self, model: GlobalVae) -> None:
        output = model(_dummyInputs())
        regularization_loss = model.computeRegularizationLoss(output["latent_params"])

        assert regularization_loss.dim() == 0
        assert torch.isfinite(regularization_loss)

    def test_default_regularizer_is_kl_standard_normal(self, model: GlobalVae) -> None:
        """Not passing `regularizer_strategy` at all must still wire in the default (spec §2.3)."""
        assert isinstance(model.regularizers["z_fused"], KlStandardNormalRegularizer)

    def test_custom_regularizer_strategy_is_used(self) -> None:
        """`regularizer_strategy` must actually select the given strategy, not just accept it."""
        zero_reg_model = GlobalVae.createSingleLatent(
            modality_configs={
                "signal": {
                    "encoder": "dummy_signal_encoder_en_l1_dn",
                    "decoder": "dummy_signal_decoder_en_l1_dn",
                },
            },
            fusion_strategy="dummy_poe_en_l1_dn",
            latent_dim=LATENT_DIM,
            regularizer_strategy="dummy_zero_regularizer_en_l1_dn",
        )
        output = zero_reg_model({"signal": torch.randn(BATCH_SIZE, SIGNAL_INPUT_DIM)})
        regularization_loss = zero_reg_model.computeRegularizationLoss(output["latent_params"])
        assert torch.equal(regularization_loss, torch.tensor(0.0))

    def test_gradients_flow_to_every_encoder_and_decoder(self, model: GlobalVae) -> None:
        """A full backward pass must reach every encoder and decoder (ADR 0002)."""
        output = model(_dummyInputs())
        reconstruction_loss = sum(
            (recon.pow(2).mean() for recon in output["reconstructions"].values()),
            start=torch.tensor(0.0),
        )
        kl_loss = model.computeRegularizationLoss(output["latent_params"])
        (reconstruction_loss + kl_loss).backward()

        for name, encoder in model.encoders.items():
            grads = [p.grad for p in encoder.parameters()]
            assert all(g is not None for g in grads), f"encoder '{name}' got no gradient"
            assert any(torch.any(g != 0) for g in grads if g is not None), (
                f"encoder '{name}' got an all-zero gradient"
            )

        for name, decoder in model.decoders.items():
            grads = [p.grad for p in decoder.parameters()]
            assert all(g is not None for g in grads), f"decoder '{name}' got no gradient"
            assert any(torch.any(g != 0) for g in grads if g is not None), (
                f"decoder '{name}' got an all-zero gradient"
            )

    def test_single_modality_needs_no_fusion_strategy(self) -> None:
        """A latent space fed by exactly one encoder needs no Fusion at all (spec §4).

        `fusion_strategy` must be genuinely optional here, not merely
        accept `None` and fail later: this is the `signal -> z -> signal`
        single-modality case from spec §6.1 milestone 1.
        """
        single_modality_model = GlobalVae.createSingleLatent(
            modality_configs={
                "signal": {
                    "encoder": "dummy_signal_encoder_en_l1_dn",
                    "decoder": "dummy_signal_decoder_en_l1_dn",
                },
            },
            latent_dim=LATENT_DIM,
        )
        assert "z_fused" not in single_modality_model.fusions

        output = single_modality_model({"signal": torch.randn(BATCH_SIZE, SIGNAL_INPUT_DIM)})
        assert output["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_INPUT_DIM)

    def test_missing_fusion_strategy_with_several_modalities_still_raises(self) -> None:
        """`fusion_strategy=None` must still fail loudly once there is more than one encoder."""
        with pytest.raises(ValueError, match="fusion_strategies"):
            GlobalVae.createSingleLatent(
                modality_configs={
                    "signal": {
                        "encoder": "dummy_signal_encoder_en_l1_dn",
                        "decoder": "dummy_signal_decoder_en_l1_dn",
                    },
                    "image": {
                        "encoder": "dummy_image_encoder_en_l1_dn",
                        "decoder": "dummy_image_decoder_en_l1_dn",
                    },
                },
                latent_dim=LATENT_DIM,
            )

    def test_unknown_registry_name_raises_key_error(self) -> None:
        """Sanity check for the registry error path `GlobalVae` relies on (spec §10)."""
        with pytest.raises(KeyError):
            GlobalVae.createSingleLatent(
                modality_configs={
                    "signal": {"encoder": "does_not_exist", "decoder": "does_not_exist"},
                },
                fusion_strategy="dummy_poe_en_l1_dn",
                latent_dim=LATENT_DIM,
            )
