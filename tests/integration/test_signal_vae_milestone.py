"""Integration test for spec §6.1 milestone 1 with the *real* modules (C11).

`tests/integration/test_en_l1_dn_default.py` covers `EN-L1-DN` end to end,
but only with trivial linear dummy encoders/decoders on two modalities
(spec §2.1's `EN-L1-DN` row in the abstract). It never exercises the actual
milestone spec §6.1 describes first: "a working single-modality signal
VAE... One encoder, one latent space, one decoder — no fusion, no second
modality, just `signal -> z -> signal`... trained end to end", built from
the real `1d_cnn_encoder_v1`/`1d_cnn_decoder_v1` (`OneDCnnEncoder`/
`OneDCnnDecoder`) via `GlobalVae.createSingleLatent`, with no
`fusion_strategy` at all (spec §4: a latent space fed by exactly one
encoder needs none). This file is that missing test.

`INPUT_OUTPUT_LENGTH = 64` is `OneDCnnDecoder`'s own natural output length
under its default `hidden_channels`/`seed_length` with `upsample_modes=
"conv_transpose"` (`8 -> 16 -> 32 -> 64`, the same default doubling
documented and tested in `test_signal_decoder.py`), and comfortably above
`OneDCnnEncoder`'s own default minimum input length (`8`, from
`computeMinimumInputLength`), so both real modules can be used with their
out-of-the-box defaults, matching how a first user of this framework would
actually reach for them.
"""

import torch

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.OneDCnnDecoder import OneDCnnDecoder
from global_vae.encoders.OneDCnnEncoder import OneDCnnEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward

INPUT_OUTPUT_LENGTH = 64
LATENT_DIM = 8
BATCH_SIZE = 6


def _buildRealSignalVae(latent_dim: int = LATENT_DIM) -> GlobalVae:
    """Build the exact spec §6.1 milestone 1 model: `signal -> z -> signal`,
    real `OneDCnnEncoder`/`OneDCnnDecoder`, no fusion (single modality)."""
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {"encoder": "1d_cnn_encoder_v1", "decoder": "1d_cnn_decoder_v1"},
        },
        latent_dim=latent_dim,
        encoder_kwargs={"signal": {"latent_dim": latent_dim}},
        decoder_kwargs={
            "signal": {
                "latent_dim": latent_dim,
                "output_length": INPUT_OUTPUT_LENGTH,
                "upsample_modes": "conv_transpose",
            }
        },
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)} for _ in range(num_batches)]


class TestModelAssembly:
    def test_uses_the_real_registered_encoder_and_decoder_classes(self) -> None:
        model = _buildRealSignalVae()
        assert isinstance(model.encoders["signal"], OneDCnnEncoder)
        assert isinstance(model.decoders["signal"], OneDCnnDecoder)

    def test_no_fusion_module_is_built_for_a_single_encoder(self) -> None:
        """Spec §4: a latent space fed by exactly one encoder needs no Fusion
        strategy at all; GlobalVae.createSingleLatent's fusion_strategy stays
        the default None here, matching the milestone description exactly
        ("no fusion, no second modality")."""
        model = _buildRealSignalVae()
        assert "z_fused" not in model.fusions
        assert len(model.fusions) == 0

    def test_single_latent_space_named_z_fused(self) -> None:
        model = _buildRealSignalVae()
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.latent_spaces["z_fused"].dim == LATENT_DIM


class TestForwardPass:
    def test_reconstruction_shape_matches_the_input(self) -> None:
        model = _buildRealSignalVae()
        output = model({"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)})
        assert output["reconstructions"]["signal"].shape == (BATCH_SIZE, INPUT_OUTPUT_LENGTH)

    def test_latent_params_and_samples_have_the_configured_dimensionality(self) -> None:
        model = _buildRealSignalVae()
        output = model({"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)})
        mu, logvar = output["latent_params"]["z_fused"]
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)
        assert output["latent_samples"]["z_fused"].shape == (BATCH_SIZE, LATENT_DIM)

    def test_use_mean_true_is_deterministic(self) -> None:
        model = _buildRealSignalVae()
        x = {"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)}
        first = model(x, use_mean=True)
        second = model(x, use_mean=True)
        assert torch.equal(first["reconstructions"]["signal"], second["reconstructions"]["signal"])


class TestGradientFlow:
    def test_gradients_reach_every_encoder_and_decoder_parameter(self) -> None:
        model = _buildRealSignalVae()
        output = model({"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)})
        reconstruction_loss = output["reconstructions"]["signal"].pow(2).mean()
        regularization_loss = model.computeRegularizationLoss(output["latent_params"])
        backward(reconstruction_loss + regularization_loss)

        for name, param in model.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"


class TestTrainedEndToEnd:
    """Spec §6.1 milestone 1's own phrasing: "...trained end to end..."."""

    def test_loss_decreases_over_epochs_with_the_real_modules(self) -> None:
        torch.manual_seed(42)
        model = _buildRealSignalVae()
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 1e-3})
        dataset = _fixedDataset(num_batches=3)

        history = trainer.fit(dataset, num_epochs=20)

        first_few = sum(entry["train/loss/total"] for entry in history[:3]) / 3
        last_few = sum(entry["train/loss/total"] for entry in history[-3:]) / 3
        assert last_few < first_few
        assert all(torch.isfinite(torch.tensor(entry["train/loss/total"])) for entry in history)

    def test_regularization_loss_is_a_finite_scalar(self) -> None:
        model = _buildRealSignalVae()
        output = model({"signal": torch.randn(BATCH_SIZE, INPUT_OUTPUT_LENGTH)})
        regularization_loss = model.computeRegularizationLoss(output["latent_params"])
        assert regularization_loss.dim() == 0
        assert torch.isfinite(regularization_loss)
