"""Unit and integration tests for the fusion registry and every built-in strategy.

Covers spec §4's strategies: `poe`, `moe`, `concat_mlp`, `cross_attention`.

Mirrors the test style already used for the other self-registration registries in
this codebase (encoders, decoders, assemblers, regularizers): registration,
lookup, and the duplicate/unknown-name error paths, plus a value-correctness
check for each strategy. `poe.py` had no direct test of its own before this file
(only indirect exercise through dummy fusion classes in unrelated tests), so it
is covered here too, not just the three new strategies.

`TestFusionWiredIntoGlobalVae` additionally exercises each new strategy through
the *real* assembly path (`GlobalVae.createSingleLatent`), not only the bare
`AbstractFusion` module in isolation, following spec §10's own testing
requirement ("an integration test that instantiates each of the 8 architecture
combinations end-to-end") applied to fusion specifically: registration and
correct math are necessary but not sufficient, since a fusion strategy is only
useful once it actually slots into `GlobalVae.forward` without any change to
that class (spec §4's whole point).
"""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.concat_mlp import ConcatMlpFusion
from global_vae.fusion.cross_attention import CrossAttentionFusion
from global_vae.fusion.moe import MixtureOfExperts
from global_vae.fusion.poe import ProductOfExperts
from global_vae.fusion.registry import getFusionClass, listRegisteredFusions, registerFusion
from global_vae.models.global_vae import GlobalVae

LATENT_DIM = 8
BATCH_SIZE = 4


def _randomParams(
    names: list[str], batch: int = BATCH_SIZE, dim: int = LATENT_DIM, requires_grad: bool = False
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build a random `params` dict of the shape every `AbstractFusion.forward` expects.

    Every returned tensor is a leaf tensor when `requires_grad=True` (built, then
    `.requires_grad_()`'d in place, rather than produced by an intermediate op like
    `* 0.1`), so `.grad` actually populates after `.backward()` instead of silently
    staying `None` on a non-leaf tensor.
    """
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        mu = torch.randn(batch, dim)
        logvar = torch.randn(batch, dim) * 0.1
        if requires_grad:
            mu.requires_grad_()
            logvar.requires_grad_()
        result[name] = (mu, logvar)
    return result


# --------------------------------------------------------------------------- #
# Registry: registration, lookup, duplicate/unknown-name error paths.
# --------------------------------------------------------------------------- #


def test_poe_is_registered_by_default() -> None:
    """`poe` is registered and resolves to `ProductOfExperts`."""
    assert "poe" in listRegisteredFusions()
    assert getFusionClass("poe") is ProductOfExperts


def test_moe_is_registered_by_default() -> None:
    """`moe` is registered and resolves to `MixtureOfExperts`."""
    assert "moe" in listRegisteredFusions()
    assert getFusionClass("moe") is MixtureOfExperts


def test_concat_mlp_is_registered_by_default() -> None:
    """`concat_mlp` is registered and resolves to `ConcatMlpFusion`."""
    assert "concat_mlp" in listRegisteredFusions()
    assert getFusionClass("concat_mlp") is ConcatMlpFusion


def test_cross_attention_is_registered_by_default() -> None:
    """`cross_attention` is registered and resolves to `CrossAttentionFusion`."""
    assert "cross_attention" in listRegisteredFusions()
    assert getFusionClass("cross_attention") is CrossAttentionFusion


def test_unknown_fusion_name_raises_key_error() -> None:
    """Looking up an unregistered name raises `KeyError` naming it."""
    with pytest.raises(KeyError, match="does_not_exist"):
        getFusionClass("does_not_exist")


def test_duplicate_registration_raises_value_error() -> None:
    """Registering a second class under an existing name raises `ValueError`."""

    @registerFusion("dummy_fusion_duplicate_check")
    class _First(AbstractFusion):
        def forward(
            self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            mu, logvar = next(iter(params.values()))
            return mu, logvar

        @property
        def handlesMissingModalities(self) -> bool:
            return True

    with pytest.raises(ValueError, match="already registered"):

        @registerFusion("dummy_fusion_duplicate_check")
        class _Second(AbstractFusion):
            def forward(
                self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                mu, logvar = next(iter(params.values()))
                return mu, logvar

            @property
            def handlesMissingModalities(self) -> bool:
                return True


# --------------------------------------------------------------------------- #
# Product-of-Experts (poe.py): no prior direct test existed for this file.
# --------------------------------------------------------------------------- #


class TestProductOfExperts:
    """`poe.py`, which had no direct test before this file."""

    def test_empty_params_raises(self) -> None:
        """An empty `params` is a `ValueError`."""
        with pytest.raises(ValueError, match="empty"):
            ProductOfExperts()({})

    def test_handles_missing_modalities_is_true(self) -> None:
        """PoE reports native missing-modality tolerance."""
        assert ProductOfExperts().handlesMissingModalities is True

    def test_matches_manual_precision_weighted_formula_without_prior_expert(self) -> None:
        """The output equals the hand-computed precision-weighted average."""
        fusion = ProductOfExperts(include_prior_expert=False)
        params = _randomParams(["a", "b", "c"])
        mu, logvar = fusion(params)

        precision_sum = torch.zeros(BATCH_SIZE, LATENT_DIM)
        weighted_mu_sum = torch.zeros(BATCH_SIZE, LATENT_DIM)
        for expert_mu, expert_logvar in params.values():
            precision = torch.exp(-expert_logvar)
            precision_sum = precision_sum + precision
            weighted_mu_sum = weighted_mu_sum + precision * expert_mu
        expected_mu = weighted_mu_sum / precision_sum
        expected_logvar = -torch.log(precision_sum)

        assert torch.allclose(mu, expected_mu, atol=1e-5)
        assert torch.allclose(logvar, expected_logvar, atol=1e-5)

    def test_single_expert_without_prior_reduces_to_that_expert(self) -> None:
        """Without the prior expert, one expert fuses to itself."""
        fusion = ProductOfExperts(include_prior_expert=False)
        params = _randomParams(["only"])
        mu, logvar = fusion(params)
        assert torch.allclose(mu, params["only"][0], atol=1e-5)
        assert torch.allclose(logvar, params["only"][1], atol=1e-5)

    def test_prior_expert_pulls_a_single_expert_toward_the_prior(self) -> None:
        """Even a single expert is combined with the prior by default.

        With `include_prior_expert=True`, the fused posterior must differ from that
        expert alone (MVAE, Wu & Goodman 2018).
        """
        fusion = ProductOfExperts(include_prior_expert=True)
        params = {
            "only": (torch.full((BATCH_SIZE, LATENT_DIM), 5.0), torch.zeros(BATCH_SIZE, LATENT_DIM))
        }
        mu, _ = fusion(params)
        assert not torch.allclose(mu, params["only"][0])
        assert torch.all(mu.abs() < params["only"][0].abs())

    def test_gradients_flow_to_every_expert(self) -> None:
        """Gradient reaches every expert's `mu` and `logvar`."""
        fusion = ProductOfExperts()
        params = _randomParams(["a", "b"], requires_grad=True)
        mu, logvar = fusion(params)
        (mu.sum() + logvar.sum()).backward()
        for expert_mu, expert_logvar in params.values():
            assert expert_mu.grad is not None and torch.any(expert_mu.grad != 0)
            assert expert_logvar.grad is not None


# --------------------------------------------------------------------------- #
# Mixture-of-Experts (moe.py).
# --------------------------------------------------------------------------- #


class TestMixtureOfExperts:
    """`moe.py`: closed-form moment matching of the mixture."""

    def test_empty_params_raises(self) -> None:
        """An empty `params` is a `ValueError`."""
        with pytest.raises(ValueError, match="empty"):
            MixtureOfExperts()({})

    def test_handles_missing_modalities_is_true(self) -> None:
        """MoE reports native missing-modality tolerance."""
        assert MixtureOfExperts().handlesMissingModalities is True

    def test_matches_manual_moment_matching_formula(self) -> None:
        """The output equals the hand-computed first two moments of the mixture."""
        fusion = MixtureOfExperts()
        params = _randomParams(["a", "b", "c"])
        mu, logvar = fusion(params)

        weight = 1.0 / 3.0
        expected_mu = sum(weight * expert_mu for expert_mu, _ in params.values())
        expected_second_moment = sum(
            weight * (torch.exp(expert_logvar) + expert_mu.pow(2))
            for expert_mu, expert_logvar in params.values()
        )
        expected_logvar = torch.log(expected_second_moment - expected_mu.pow(2))

        assert torch.allclose(mu, expected_mu, atol=1e-5)
        assert torch.allclose(logvar, expected_logvar, atol=1e-4)

    def test_single_component_reduces_exactly_to_that_component(self) -> None:
        """A one-component mixture is that component, unchanged."""
        fusion = MixtureOfExperts()
        params = _randomParams(["only"])
        mu, logvar = fusion(params)
        assert torch.allclose(mu, params["only"][0], atol=1e-5)
        assert torch.allclose(logvar, params["only"][1], atol=1e-5)

    def test_disagreeing_experts_increase_fused_variance_over_any_single_expert(self) -> None:
        """Fusing two confident but disagreeing experts increases uncertainty.

        This is the signature MoE behaviour, unlike PoE, which only ever sharpens.
        """
        fusion = MixtureOfExperts()
        params = {
            "a": (torch.full((2, 4), -5.0), torch.zeros(2, 4)),
            "b": (torch.full((2, 4), 5.0), torch.zeros(2, 4)),
        }
        _, fused_logvar = fusion(params)
        assert torch.all(fused_logvar > 0.0)

    def test_custom_weights_are_renormalized_over_active_modalities(self) -> None:
        """Explicit weights are renormalized to sum to one over the active modalities."""
        fusion = MixtureOfExperts(modality_weights={"a": 3.0, "b": 1.0})
        params = _randomParams(["a", "b"])
        mu, _ = fusion(params)
        expected_mu = 0.75 * params["a"][0] + 0.25 * params["b"][0]
        assert torch.allclose(mu, expected_mu, atol=1e-5)

    def test_unspecified_weight_defaults_to_one_before_renormalization(self) -> None:
        """A modality absent from `modality_weights` gets weight `1.0` before renormalizing."""
        fusion = MixtureOfExperts(modality_weights={"a": 3.0})
        params = _randomParams(["a", "b"])
        mu, _ = fusion(params)
        expected_mu = 0.75 * params["a"][0] + 0.25 * params["b"][0]
        assert torch.allclose(mu, expected_mu, atol=1e-5)

    def test_negative_weight_raises_at_construction(self) -> None:
        """A negative weight is rejected at construction."""
        with pytest.raises(ValueError, match="non-negative"):
            MixtureOfExperts(modality_weights={"a": -1.0})

    def test_all_zero_weights_raises_at_forward(self) -> None:
        """Weights summing to zero over the active modalities raise at `forward`."""
        fusion = MixtureOfExperts(modality_weights={"a": 0.0, "b": 0.0})
        with pytest.raises(ValueError, match="sum to 0"):
            fusion(_randomParams(["a", "b"]))

    def test_gradients_flow_to_every_component(self) -> None:
        """Gradient reaches every component's `mu` and `logvar`."""
        fusion = MixtureOfExperts()
        params = _randomParams(["a", "b"], requires_grad=True)
        mu, logvar = fusion(params)
        (mu.sum() + logvar.sum()).backward()
        for expert_mu, expert_logvar in params.values():
            assert expert_mu.grad is not None and torch.any(expert_mu.grad != 0)
            assert expert_logvar.grad is not None


# --------------------------------------------------------------------------- #
# Concatenation + MLP (concat_mlp.py).
# --------------------------------------------------------------------------- #


class TestConcatMlpFusion:
    """`concat_mlp.py`: concatenation, MLP projection, and its explicit imputation scheme."""

    def test_empty_modality_dims_raises_at_construction(self) -> None:
        """Declaring no modalities is rejected at construction."""
        with pytest.raises(ValueError, match="at least one entry"):
            ConcatMlpFusion(modality_dims={}, latent_dim=LATENT_DIM)

    def test_empty_params_raises_at_forward(self) -> None:
        """An empty `params` is a `ValueError`."""
        fusion = ConcatMlpFusion(modality_dims={"a": LATENT_DIM}, latent_dim=LATENT_DIM)
        with pytest.raises(ValueError, match="empty"):
            fusion({})

    def test_handles_missing_modalities_is_false(self) -> None:
        """Concat+MLP does not report native tolerance (spec §4, §5)."""
        fusion = ConcatMlpFusion(modality_dims={"a": LATENT_DIM}, latent_dim=LATENT_DIM)
        assert fusion.handlesMissingModalities is False

    def test_output_shape_with_every_modality_present(self) -> None:
        """The fused posterior has shape `(batch, latent_dim)`."""
        fusion = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM, "b": LATENT_DIM}, latent_dim=LATENT_DIM
        )
        mu, logvar = fusion(_randomParams(["a", "b"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_missing_modality_is_imputed_instead_of_raising(self) -> None:
        """A declared but absent modality is imputed, not an error."""
        fusion = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM, "b": LATENT_DIM}, latent_dim=LATENT_DIM
        )
        mu, logvar = fusion({"a": _randomParams(["a"])["a"]})
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_presence_mask_changes_output_relative_to_pure_zero_imputation(self) -> None:
        """A missing slot must be distinguishable from a genuinely-zero present modality.

        The presence mask is what makes the difference: without it, the two inputs are
        identical to the MLP.
        """
        torch.manual_seed(0)
        with_mask = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM, "b": LATENT_DIM},
            latent_dim=LATENT_DIM,
            use_presence_mask=True,
        )
        torch.manual_seed(0)
        without_mask = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM, "b": LATENT_DIM},
            latent_dim=LATENT_DIM,
            use_presence_mask=False,
        )

        near_zero_a = (torch.full((2, LATENT_DIM), 1e-6), torch.full((2, LATENT_DIM), 1e-6))
        # Present, near-zero "a" and "b" both truly at zero (identical raw values to the
        # imputed zeros a missing "b" would receive).
        both_present = {
            "a": near_zero_a,
            "b": (torch.zeros(2, LATENT_DIM), torch.zeros(2, LATENT_DIM)),
        }
        b_missing = {"a": near_zero_a}

        mu_with_mask_present, _ = with_mask(both_present)
        mu_with_mask_missing, _ = with_mask(b_missing)
        assert not torch.allclose(mu_with_mask_present, mu_with_mask_missing)

        mu_without_mask_present, _ = without_mask(both_present)
        mu_without_mask_missing, _ = without_mask(b_missing)
        assert torch.allclose(mu_without_mask_present, mu_without_mask_missing, atol=1e-5)

    def test_unknown_modality_raises_key_error(self) -> None:
        """A modality never declared in `modality_dims` raises `KeyError`."""
        fusion = ConcatMlpFusion(modality_dims={"a": LATENT_DIM}, latent_dim=LATENT_DIM)
        with pytest.raises(KeyError, match="unknown"):
            fusion({"unknown": _randomParams(["unknown"])["unknown"]})

    def test_modality_order_is_insertion_order_not_alphabetical(self) -> None:
        """Slot order follows the mapping's insertion order, not alphabetical order."""
        fusion = ConcatMlpFusion(
            modality_dims={"z": LATENT_DIM, "a": LATENT_DIM}, latent_dim=LATENT_DIM
        )
        assert fusion._modality_order == ["z", "a"]

    def test_no_hidden_layers_still_builds_a_valid_module(self) -> None:
        """`hidden_dims=()` gives a single linear layer per head and still works."""
        fusion = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM},
            latent_dim=LATENT_DIM,
            hidden_dims=(),
            use_presence_mask=False,
        )
        mu, logvar = fusion(_randomParams(["a"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_gradients_flow(self) -> None:
        """Gradient reaches every expert's `mu` and `logvar`."""
        fusion = ConcatMlpFusion(
            modality_dims={"a": LATENT_DIM, "b": LATENT_DIM}, latent_dim=LATENT_DIM
        )
        params = _randomParams(["a", "b"], requires_grad=True)
        mu, logvar = fusion(params)
        (mu.sum() + logvar.sum()).backward()
        for expert_mu, expert_logvar in params.values():
            assert expert_mu.grad is not None and torch.any(expert_mu.grad != 0)
            assert expert_logvar.grad is not None


# --------------------------------------------------------------------------- #
# Cross-attention / transformer fusion (cross_attention.py).
# --------------------------------------------------------------------------- #


class TestCrossAttentionFusion:
    """`cross_attention.py`: one token per active modality, Transformer, mean pool."""

    def test_empty_params_raises(self) -> None:
        """An empty `params` is a `ValueError`."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        with pytest.raises(ValueError, match="empty"):
            fusion({})

    def test_handles_missing_modalities_is_true(self) -> None:
        """Cross-attention reports native missing-modality tolerance."""
        assert (
            CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2).handlesMissingModalities
            is True
        )

    def test_num_heads_not_dividing_d_model_raises(self) -> None:
        """`num_heads` must divide `d_model`, checked at construction."""
        with pytest.raises(ValueError, match="evenly divisible"):
            CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=3)

    def test_output_shape_with_several_active_modalities(self) -> None:
        """The fused posterior has shape `(batch, latent_dim)`."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        mu, logvar = fusion(_randomParams(["a", "b", "c"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_output_shape_with_a_single_active_modality(self) -> None:
        """A single token must still attend fine (to itself)."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        mu, logvar = fusion(_randomParams(["only"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_custom_d_model_is_used(self) -> None:
        """A `d_model` different from `latent_dim` is honored."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, d_model=32, num_heads=4)
        assert fusion.d_model == 32
        mu, logvar = fusion(_randomParams(["a", "b"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_unknown_modality_raises_when_known_modalities_given(self) -> None:
        """A modality outside `known_modalities` raises `ValueError`."""
        fusion = CrossAttentionFusion(
            latent_dim=LATENT_DIM, num_heads=2, known_modalities=["a", "b"]
        )
        with pytest.raises(ValueError, match="unknown"):
            fusion(_randomParams(["unknown"]))

    def test_without_known_modalities_any_modality_name_is_accepted(self) -> None:
        """With no `known_modalities`, any modality name is accepted."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        mu, logvar = fusion(_randomParams(["anything_goes"]))
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)

    def test_modality_embeddings_are_only_built_when_known_modalities_given(self) -> None:
        """Per-modality embeddings exist only when `known_modalities` is given."""
        with_embeddings = CrossAttentionFusion(
            latent_dim=LATENT_DIM, num_heads=2, known_modalities=["a"]
        )
        without_embeddings = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        assert with_embeddings.modality_embeddings is not None
        assert without_embeddings.modality_embeddings is None

    def test_gradients_flow(self) -> None:
        """Gradient reaches every expert's `mu` and `logvar`."""
        fusion = CrossAttentionFusion(latent_dim=LATENT_DIM, num_heads=2)
        params = _randomParams(["a", "b"], requires_grad=True)
        mu, logvar = fusion(params)
        (mu.sum() + logvar.sum()).backward()
        for expert_mu, expert_logvar in params.values():
            assert expert_mu.grad is not None and torch.any(expert_mu.grad != 0)
            assert expert_logvar.grad is not None


# --------------------------------------------------------------------------- #
# Wired into GlobalVae end to end (spec §4: a fusion strategy must slot into
# GlobalVae.forward with zero change to that class).
# --------------------------------------------------------------------------- #


@registerEncoder("dummy_encoder_test_fusion")
class _DummyEncoder(AbstractEncoder):
    """Trivial linear stand-in encoder, for this test file only."""

    def __init__(self, input_dim: int = 6, latent_dim: int = LATENT_DIM) -> None:
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
        return "dummy"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_decoder_test_fusion")
class _DummyDecoder(AbstractDecoder):
    """Trivial linear stand-in decoder, for this test file only."""

    def __init__(self, output_dim: int = 6, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "dummy"


_MODALITY_CONFIGS = {
    "signal": {"encoder": "dummy_encoder_test_fusion", "decoder": "dummy_decoder_test_fusion"},
    "image": {"encoder": "dummy_encoder_test_fusion", "decoder": "dummy_decoder_test_fusion"},
}


def _buildModel(fusion_strategy: str, fusion_kwargs: dict[str, object]) -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs=_MODALITY_CONFIGS,
        latent_dim=LATENT_DIM,
        fusion_strategy=fusion_strategy,
        fusion_kwargs={"z_fused": fusion_kwargs},
    )


class TestFusionWiredIntoGlobalVae:
    """Each new strategy through the real `GlobalVae.createSingleLatent` assembly path."""

    @pytest.mark.parametrize(
        ("strategy", "kwargs_factory"),
        [
            ("moe", lambda: {}),
            (
                "concat_mlp",
                lambda: {
                    "latent_dim": LATENT_DIM,
                    "modality_dims": {"signal": LATENT_DIM, "image": LATENT_DIM},
                },
            ),
            ("cross_attention", lambda: {"latent_dim": LATENT_DIM, "num_heads": 2}),
        ],
    )
    def test_forward_and_backward_with_every_modality_present(
        self, strategy: str, kwargs_factory: Callable[[], dict[str, object]]
    ) -> None:
        """Forward and backward work with every modality present."""
        model = _buildModel(strategy, kwargs_factory())
        inputs = {"signal": torch.randn(BATCH_SIZE, 6), "image": torch.randn(BATCH_SIZE, 6)}

        outputs = model(inputs)
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        for reconstruction in outputs["reconstructions"].values():
            assert reconstruction.shape == (BATCH_SIZE, 6)

        total_loss = sum(r.sum() for r in outputs["reconstructions"].values())
        total_loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @pytest.mark.parametrize(
        ("strategy", "kwargs_factory"),
        [
            ("moe", lambda: {}),
            (
                "concat_mlp",
                lambda: {
                    "latent_dim": LATENT_DIM,
                    "modality_dims": {"signal": LATENT_DIM, "image": LATENT_DIM},
                },
            ),
            ("cross_attention", lambda: {"latent_dim": LATENT_DIM, "num_heads": 2}),
        ],
    )
    def test_forward_with_one_modality_missing(
        self, strategy: str, kwargs_factory: Callable[[], dict[str, object]]
    ) -> None:
        """Spec §5: querying with a subset of modalities must not raise.

        `concat_mlp` handles it through its own explicit imputation scheme.
        """
        model = _buildModel(strategy, kwargs_factory())
        outputs = model({"signal": torch.randn(BATCH_SIZE, 6)})
        assert set(outputs["reconstructions"]) == {"signal", "image"}

    def test_regularization_loss_is_finite_for_every_new_strategy(self) -> None:
        """The regularization loss on the fused posterior is finite."""
        for strategy, kwargs in [
            ("moe", {}),
            (
                "concat_mlp",
                {
                    "latent_dim": LATENT_DIM,
                    "modality_dims": {"signal": LATENT_DIM, "image": LATENT_DIM},
                },
            ),
            ("cross_attention", {"latent_dim": LATENT_DIM, "num_heads": 2}),
        ]:
            model = _buildModel(strategy, kwargs)
            inputs = {"signal": torch.randn(BATCH_SIZE, 6), "image": torch.randn(BATCH_SIZE, 6)}
            outputs = model(inputs)
            loss = model.computeRegularizationLoss(outputs["latent_params"])
            assert torch.isfinite(loss)
