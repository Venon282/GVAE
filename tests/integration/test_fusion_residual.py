"""Tests for the residual connection around a fusion strategy (spec §4, §9).

Three levels, mirroring how the flag reaches a user:

- `ResidualFusion` on its own, against every built-in strategy (the flag is defined once, for
  all strategies, so it is tested against all of them, not a sample);
- `GlobalVae` wiring, through both `createSingleLatent` and an explicit `RoutingGraph`;
- the config path, `residual: true` in YAML through `buildModelFromConfig`.

The defining properties are checked directly: at initialization the fused posterior is exactly
the mean of the active experts, whichever strategy is wrapped, and a gate of `1` recovers the
wrapped strategy exactly.
"""

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
from global_vae.config.model import FusionConfig, ModelConfig, buildModelFromConfig
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import getFusionClass
from global_vae.fusion.residual import ResidualFusion
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.models.global_vae import GlobalVae

DIM = 8
BATCH = 4
MODALITIES = ["a", "b", "c"]

# Every built-in strategy with the constructor kwargs it needs for three modalities of width DIM.
_STRATEGIES: list[tuple[str, dict[str, Any]]] = [
    ("poe", {}),
    ("moe", {}),
    ("concat_mlp", {"modality_dims": dict.fromkeys(MODALITIES, DIM), "latent_dim": DIM}),
    ("cross_attention", {"latent_dim": DIM, "num_heads": 2}),
]
_LEARNED = [entry for entry in _STRATEGIES if entry[0] in ("concat_mlp", "cross_attention")]


def _inner(strategy: str, kwargs: dict[str, Any]) -> AbstractFusion:
    """Build one built-in strategy through the registry, the way `GlobalVae` does."""
    return getFusionClass(strategy)(**kwargs)


def _params(
    names: list[str], requires_grad: bool = False
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Random per-modality `(mu, logvar)`; leaf tensors when `requires_grad` is set."""
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        mu = torch.randn(BATCH, DIM)
        logvar = torch.randn(BATCH, DIM) * 0.1
        if requires_grad:
            mu.requires_grad_()
            logvar.requires_grad_()
        result[name] = (mu, logvar)
    return result


def _mean(params: dict[str, tuple[torch.Tensor, torch.Tensor]], index: int) -> torch.Tensor:
    """Mean over experts of `mu` (`index=0`) or `logvar` (`index=1`)."""
    return torch.stack([pair[index] for pair in params.values()]).mean(dim=0)


def _openGates(fusion: ResidualFusion, value: float) -> None:
    """Set both learned gates to `value` without recording it in the autograd graph."""
    with torch.no_grad():
        fusion.gate_mu.fill_(value)
        fusion.gate_logvar.fill_(value)


@pytest.mark.parametrize(("strategy", "kwargs"), _STRATEGIES)
class TestResidualFusionAgainstEveryStrategy:
    """The wrapper's contract, checked against each built-in strategy."""

    def test_output_shape_matches_the_experts(self, strategy: str, kwargs: dict[str, Any]) -> None:
        """The fused posterior has shape `(batch, latent_dim)`."""
        mu, logvar = ResidualFusion(_inner(strategy, kwargs))(_params(MODALITIES))
        assert mu.shape == (BATCH, DIM)
        assert logvar.shape == (BATCH, DIM)

    def test_identity_at_initialization_is_the_mean_of_the_experts(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """With the default gates of 0, whichever strategy is wrapped, fused == mean(experts)."""
        params = _params(MODALITIES)
        mu, logvar = ResidualFusion(_inner(strategy, kwargs))(params)
        assert torch.allclose(mu, _mean(params, 0), atol=1e-6)
        assert torch.allclose(logvar, _mean(params, 1), atol=1e-6)

    def test_gate_of_one_recovers_the_wrapped_strategy_exactly(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """Enabling the flag loses nothing: gates at 1 reproduce the strategy's own output."""
        inner = _inner(strategy, kwargs)
        residual = ResidualFusion(inner)
        _openGates(residual, 1.0)
        params = _params(MODALITIES)
        expected_mu, expected_logvar = inner(params)
        mu, logvar = residual(params)
        assert torch.allclose(mu, expected_mu, atol=1e-6)
        assert torch.allclose(logvar, expected_logvar, atol=1e-6)

    def test_intermediate_gate_interpolates_linearly(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """A gate of 0.25 is `0.75 * skip + 0.25 * strategy`."""
        inner = _inner(strategy, kwargs)
        residual = ResidualFusion(inner)
        _openGates(residual, 0.25)
        params = _params(MODALITIES)
        inner_mu, _ = inner(params)
        mu, _ = residual(params)
        assert torch.allclose(mu, 0.75 * _mean(params, 0) + 0.25 * inner_mu, atol=1e-6)

    def test_missing_modality_support_mirrors_the_wrapped_strategy(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """Wrapping never changes `handlesMissingModalities`."""
        inner = _inner(strategy, kwargs)
        assert ResidualFusion(inner).handlesMissingModalities == inner.handlesMissingModalities

    def test_a_subset_of_modalities_skips_over_only_the_active_ones(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """With modality `c` absent, the skip path is the mean of `a` and `b` alone."""
        params = _params(["a", "b"])
        mu, _ = ResidualFusion(_inner(strategy, kwargs))(params)
        assert torch.allclose(mu, _mean(params, 0), atol=1e-6)

    def test_gates_receive_gradient_and_move_under_an_optimizer_step(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """The gates are trainable from the start, even at 0 (the ReZero property)."""
        residual = ResidualFusion(_inner(strategy, kwargs))
        mu, logvar = residual(_params(MODALITIES))
        (mu.sum() + logvar.sum()).backward()
        assert residual.gate_mu.grad is not None and residual.gate_mu.grad != 0
        assert residual.gate_logvar.grad is not None and residual.gate_logvar.grad != 0

        before = residual.gate_mu.detach().clone()
        torch.optim.SGD(residual.parameters(), lr=0.1).step()
        assert not torch.equal(residual.gate_mu.detach(), before)

    def test_gradient_reaches_every_expert(self, strategy: str, kwargs: dict[str, Any]) -> None:
        """Both the skip path and the strategy path carry gradient back to the inputs."""
        residual = ResidualFusion(_inner(strategy, kwargs))
        _openGates(residual, 0.5)
        params = _params(MODALITIES, requires_grad=True)
        mu, logvar = residual(params)
        (mu.sum() + logvar.sum()).backward()
        for expert_mu, expert_logvar in params.values():
            assert expert_mu.grad is not None and torch.any(expert_mu.grad != 0)
            assert expert_logvar.grad is not None

    def test_empty_params_raises(self, strategy: str, kwargs: dict[str, Any]) -> None:
        """Like every fusion, an empty `params` is a `ValueError`."""
        with pytest.raises(ValueError, match="empty"):
            ResidualFusion(_inner(strategy, kwargs))({})


@pytest.mark.parametrize(("strategy", "kwargs"), _LEARNED)
def test_wrapped_learned_strategy_trains_once_its_gate_is_open(
    strategy: str, kwargs: dict[str, Any]
) -> None:
    """A learned strategy's own parameters get gradient through a non-zero gate."""
    inner = _inner(strategy, kwargs)
    residual = ResidualFusion(inner)
    _openGates(residual, 0.5)
    mu, logvar = residual(_params(MODALITIES))
    (mu.sum() + logvar.sum()).backward()
    inner_grads = [parameter.grad for parameter in inner.parameters()]
    assert inner_grads
    assert all(grad is not None for grad in inner_grads)


class TestResidualFusionErrors:
    """Configurations with no well-defined skip path fail with a clear message."""

    def test_experts_of_different_shapes_raise(self) -> None:
        """A mean over experts of different widths is undefined, so it is named, not broadcast."""
        fusion = ResidualFusion(_inner("moe", {}))
        params = {
            "a": (torch.randn(BATCH, DIM), torch.zeros(BATCH, DIM)),
            "b": (torch.randn(BATCH, DIM + 2), torch.zeros(BATCH, DIM + 2)),
        }
        with pytest.raises(ValueError, match="share one shape"):
            fusion(params)

    def test_a_strategy_that_changes_dimensionality_raises(self) -> None:
        """A `concat_mlp` projecting to a different width leaves the skip path nowhere to land."""
        inner = _inner(
            "concat_mlp", {"modality_dims": dict.fromkeys(MODALITIES, DIM), "latent_dim": DIM // 2}
        )
        with pytest.raises(ValueError, match="one dimensionality"):
            ResidualFusion(inner)(_params(MODALITIES))

    def test_the_wrapped_strategys_own_errors_propagate_unchanged(self) -> None:
        """The wrapper does not swallow or rewrite the strategy's validation errors."""
        inner = _inner("concat_mlp", {"modality_dims": {"a": DIM}, "latent_dim": DIM})
        with pytest.raises(KeyError, match="not declared"):
            ResidualFusion(inner)(_params(["unknown"]))


@registerEncoder("dummy_encoder_test_fusion_residual")
class _DummyEncoder(AbstractEncoder):
    """Trivial linear encoder, for this test file only."""

    def __init__(self, input_dim: int = 6, latent_dim: int = DIM) -> None:
        """Build two linear heads."""
        super().__init__()
        self._latent_dim = latent_dim
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(mu, logvar)`."""
        return self.to_mu(x), self.to_logvar(x)

    @property
    def latent_dim(self) -> int:
        """Latent width."""
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        """Modality name."""
        return "dummy"

    @property
    def minimal_input_length(self) -> int:
        """Minimal input length."""
        return 1


@registerDecoder("dummy_decoder_test_fusion_residual")
class _DummyDecoder(AbstractDecoder):
    """Trivial linear decoder, for this test file only."""

    def __init__(self, output_dim: int = 6, latent_dim: int = DIM) -> None:
        """Build one linear projection."""
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from `z`."""
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        """Modality name."""
        return "dummy"


_MODALITY_CONFIGS = {
    name: {
        "encoder": "dummy_encoder_test_fusion_residual",
        "decoder": "dummy_decoder_test_fusion_residual",
    }
    for name in ("signal", "image")
}
_TWO_MODALITY_KWARGS: list[tuple[str, dict[str, Any]]] = [
    ("poe", {}),
    ("moe", {}),
    ("concat_mlp", {"modality_dims": {"signal": DIM, "image": DIM}, "latent_dim": DIM}),
    ("cross_attention", {"latent_dim": DIM, "num_heads": 2}),
]


def _model(strategy: str, kwargs: dict[str, Any], residual: bool) -> GlobalVae:
    """Build a two-modality single-latent model through `createSingleLatent`."""
    return GlobalVae.createSingleLatent(
        modality_configs=_MODALITY_CONFIGS,
        latent_dim=DIM,
        fusion_strategy=strategy,
        fusion_residual=residual,
        fusion_kwargs={"z_fused": kwargs},
    )


class TestResidualInGlobalVae:
    """`GlobalVae` applies the flag per latent space, and rejects it where no fusion exists."""

    @pytest.mark.parametrize(("strategy", "kwargs"), _TWO_MODALITY_KWARGS)
    def test_flag_wraps_the_strategy_and_the_model_trains(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """With the flag on, forward and backward work with every modality present."""
        model = _model(strategy, kwargs, residual=True)
        assert isinstance(model.fusions["z_fused"], ResidualFusion)

        outputs = model({"signal": torch.randn(BATCH, 6), "image": torch.randn(BATCH, 6)})
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        sum(r.sum() for r in outputs["reconstructions"].values()).backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    @pytest.mark.parametrize(("strategy", "kwargs"), _TWO_MODALITY_KWARGS)
    def test_flag_on_with_one_modality_missing_does_not_raise(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """Spec §5 still holds with the flag on: querying a subset of modalities works."""
        model = _model(strategy, kwargs, residual=True)
        outputs = model({"signal": torch.randn(BATCH, 6)})
        assert set(outputs["reconstructions"]) == {"signal", "image"}

    @pytest.mark.parametrize(("strategy", "kwargs"), _TWO_MODALITY_KWARGS)
    def test_flag_off_leaves_the_strategy_unwrapped(
        self, strategy: str, kwargs: dict[str, Any]
    ) -> None:
        """The default is exactly the previous behavior: no wrapper, no `inner`, no gates."""
        model = _model(strategy, kwargs, residual=False)
        fusion = model.fusions["z_fused"]
        assert not isinstance(fusion, ResidualFusion)
        assert type(fusion) is getFusionClass(strategy)
        assert not any("gate_" in key or ".inner." in key for key in model.state_dict())

    def test_flag_on_adds_exactly_the_gates_and_moves_the_strategy_under_inner(self) -> None:
        """Enabling the flag is an architecture change, visible in the state dict."""
        keys = set(_model("moe", {}, residual=True).state_dict())
        assert "fusions.z_fused.gate_mu" in keys
        assert "fusions.z_fused.gate_logvar" in keys

    def test_explicit_routing_graph_path_accepts_the_flag_per_latent_space(self) -> None:
        """`GlobalVae(..., fusion_residual={...})` works without `createSingleLatent`."""
        graph = buildSingleLatentRoutingGraph(
            encoder_names=["signal", "image"],
            decoder_names=["signal", "image"],
            latent_dim=DIM,
            latent_name="z",
        )
        model = GlobalVae(
            encoder_configs={n: c["encoder"] for n, c in _MODALITY_CONFIGS.items()},
            decoder_configs={n: c["decoder"] for n, c in _MODALITY_CONFIGS.items()},
            routing_graph=graph,
            fusion_strategies={"z": "moe"},
            fusion_residual={"z": True},
        )
        assert isinstance(model.fusions["z"], ResidualFusion)

    def test_single_modality_model_with_the_flag_raises(self) -> None:
        """A latent space fed by one encoder has no fusion to add a residual to."""
        with pytest.raises(ValueError, match="fewer than two encoders"):
            GlobalVae.createSingleLatent(
                modality_configs={"signal": _MODALITY_CONFIGS["signal"]},
                latent_dim=DIM,
                fusion_residual=True,
            )

    def test_unknown_latent_space_name_with_the_flag_raises(self) -> None:
        """Enabling the flag for a latent space that does not exist is named, not ignored."""
        graph = buildSingleLatentRoutingGraph(["signal", "image"], ["signal", "image"], DIM, "z")
        with pytest.raises(ValueError, match="does_not_exist"):
            GlobalVae(
                encoder_configs={n: c["encoder"] for n, c in _MODALITY_CONFIGS.items()},
                decoder_configs={n: c["decoder"] for n, c in _MODALITY_CONFIGS.items()},
                routing_graph=graph,
                fusion_strategies={"z": "moe"},
                fusion_residual={"does_not_exist": True},
            )

    def test_a_false_entry_for_an_unfused_latent_space_is_harmless(self) -> None:
        """Only `True` needs a fusion; `False` (the default spelled out) never raises."""
        model = GlobalVae.createSingleLatent(
            modality_configs={"signal": _MODALITY_CONFIGS["signal"]},
            latent_dim=DIM,
            fusion_residual=False,
        )
        assert len(model.fusions) == 0


class TestResidualFromConfig:
    """`residual: true` next to `strategy:` in YAML reaches the built model (spec §9)."""

    @staticmethod
    def _config(fusion_block: str) -> ModelConfig:
        """Materialize a two-signal-modality `ModelConfig` from YAML with the given fusion."""
        yaml_text = f"""
        name: two_signals
        modalities:
          signal_a:
            encoder: {{name: 1d_cnn_encoder_v1}}
            decoder:
              name: 1d_cnn_decoder_v1
              kwargs: {{output_length: 64, upsample_modes: conv_transpose}}
          signal_b:
            encoder: {{name: 1d_cnn_encoder_v1}}
            decoder:
              name: 1d_cnn_decoder_v1
              kwargs: {{output_length: 64, upsample_modes: conv_transpose}}
        latent_mode: single
        single_latent:
          dim: 16
          fusion:
{fusion_block}
        """
        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(yaml_text))
        )
        assert isinstance(config, ModelConfig)
        return config

    def test_default_is_off(self) -> None:
        """`FusionConfig.residual` defaults to `False`, so existing configs are unchanged."""
        assert FusionConfig(strategy="poe").residual is False

    def test_yaml_flag_true_builds_a_residual_fusion(self) -> None:
        """The exact spec §9 spelling, `strategy: poe` plus `residual: true`, takes effect."""
        config = self._config("            strategy: poe\n            residual: true")
        assert config.single_latent is not None
        assert config.single_latent.fusion is not None
        assert config.single_latent.fusion.residual is True

        model = buildModelFromConfig(config)
        assert isinstance(model.fusions["z_fused"], ResidualFusion)
        outputs = model({"signal_a": torch.randn(2, 64), "signal_b": torch.randn(2, 64)})
        assert outputs["reconstructions"]["signal_a"].shape == (2, 64)

    def test_yaml_without_the_flag_builds_the_bare_strategy(self) -> None:
        """Omitting `residual` keeps the strategy unwrapped."""
        model = buildModelFromConfig(self._config("            strategy: moe"))
        assert not isinstance(model.fusions["z_fused"], ResidualFusion)

    def test_flag_composes_with_a_strategy_that_needs_kwargs(self) -> None:
        """`residual: true` alongside `kwargs` for a learned strategy builds and runs."""
        block = (
            "            strategy: cross_attention\n"
            "            residual: true\n"
            "            kwargs: {latent_dim: 16, num_heads: 2}"
        )
        model = buildModelFromConfig(self._config(block))
        assert isinstance(model.fusions["z_fused"], ResidualFusion)
        model({"signal_a": torch.randn(2, 64), "signal_b": torch.randn(2, 64)})
