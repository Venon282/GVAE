"""The two pair-stage loss terms, `latent_alignment` and `cross_reconstruction` (roadmap P6-4).

ADR 0054 decides what they compute and what they refuse. The model here is two modalities that each
have a latent space of their own (`z_signal`, `z_other`), with small linear encoders and decoders so
that every test is fast, and the values are checked against numbers computed by hand or by the
functions the terms are built on. What is pinned:

- the arguments: `pairs` (a non-empty list of two different names, each pair once), `distance`
  (required: a name or `{name, kwargs}`), `stop_gradient` (`none`, `source` or `target`);
- the build-time checks: a name that is not a latent space of the model, two spaces of different
  widths (roadmap D-16), a target that no decoder can decode alone, a trainer with modality
  dropout;
- the values: the batch mean of the distance on the posteriors of the step, summed over the pairs;
  the trainer's own loss, weight and reduction for each modality of a cross decoding;
- the rule that a batch must carry every modality of a pair, and is refused by name when it does
  not;
- both sides of `stop_gradient`, and that the parameters' `requires_grad` flags are put back;
- the terms run through the real `Trainer`: the value in the metrics, the weight, the history.
"""

from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from omegaconf import OmegaConf
from torch import nn

from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.training import buildLossTerms
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.losses.distances.mean_mse import MeanMseDistance
from global_vae.losses.distances.wasserstein2_diag import Wasserstein2DiagDistance
from global_vae.losses.reconstruction.aggregate import computeTotalReconstructionLoss
from global_vae.losses.terms.base import TrainerSettings
from global_vae.losses.terms.cross_reconstruction import CrossReconstructionTerm
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.losses.terms.pairs import (
    ANCHOR_DIRECTIONS,
    STOP_GRADIENT_CHOICES,
    checkStopGradient,
    parsePairs,
    requireModalitiesInBatch,
    resolvePairs,
)
from global_vae.losses.terms.registry import (
    getLossTermClass,
    listRegisteredLossTerms,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import StepLosses, Trainer

IN_DIM = 8
LATENT = 3
HIDDEN = 10
BATCH = 6


@registerEncoder("dummy_encoder_pair_terms_test")
class _Encoder(AbstractEncoder):
    def __init__(self, input_dim: int, latent_dim: int, modality: str) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._modality = modality
        self.hidden = nn.Linear(input_dim, HIDDEN)
        self.to_mu = nn.Linear(HIDDEN, latent_dim)
        self.to_logvar = nn.Linear(HIDDEN, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.hidden(x))
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


@registerDecoder("dummy_decoder_pair_terms_test")
class _Decoder(AbstractDecoder):
    def __init__(self, output_dim: int, latent_dim: int, modality: str) -> None:
        super().__init__()
        self._modality = modality
        self.hidden = nn.Linear(latent_dim, HIDDEN)
        self.project = nn.Linear(HIDDEN, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(torch.tanh(self.hidden(z)))
        return reconstruction

    @property
    def modality_name(self) -> str:
        return self._modality


def _model(widths: Mapping[str, int] | None = None, seed: int = 0) -> GlobalVae:
    """Modalities with a latent space each: `signal` and `other` by default, widths by name."""
    widths = widths or {"signal": LATENT, "other": LATENT}
    torch.manual_seed(seed)
    graph = RoutingGraph(
        latent_specs={f"z_{m}": LatentSpace(f"z_{m}", width) for m, width in widths.items()},
        encoder_to_latents={m: [f"z_{m}"] for m in widths},
        latent_to_decoders={f"z_{m}": [m] for m in widths},
    )
    return GlobalVae(
        encoder_configs={m: "dummy_encoder_pair_terms_test" for m in widths},
        decoder_configs={m: "dummy_decoder_pair_terms_test" for m in widths},
        routing_graph=graph,
        encoder_kwargs={
            m: {"input_dim": IN_DIM, "latent_dim": width, "modality": m}
            for m, width in widths.items()
        },
        decoder_kwargs={
            m: {"output_dim": IN_DIM, "latent_dim": width, "modality": m}
            for m, width in widths.items()
        },
    )


def _batch(modalities: tuple[str, ...] = ("signal", "other"), seed: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {m: torch.randn(BATCH, IN_DIM, generator=generator) for m in modalities}


def _outputs(model: GlobalVae, batch: dict[str, Any], seed: int = 1) -> dict[str, Any]:
    torch.manual_seed(seed)
    return dict(model(model.selectEncoderInputs(batch)))


def _settings(**changes: Any) -> TrainerSettings:
    fields: dict[str, Any] = {
        "reconstruction_loss_fn": F.mse_loss,
        "reconstruction_weights": 1.0,
        "reconstruction_reduction": "mean",
        "modality_dropout_p": 0.0,
    }
    fields.update(changes)
    return TrainerSettings(**fields)


def _alignment(model: GlobalVae | None = None, **kwargs: Any) -> LatentAlignmentTerm:
    kwargs.setdefault("pairs", [["z_signal", "z_other"]])
    kwargs.setdefault("distance", "mean_mse")
    term = LatentAlignmentTerm(**kwargs)
    term.validateModel(model or _model())
    term.bindTrainerSettings(_settings())
    return term


def _cross(
    model: GlobalVae | None = None, settings: TrainerSettings | None = None, **kwargs: Any
) -> CrossReconstructionTerm:
    kwargs.setdefault("pairs", [["z_signal", "z_other"]])
    term = CrossReconstructionTerm(**kwargs)
    term.validateModel(model or _model())
    term.bindTrainerSettings(settings or _settings())
    return term


def _params(model: GlobalVae, *paths: str) -> list[nn.Parameter]:
    """The parameters at these dotted paths, e.g. `encoders.signal.to_mu.weight`."""
    return [model.get_parameter(path) for path in paths]


def _reachedBy(loss: torch.Tensor, parameters: Any) -> list[bool]:
    """Backpropagate `loss` and say, for each parameter, whether a gradient reached it."""
    for parameter in parameters:
        parameter.grad = None
    loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
    return [
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in parameters
    ]


class TestRegistration:
    def test_both_terms_are_registered_built_ins(self) -> None:
        assert {"latent_alignment", "cross_reconstruction"} <= set(listRegisteredLossTerms())
        assert getLossTermClass("latent_alignment") is LatentAlignmentTerm
        assert getLossTermClass("cross_reconstruction") is CrossReconstructionTerm

    def test_the_terms_have_no_parameters_and_no_state(self) -> None:
        for term in (_alignment(), _cross()):
            assert list(term.parameters()) == []
            assert term.state_dict() == {}

    def test_the_trainer_settings_hold_the_four_settings_a_term_may_read(self) -> None:
        settings = _settings(modality_dropout_p=0.3)
        assert settings.modality_dropout_p == 0.3
        with pytest.raises(AttributeError):
            settings.modality_dropout_p = 0.0  # type: ignore[misc]


class TestParsePairs:
    def test_a_list_of_lists_is_accepted_and_returned_as_tuples(self) -> None:
        assert parsePairs([["a", "b"], ["b", "c"]], "t", True) == (("a", "b"), ("b", "c"))

    def test_tuples_and_an_omegaconf_list_are_accepted(self) -> None:
        assert parsePairs((("a", "b"),), "t", True) == (("a", "b"),)
        assert parsePairs(OmegaConf.create([["a", "b"]]), "t", True) == (("a", "b"),)

    @pytest.mark.parametrize("pairs", ["a,b", ("a", "b"), 3, None, {"a": "b"}])
    def test_pairs_that_are_not_a_list_of_pairs_are_refused(self, pairs: Any) -> None:
        with pytest.raises(ValueError, match="t: .*pair"):
            parsePairs(pairs, "t", True)

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="t: `pairs` is empty"):
            parsePairs([], "t", True)

    @pytest.mark.parametrize(
        "entry", [["a"], ["a", "b", "c"], "ab", [1, 2], ["a", ""], ["a", None], 5]
    )
    def test_an_entry_that_is_not_two_names_is_refused_and_named(self, entry: Any) -> None:
        with pytest.raises(ValueError, match=r"t: pairs\[1\] must be a pair of two latent space"):
            parsePairs([["a", "b"], entry], "t", True)

    def test_a_space_paired_with_itself_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"pairs\[0\] pairs the latent space 'a' with itself"):
            parsePairs([["a", "a"]], "t", True)

    def test_a_pair_listed_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"pairs\[1\] lists the pair \[a, b\] a second time"):
            parsePairs([["a", "b"], ["a", "b"]], "t", True)

    def test_the_reverse_is_a_second_pair_only_when_the_order_means_nothing(self) -> None:
        assert parsePairs([["a", "b"], ["b", "a"]], "t", True) == (("a", "b"), ("b", "a"))
        with pytest.raises(ValueError, match=r"a second time \(in either order\)"):
            parsePairs([["a", "b"], ["b", "a"]], "t", False)


class TestStopGradientCheck:
    @pytest.mark.parametrize("value", STOP_GRADIENT_CHOICES)
    def test_each_choice_is_accepted(self, value: str) -> None:
        assert checkStopGradient(value, "t") == value

    def test_the_choices_are_none_source_and_target(self) -> None:
        assert STOP_GRADIENT_CHOICES == ("none", "source", "target")

    @pytest.mark.parametrize("value", ["both", "", None, True, "Source"])
    def test_anything_else_is_refused_and_the_choices_listed(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"t: stop_gradient must be one of \['none', 'source'"):
            checkStopGradient(value, "t")


class TestLatentAlignmentArguments:
    def test_the_distance_is_required(self) -> None:
        with pytest.raises(TypeError, match="distance"):
            LatentAlignmentTerm(pairs=[["a", "b"]])  # type: ignore[call-arg]

    def test_a_name_builds_the_distance_with_its_defaults(self) -> None:
        term = LatentAlignmentTerm(pairs=[["a", "b"]], distance="mean_mse")
        assert isinstance(term.distance, MeanMseDistance)
        assert term.distance.dimension_reduction == "sum"

    def test_a_mapping_gives_the_name_and_the_kwargs(self) -> None:
        term = LatentAlignmentTerm(
            pairs=[["a", "b"]],
            distance={"name": "mean_mse", "kwargs": {"dimension_reduction": "mean"}},
        )
        assert term.distance.dimension_reduction == "mean"

    def test_a_mapping_may_leave_the_kwargs_out_or_null(self) -> None:
        for distance in ({"name": "mean_mse"}, {"name": "mean_mse", "kwargs": None}):
            term = LatentAlignmentTerm(pairs=[["a", "b"]], distance=distance)
            assert isinstance(term.distance, MeanMseDistance)

    def test_an_omegaconf_mapping_is_accepted(self) -> None:
        distance = OmegaConf.create({"name": "wasserstein2_diag", "kwargs": {"squared": False}})
        term = LatentAlignmentTerm(pairs=[["a", "b"]], distance=distance)
        assert isinstance(term.distance, Wasserstein2DiagDistance)
        assert term.distance.squared is False

    def test_an_unknown_distance_is_a_key_error_that_lists_the_registered_ones(self) -> None:
        with pytest.raises(KeyError, match="wasserstein2_diag"):
            LatentAlignmentTerm(pairs=[["a", "b"]], distance="euclid")

    @pytest.mark.parametrize(
        "distance", [{"kwargs": {}}, {"name": "mean_mse", "weight": 1}, {"name": "x", "k": 1}]
    )
    def test_a_mapping_takes_only_name_and_kwargs(self, distance: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="takes the keys `name` and `kwargs`"):
            LatentAlignmentTerm(pairs=[["a", "b"]], distance=distance)

    @pytest.mark.parametrize("distance", [3, None, ["mean_mse"]])
    def test_anything_else_is_refused_and_says_the_distance_is_required(
        self, distance: Any
    ) -> None:
        with pytest.raises(ValueError, match="name of a latent distance or a mapping"):
            LatentAlignmentTerm(pairs=[["a", "b"]], distance=distance)

    def test_bad_pairs_and_a_bad_stop_gradient_are_refused(self) -> None:
        with pytest.raises(ValueError, match="latent_alignment: `pairs` is empty"):
            LatentAlignmentTerm(pairs=[], distance="mean_mse")
        with pytest.raises(ValueError, match="latent_alignment: stop_gradient must be one of"):
            LatentAlignmentTerm(pairs=[["a", "b"]], distance="mean_mse", stop_gradient="x")

    def test_a_symmetric_distance_counts_the_two_orders_of_a_pair_once(self) -> None:
        with pytest.raises(ValueError, match="in either order"):
            LatentAlignmentTerm(pairs=[["a", "b"], ["b", "a"]], distance="mean_mse")

    def test_an_asymmetric_distance_may_list_both_orders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(MeanMseDistance, "is_symmetric", False)
        term = LatentAlignmentTerm(pairs=[["a", "b"], ["b", "a"]], distance="mean_mse")
        assert term.pairs == (("a", "b"), ("b", "a"))


class TestLatentAlignmentValidateModel:
    def test_a_name_that_is_not_a_latent_space_is_refused_and_the_real_ones_listed(self) -> None:
        term = LatentAlignmentTerm(pairs=[["z_signal", "z_oops"]], distance="mean_mse")
        with pytest.raises(
            ValueError,
            match=r"names 'z_oops', which is not a latent space.*\['z_other', 'z_signal'\]",
        ):
            term.validateModel(_model())

    def test_two_spaces_of_different_widths_are_refused_with_both_widths(self) -> None:
        model = _model({"signal": 3, "other": 5})
        term = LatentAlignmentTerm(pairs=[["z_signal", "z_other"]], distance="mean_mse")
        with pytest.raises(ValueError, match=r"latent widths 3 and 5.*roadmap D-16"):
            term.validateModel(model)

    def test_equal_widths_pass(self) -> None:
        _alignment(_model({"signal": 4, "other": 4}))

    def test_a_trainer_with_modality_dropout_is_refused(self) -> None:
        term = LatentAlignmentTerm(pairs=[["z_signal", "z_other"]], distance="mean_mse")
        with pytest.raises(ValueError, match=r"modality dropout must be off.*modality_dropout_p=0"):
            term.bindTrainerSettings(_settings(modality_dropout_p=0.1))


class TestLatentAlignmentForward:
    def test_it_is_the_batch_mean_of_the_distance_on_the_posteriors_of_the_step(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        term = _alignment(model)
        params = outputs["latent_params"]
        expected = MeanMseDistance()(params["z_signal"], params["z_other"]).mean()
        assert torch.equal(term(model, batch, outputs, 0), expected)

    def test_it_reads_the_posteriors_it_is_given_and_encodes_nothing(self) -> None:
        model, batch = _model(), _batch()
        mu_a, mu_b = torch.zeros(BATCH, LATENT), torch.ones(BATCH, LATENT)
        logvar = torch.zeros(BATCH, LATENT)
        outputs = {"latent_params": {"z_signal": (mu_a, logvar), "z_other": (mu_b, logvar)}}
        # A gap of 1 in each of the 3 dimensions, summed over the dimensions: 3 for every sample.
        assert _alignment(model)(model, batch, outputs, 0).item() == pytest.approx(3.0)

    def test_the_value_is_a_scalar_in_float32_whatever_the_precision_of_the_posteriors(
        self,
    ) -> None:
        model, batch = _model(), _batch()
        half = torch.zeros(BATCH, LATENT, dtype=torch.bfloat16)
        outputs = {"latent_params": {"z_signal": (half, half), "z_other": (half + 1, half)}}
        value = _alignment(model)(model, batch, outputs, 0)
        assert value.shape == () and value.dtype == torch.float32

    def test_pairs_are_summed(self) -> None:
        model = _model({"signal": LATENT, "other": LATENT, "third": LATENT})
        batch = _batch(("signal", "other", "third"))
        outputs = _outputs(model, batch)
        both = _alignment(model, pairs=[["z_signal", "z_other"], ["z_signal", "z_third"]])
        first = _alignment(model, pairs=[["z_signal", "z_other"]])
        second = _alignment(model, pairs=[["z_signal", "z_third"]])
        assert torch.allclose(
            both(model, batch, outputs, 0),
            first(model, batch, outputs, 0) + second(model, batch, outputs, 0),
        )

    def test_the_distance_chosen_is_the_distance_used(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        values = {
            name: _alignment(model, distance=name)(model, batch, outputs, 0).item()
            for name in ("mean_mse", "wasserstein2_diag", "symmetric_kl", "cosine")
        }
        assert len(set(values.values())) == 4

    def test_the_kwargs_of_the_distance_reach_it(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        total = _alignment(model, distance="mean_mse")(model, batch, outputs, 0)
        mean = _alignment(
            model, distance={"name": "mean_mse", "kwargs": {"dimension_reduction": "mean"}}
        )(model, batch, outputs, 0)
        assert torch.allclose(mean, total / LATENT)

    def test_the_distance_refuses_what_it_cannot_score_and_says_so(self) -> None:
        model = _model()
        batch = {m: v[:1] for m, v in _batch().items()}
        outputs = _outputs(model, batch)
        with pytest.raises(ValueError, match="InfoNceDistance needs at least 2 samples"):
            _alignment(model, distance="info_nce")(model, batch, outputs, 0)

    def test_the_gradient_reaches_both_spaces_by_default(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        loss = _alignment(model)(model, batch, outputs, 0)
        encoders = _params(model, "encoders.signal.to_mu.weight", "encoders.other.to_mu.weight")
        assert _reachedBy(loss, encoders) == [True, True]

    @pytest.mark.parametrize(
        ("stop", "reached"), [("source", [False, True]), ("target", [True, False])]
    )
    def test_stop_gradient_cuts_the_first_or_the_second_space(
        self, stop: str, reached: list[bool]
    ) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        loss = _alignment(model, stop_gradient=stop)(model, batch, outputs, 0)
        encoders = _params(model, "encoders.signal.to_mu.weight", "encoders.other.to_mu.weight")
        assert _reachedBy(loss, encoders) == reached

    def test_stop_gradient_leaves_the_value_as_it_was(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        plain = _alignment(model)(model, batch, outputs, 0)
        for stop in ("source", "target"):
            assert torch.equal(
                _alignment(model, stop_gradient=stop)(model, batch, outputs, 0), plain
            )

    def test_a_batch_that_lacks_a_modality_of_a_pair_is_refused_by_name(self) -> None:
        model = _model()
        outputs = _outputs(model, _batch())
        with pytest.raises(ValueError) as caught:
            _alignment(model)(model, _batch(("signal",)), outputs, 0)
        message = str(caught.value)
        assert "needs paired batches" in message
        assert (
            "'other'" in message
            and "latent space 'z_other' of the pair [z_signal, z_other]" in message
        )
        assert "The batch has ['signal']" in message
        assert "'signal' (" not in message


class TestRequireModalitiesInBatch:
    def test_a_complete_batch_passes_and_an_empty_need_list_passes(self) -> None:
        requireModalitiesInBatch({"a": torch.zeros(1)}, [("a", "why")], "t")
        requireModalitiesInBatch({}, [], "t")

    def test_every_missing_modality_is_named_with_its_reason(self) -> None:
        with pytest.raises(ValueError) as caught:
            requireModalitiesInBatch(
                {"c": torch.zeros(1)}, [("a", "first reason"), ("b", "second"), ("c", "ok")], "t"
            )
        message = str(caught.value)
        assert message.startswith("Loss term 't' needs paired batches")
        assert "'a' (first reason); 'b' (second)" in message
        assert "skipped term would hide a loader" in message


class TestCrossReconstructionArguments:
    def test_the_pairs_are_ordered_and_the_two_orders_are_two_pairs(self) -> None:
        term = CrossReconstructionTerm(pairs=[["a", "b"], ["b", "a"]])
        assert term.pairs == (("a", "b"), ("b", "a"))
        assert term.stop_gradient == "none"

    def test_a_pair_listed_twice_in_the_same_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cross_reconstruction: pairs.*a second time"):
            CrossReconstructionTerm(pairs=[["a", "b"], ["a", "b"]])

    def test_a_bad_stop_gradient_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cross_reconstruction: stop_gradient must be one of"):
            CrossReconstructionTerm(pairs=[["a", "b"]], stop_gradient="both")


class TestCrossReconstructionValidateModel:
    def test_a_name_that_is_not_a_latent_space_is_refused(self) -> None:
        term = CrossReconstructionTerm(pairs=[["z_oops", "z_other"]])
        with pytest.raises(ValueError, match="names 'z_oops', which is not a latent space"):
            term.validateModel(_model())

    def test_two_spaces_of_different_widths_are_refused_with_both_widths(self) -> None:
        term = CrossReconstructionTerm(pairs=[["z_signal", "z_other"]])
        with pytest.raises(ValueError, match=r"latent widths 3 and 5.*roadmap D-16"):
            term.validateModel(_model({"signal": 3, "other": 5}))

    def test_a_target_that_no_decoder_can_decode_alone_is_refused_at_build_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _model()
        monkeypatch.setattr(model, "modalitiesDecodedFrom", lambda names: ())
        term = CrossReconstructionTerm(pairs=[["z_signal", "z_other"]])
        with pytest.raises(ValueError, match=r"pair \[z_signal, z_other\] cannot be decoded"):
            term.validateModel(model)

    def test_a_trainer_with_modality_dropout_is_refused(self) -> None:
        term = CrossReconstructionTerm(pairs=[["z_signal", "z_other"]])
        with pytest.raises(ValueError, match=r"modality dropout must be off.*modality_dropout_p=0"):
            term.bindTrainerSettings(_settings(modality_dropout_p=0.5))

    def test_a_term_that_was_never_bound_says_what_it_needs(self) -> None:
        model, batch = _model(), _batch()
        term = CrossReconstructionTerm(pairs=[["z_signal", "z_other"]])
        term.validateModel(model)
        with pytest.raises(RuntimeError, match="not bound to a trainer.*bindTrainerSettings"):
            term(model, batch, _outputs(model, batch), 0)


class TestCrossReconstructionForward:
    def test_it_scores_the_target_decoders_output_from_the_source_sample(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        decoded = model.decode({"z_other": outputs["latent_samples"]["z_signal"]})
        assert set(decoded) == {"other"}
        expected = computeTotalReconstructionLoss(decoded, batch)
        assert torch.equal(_cross(model)(model, batch, outputs, 0), expected)

    def test_it_is_not_the_self_reconstruction(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        own = computeTotalReconstructionLoss({"other": outputs["reconstructions"]["other"]}, batch)
        assert not torch.allclose(_cross(model)(model, batch, outputs, 0), own)

    def test_the_two_directions_are_summed(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        both = _cross(model, pairs=[["z_signal", "z_other"], ["z_other", "z_signal"]])
        forward = _cross(model, pairs=[["z_signal", "z_other"]])
        backward = _cross(model, pairs=[["z_other", "z_signal"]])
        assert torch.allclose(
            both(model, batch, outputs, 0),
            forward(model, batch, outputs, 0) + backward(model, batch, outputs, 0),
        )

    def test_each_target_is_scored_with_the_loss_weight_and_reduction_of_the_trainer(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        settings = _settings(
            reconstruction_loss_fn={"signal": F.l1_loss, "other": F.mse_loss},
            reconstruction_weights={"signal": 2.0, "other": 0.5},
            reconstruction_reduction={"signal": "sum", "other": "mean"},
        )
        samples = outputs["latent_samples"]
        for pair, target, source in (
            (["z_signal", "z_other"], "other", "z_signal"),
            (["z_other", "z_signal"], "signal", "z_other"),
        ):
            decoded = model.decode({f"z_{target}": samples[source]})
            expected = computeTotalReconstructionLoss(
                decoded,
                batch,
                weights=settings.reconstruction_weights,
                loss_fn=settings.reconstruction_loss_fn,
                reduction=settings.reconstruction_reduction,
            )
            term = _cross(model, settings, pairs=[pair])
            assert torch.equal(term(model, batch, outputs, 0), expected), target

    def test_the_weight_and_the_loss_of_a_modality_change_the_value_as_they_say(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)

        def value(**changes: Any) -> float:
            return float(_cross(model, _settings(**changes))(model, batch, outputs, 0).item())

        base = value()
        assert value(reconstruction_weights={"other": 3.0}) == pytest.approx(3.0 * base)
        assert value(reconstruction_loss_fn=F.l1_loss) != pytest.approx(base)
        assert value(reconstruction_reduction="sum") != pytest.approx(base)

    def test_a_batch_that_lacks_the_target_or_the_source_modality_is_refused_by_name(self) -> None:
        model = _model()
        outputs = _outputs(model, _batch())
        term = _cross(model)
        with pytest.raises(ValueError) as target_missing:
            term(model, _batch(("signal",)), outputs, 0)
        assert "needs paired batches" in str(target_missing.value)
        assert "'other' (it is what the target space 'z_other' decodes to)" in str(
            target_missing.value
        )
        with pytest.raises(ValueError) as source_missing:
            term(model, _batch(("other",)), outputs, 0)
        assert "'signal' (it feeds the source space 'z_signal' of [z_signal, z_other])" in str(
            source_missing.value
        )

    @pytest.mark.parametrize(
        ("stop", "decoder_reached", "encoder_reached"),
        [("none", True, True), ("source", True, False), ("target", False, True)],
    )
    def test_stop_gradient_cuts_the_source_or_the_target_side(
        self, stop: str, decoder_reached: bool, encoder_reached: bool
    ) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        loss = _cross(model, stop_gradient=stop)(model, batch, outputs, 0)
        decoder = _params(model, "decoders.other.project.weight", "decoders.other.hidden.weight")
        encoder = _params(model, "encoders.signal.to_mu.weight", "encoders.signal.hidden.weight")
        assert _reachedBy(loss, decoder) == [decoder_reached] * 2
        assert _reachedBy(loss, encoder) == [encoder_reached] * 2

    def test_stop_gradient_leaves_the_value_as_it_was(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        plain = _cross(model)(model, batch, outputs, 0)
        for stop in ("source", "target"):
            assert torch.equal(_cross(model, stop_gradient=stop)(model, batch, outputs, 0), plain)

    def test_the_requires_grad_flags_are_put_back_and_a_frozen_parameter_stays_frozen(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        model.get_parameter("decoders.other.project.bias").requires_grad_(False)
        before = {name: p.requires_grad for name, p in model.named_parameters()}
        _cross(model, stop_gradient="target")(model, batch, outputs, 0)
        assert {name: p.requires_grad for name, p in model.named_parameters()} == before
        assert before["decoders.other.project.bias"] is False

    def test_the_flags_are_put_back_when_the_decoding_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("decoder failed")

        monkeypatch.setattr(model, "decode", explode)
        before = {name: p.requires_grad for name, p in model.named_parameters()}
        with pytest.raises(RuntimeError, match="decoder failed"):
            _cross(model, stop_gradient="target")(model, batch, outputs, 0)
        assert {name: p.requires_grad for name, p in model.named_parameters()} == before

    def test_a_decoding_that_stops_the_gradient_does_not_touch_the_other_term_s_gradient(
        self,
    ) -> None:
        # The self reconstruction of the same step still trains the decoder as before.
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        cross = _cross(model, stop_gradient="target")(model, batch, outputs, 0)
        own = computeTotalReconstructionLoss(outputs["reconstructions"], batch)
        decoder = _params(model, "decoders.other.project.weight")
        assert _reachedBy(cross, decoder) == [False]
        assert _reachedBy(own, decoder) == [True]


def _trainer(terms: dict[str, Any], **kwargs: Any) -> Trainer:
    kwargs.setdefault("optimizer_kwargs", {"lr": 0.01})
    return Trainer(_model(), device="cpu", loss_terms=terms, **kwargs)


def _both() -> dict[str, Any]:
    return {
        "latent_alignment": LatentAlignmentTerm(
            pairs=[["z_signal", "z_other"]], distance="wasserstein2_diag"
        ),
        "cross_reconstruction": CrossReconstructionTerm(
            pairs=[["z_signal", "z_other"], ["z_other", "z_signal"]]
        ),
    }


class TestWithTheTrainer:
    def test_a_trainer_builds_with_both_terms_and_binds_them(self) -> None:
        trainer = _trainer(_both())
        cross = trainer.loss_terms["cross_reconstruction"]
        assert isinstance(cross, CrossReconstructionTerm)
        assert cross._settings is not None
        assert cross._settings.reconstruction_reduction == trainer.reconstruction_reduction

    def test_the_settings_a_term_is_bound_with_are_the_ones_the_trainer_was_given(self) -> None:
        trainer = _trainer(
            _both(),
            reconstruction_weights={"signal": 2.0, "other": 0.5},
            reconstruction_reduction="sum",
        )
        cross = trainer.loss_terms["cross_reconstruction"]
        assert isinstance(cross, CrossReconstructionTerm) and cross._settings is not None
        assert cross._settings.reconstruction_weights == {"signal": 2.0, "other": 0.5}
        assert cross._settings.reconstruction_reduction == "sum"

    @pytest.mark.parametrize("name", ["latent_alignment", "cross_reconstruction"])
    def test_a_trainer_with_modality_dropout_is_refused_when_it_is_built(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"modality dropout must be off.*modality_dropout_p=0"):
            _trainer({name: _both()[name]}, modality_dropout_p=0.2)

    def test_with_modality_dropout_off_the_trainer_builds(self) -> None:
        _trainer(_both(), modality_dropout_p=0.0)

    def test_a_trainer_whose_model_lacks_a_latent_space_of_a_pair_is_refused_when_built(
        self,
    ) -> None:
        terms = {"latent_alignment": LatentAlignmentTerm(pairs=[["z_a", "z_b"]], distance="cosine")}
        with pytest.raises(ValueError, match="latent_alignment: the pair .* not a latent space"):
            _trainer(terms)

    def _losses(self, trainer: Trainer, batch: dict[str, Any] | None = None) -> StepLosses:
        torch.manual_seed(3)
        return trainer.computeLosses(batch or _batch(), step=0, apply_dropout=False)

    def test_each_term_is_in_the_losses_and_in_the_metrics_and_adds_to_the_total(self) -> None:
        trainer = _trainer(_both())
        losses = self._losses(trainer)
        assert set(losses.terms) == {"latent_alignment", "cross_reconstruction"}
        assert all(value.item() > 0.0 for value in losses.terms.values())
        expected_total = losses.reconstruction + losses.regularization + sum(losses.terms.values())
        assert torch.allclose(losses.total, expected_total)
        metrics = losses.asMetrics()
        assert "loss/latent_alignment" in metrics and "loss/cross_reconstruction" in metrics

    def test_the_weight_of_an_entry_scales_its_term(self) -> None:
        plain = self._losses(_trainer(_both()))
        weighted = self._losses(_trainer(_both(), loss_term_weights={"latent_alignment": 0.25}))
        assert weighted.terms["latent_alignment"].item() == pytest.approx(
            0.25 * plain.terms["latent_alignment"].item()
        )
        assert weighted.terms["cross_reconstruction"].item() == pytest.approx(
            plain.terms["cross_reconstruction"].item()
        )

    def test_a_batch_that_lacks_a_modality_is_refused_in_training_and_in_validation(self) -> None:
        trainer = _trainer(_both())
        with pytest.raises(ValueError, match="needs paired batches"):
            self._losses(trainer, _batch(("signal",)))
        with pytest.raises(ValueError, match="needs paired batches"):
            trainer.evaluate([_batch(("signal",))])

    def test_a_few_steps_run_and_both_terms_are_in_the_history(self) -> None:
        trainer = _trainer(_both())
        trainer.fit([_batch(seed=0), _batch(seed=1)], num_epochs=2, val_dataloader=[_batch(seed=2)])
        for split in ("train", "val"):
            for name in ("latent_alignment", "cross_reconstruction"):
                assert f"{split}/loss/{name}" in trainer.history[-1]

    def test_the_alignment_term_lowers_the_distance_it_is_made_of(self) -> None:
        batches = [_batch(seed=s) for s in range(3)]
        trainer = _trainer(
            {
                "latent_alignment": LatentAlignmentTerm(
                    pairs=[["z_signal", "z_other"]], distance="mean_mse"
                )
            },
            optimizer_kwargs={"lr": 0.02},
        )
        trainer.fit(batches, num_epochs=30, val_dataloader=[_batch(seed=9)])
        first, last = trainer.history[0], trainer.history[-1]
        assert last["val/loss/latent_alignment"] < 0.5 * first["val/loss/latent_alignment"]


def _withIds(
    batch: dict[str, Any], signal: torch.Tensor | None, other: torch.Tensor | None
) -> dict[str, Any]:
    """`batch` with the `sample_id.<modality>` entries the loader of ADR 0058 would add."""
    with_ids = dict(batch)
    if signal is not None:
        with_ids["sample_id.signal"] = signal
    if other is not None:
        with_ids["sample_id.other"] = other
    return with_ids


class TestSampleIds:
    """The pair terms compare the `sample_id.<modality>` entries of a batch (ADR 0058)."""

    @pytest.mark.parametrize("name", ["latent_alignment", "cross_reconstruction"])
    def test_a_check_is_on_by_default_and_can_be_turned_off(self, name: str) -> None:
        pairs = [["z_signal", "z_other"]]
        term = _both()[name]
        assert term.check_sample_ids
        kwargs: dict[str, Any] = {"pairs": pairs, "check_sample_ids": False}
        if name == "latent_alignment":
            kwargs["distance"] = "mean_mse"
        assert not getLossTermClass(name)(**kwargs).check_sample_ids

    def test_matching_ids_change_nothing_in_either_term(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        ids = torch.arange(BATCH)
        marked = _withIds(batch, ids, ids.clone())
        for term in (_alignment(model), _cross(model)):
            assert torch.equal(term(model, batch, outputs, 0), term(model, marked, outputs, 0))

    @pytest.mark.parametrize("which", ["alignment", "cross"])
    def test_ids_that_differ_are_refused_by_the_term_that_found_them(self, which: str) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        other = torch.arange(BATCH).flip(0)
        term = _alignment(model) if which == "alignment" else _cross(model)
        name = "latent_alignment" if which == "alignment" else "cross_reconstruction"
        with pytest.raises(ValueError, match=rf"Loss term '{name}': row i of 'signal' and row i"):
            term(model, _withIds(batch, torch.arange(BATCH), other), outputs, 0)

    def test_the_message_names_the_first_rows_and_what_to_look_at(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        shifted = torch.arange(BATCH) + 1
        with pytest.raises(ValueError, match=r"row 0 \(signal 0, other 1\).*pairs its sources"):
            _alignment(model)(model, _withIds(batch, torch.arange(BATCH), shifted), outputs, 0)

    @pytest.mark.parametrize("which", ["alignment", "cross"])
    def test_ids_for_one_modality_only_are_refused(self, which: str) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        term = _alignment(model) if which == "alignment" else _cross(model)
        with pytest.raises(ValueError, match=r"ids for \['signal'\] but none for \['other'\]"):
            term(model, _withIds(batch, torch.arange(BATCH), None), outputs, 0)

    def test_a_term_with_the_check_off_scores_a_batch_whose_ids_differ(self) -> None:
        model, batch = _model(), _batch()
        outputs = _outputs(model, batch)
        term = LatentAlignmentTerm(
            pairs=[["z_signal", "z_other"]], distance="mean_mse", check_sample_ids=False
        )
        term.validateModel(model)
        marked = _withIds(batch, torch.arange(BATCH), torch.arange(BATCH).flip(0))
        assert torch.equal(term(model, marked, outputs, 0), term(model, batch, outputs, 0))

    def test_a_batch_that_lacks_a_modality_is_refused_for_that_before_the_ids_are_read(
        self,
    ) -> None:
        model, batch = _model(), _batch(("signal",))
        outputs = _outputs(model, batch)
        with pytest.raises(ValueError, match="needs paired batches"):
            _alignment(model)(model, _withIds(batch, torch.arange(BATCH), None), outputs, 0)

    def test_the_trainer_stops_on_the_first_step_of_a_loader_that_pairs_wrongly(self) -> None:
        trainer = _trainer(_both())
        wrong = _withIds(_batch(), torch.arange(BATCH), torch.arange(BATCH).flip(0))
        with pytest.raises(ValueError, match="would train on strangers"):
            trainer.fit([wrong], num_epochs=1)
        with pytest.raises(ValueError, match="would train on strangers"):
            trainer.evaluate([wrong])

    def test_the_trainer_trains_on_batches_that_carry_matching_ids(self) -> None:
        trainer = _trainer(_both())
        ids = torch.arange(BATCH)
        batches = [_withIds(_batch(seed=s), ids, ids) for s in range(2)]
        trainer.fit(batches, num_epochs=2, val_dataloader=[_withIds(_batch(seed=5), ids, ids)])
        assert trainer.history[-1]["train/loss/latent_alignment"] >= 0.0

    def test_ids_are_not_taken_for_modalities(self) -> None:
        model = _model()
        marked = _withIds(_batch(), torch.arange(BATCH), torch.arange(BATCH))
        assert sorted(model.selectEncoderInputs(marked)) == ["other", "signal"]


THREE = ("signal", "other", "third")


def _threeModels(seed: int = 0) -> GlobalVae:
    """Three modalities with a latent space each: `z_signal`, `z_other`, `z_third`."""
    return _model({name: LATENT for name in THREE}, seed)


class TestResolvePairs:
    """Anchor mode builds the pairs of a term from one space and the spaces aligned to it."""

    def test_pairs_alone_are_parsed_as_before(self) -> None:
        got = resolvePairs([["a", "b"]], None, None, "t", allow_reversed_duplicates=False)
        assert got == (("a", "b"),)

    def test_each_other_space_makes_a_pair_with_the_anchor_second(self) -> None:
        got = resolvePairs(None, "z_a", ["z_b", "z_c"], "t", allow_reversed_duplicates=False)
        assert got == (("z_b", "z_a"), ("z_c", "z_a"))

    def test_the_directions_choose_which_way_the_pairs_run(self) -> None:
        def pairsFor(directions: str) -> tuple[tuple[str, str], ...]:
            return resolvePairs(None, "a", ["b", "c"], "t", True, directions)

        assert pairsFor("to_anchor") == (("b", "a"), ("c", "a"))
        assert pairsFor("from_anchor") == (("a", "b"), ("a", "c"))
        assert pairsFor("both") == (("b", "a"), ("a", "b"), ("c", "a"), ("a", "c"))
        assert set(ANCHOR_DIRECTIONS) == {"to_anchor", "from_anchor", "both"}

    def test_a_list_from_a_yaml_config_is_accepted(self) -> None:
        others = OmegaConf.create(["z_b", "z_c"])
        assert resolvePairs(None, "z_a", others, "t", False) == (("z_b", "z_a"), ("z_c", "z_a"))

    def test_neither_pairs_nor_an_anchor_is_refused_and_says_what_to_give(self) -> None:
        with pytest.raises(ValueError, match=r"give `pairs`.*or `anchor` with `others`"):
            resolvePairs(None, None, None, "t", False)

    def test_pairs_with_an_anchor_are_refused(self) -> None:
        with pytest.raises(ValueError, match="`pairs` or `anchor` with `others`, not both"):
            resolvePairs([["a", "b"]], "a", ["b"], "t", False)
        with pytest.raises(ValueError, match="not both"):
            resolvePairs([["a", "b"]], None, ["b"], "t", False)

    @pytest.mark.parametrize(("anchor", "others"), [("a", None), (None, ["b"])])
    def test_an_anchor_without_others_and_others_without_an_anchor_are_refused(
        self, anchor: str | None, others: list[str] | None
    ) -> None:
        with pytest.raises(ValueError, match="`anchor` and `others` go together"):
            resolvePairs(None, anchor, others, "t", False)

    @pytest.mark.parametrize("others", [[], "z_b", 3, [""], [3]])
    def test_others_must_be_a_non_empty_list_of_names(self, others: object) -> None:
        with pytest.raises(ValueError, match="`others` must"):
            resolvePairs(None, "a", others, "t", False)

    def test_the_anchor_among_the_others_is_refused(self) -> None:
        with pytest.raises(ValueError, match="anchor 'a' is also listed in `others`"):
            resolvePairs(None, "a", ["b", "a"], "t", False)

    def test_a_space_listed_twice_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"lists \['b'\] more than once"):
            resolvePairs(None, "a", ["b", "c", "b"], "t", False)

    @pytest.mark.parametrize("anchor", ["", 3])
    def test_the_anchor_must_be_a_name(self, anchor: object) -> None:
        with pytest.raises(ValueError, match="`anchor` must be a latent space name"):
            resolvePairs(None, anchor, ["b"], "t", False)

    def test_directions_must_be_one_of_the_three(self) -> None:
        with pytest.raises(ValueError, match=r"directions must be one of \['to_anchor'"):
            resolvePairs(None, "a", ["b"], "t", False, "sideways")


class TestAnchorMode:
    """Three spaces aligned to one (roadmap P6-8, ADR 0059)."""

    def _outputs(self) -> tuple[GlobalVae, dict[str, Any], dict[str, Any]]:
        model, batch = _threeModels(), _batch(THREE)
        return model, batch, _outputs(model, batch)

    def test_the_alignment_term_builds_the_pairs_and_remembers_the_anchor(self) -> None:
        term = LatentAlignmentTerm(
            distance="mean_mse", anchor="z_signal", others=["z_other", "z_third"]
        )
        assert term.pairs == (("z_other", "z_signal"), ("z_third", "z_signal"))
        assert term.anchor == "z_signal"

    def test_the_alignment_is_the_sum_of_the_distances_to_the_anchor(self) -> None:
        model, batch, outputs = self._outputs()
        anchored = LatentAlignmentTerm(
            distance="mean_mse", anchor="z_signal", others=["z_other", "z_third"]
        )
        explicit = LatentAlignmentTerm(
            distance="mean_mse", pairs=[["z_other", "z_signal"], ["z_third", "z_signal"]]
        )
        for term in (anchored, explicit):
            term.validateModel(model)
            term.bindTrainerSettings(_settings())
        value = anchored(model, batch, outputs, 0)
        assert torch.equal(value, explicit(model, batch, outputs, 0))
        mu = {name: outputs["latent_params"][f"z_{name}"][0] for name in THREE}
        by_hand = sum(((mu[m] - mu["signal"]) ** 2).mean() for m in ("other", "third"))
        assert value.item() == pytest.approx(float(by_hand.detach()) * LATENT, rel=1e-5)

    def test_it_is_not_every_pair_of_the_three_spaces(self) -> None:
        model, batch, outputs = self._outputs()
        anchored = LatentAlignmentTerm(
            distance="mean_mse", anchor="z_signal", others=["z_other", "z_third"]
        )
        every_pair = LatentAlignmentTerm(
            distance="mean_mse",
            pairs=[["z_signal", "z_other"], ["z_signal", "z_third"], ["z_other", "z_third"]],
        )
        for term in (anchored, every_pair):
            term.validateModel(model)
        assert len(anchored.pairs) == 2 and len(every_pair.pairs) == 3
        assert anchored(model, batch, outputs, 0) != every_pair(model, batch, outputs, 0)

    def test_the_stop_gradient_that_keeps_the_anchor_in_place_is_target(self) -> None:
        model, batch, outputs = self._outputs()
        term = LatentAlignmentTerm(
            distance="mean_mse",
            anchor="z_signal",
            others=["z_other", "z_third"],
            stop_gradient="target",
        )
        term.validateModel(model)
        paths = [f"encoders.{m}.to_mu.weight" for m in THREE]
        reached = _reachedBy(term(model, batch, outputs, 0), _params(model, *paths))
        assert reached == [False, True, True]

    def test_the_batch_must_carry_every_modality_of_every_space_in_it(self) -> None:
        model = _threeModels()
        batch = _batch(("signal", "other"))
        outputs = _outputs(model, batch)
        term = LatentAlignmentTerm(
            distance="mean_mse", anchor="z_signal", others=["z_other", "z_third"]
        )
        term.validateModel(model)
        with pytest.raises(ValueError, match=r"needs paired batches.*'third'"):
            term(model, batch, outputs, 0)

    def test_a_space_the_model_lacks_is_refused_when_the_term_is_built_on_it(self) -> None:
        term = LatentAlignmentTerm(distance="mean_mse", anchor="z_signal", others=["z_nope"])
        with pytest.raises(ValueError, match="'z_nope', which is not a latent space"):
            term.validateModel(_threeModels())

    def test_cross_reconstruction_runs_both_ways_by_default(self) -> None:
        term = CrossReconstructionTerm(anchor="z_signal", others=["z_other", "z_third"])
        assert term.pairs == (
            ("z_other", "z_signal"),
            ("z_signal", "z_other"),
            ("z_third", "z_signal"),
            ("z_signal", "z_third"),
        )

    def test_cross_reconstruction_in_anchor_mode_is_the_sum_of_its_explicit_pairs(self) -> None:
        model, batch, outputs = self._outputs()
        for directions in ("both", "to_anchor", "from_anchor"):
            anchored = CrossReconstructionTerm(
                anchor="z_signal", others=["z_other", "z_third"], directions=directions
            )
            explicit = CrossReconstructionTerm(pairs=anchored.pairs)
            for term in (anchored, explicit):
                term.validateModel(model)
                term.bindTrainerSettings(_settings())
            assert torch.equal(
                anchored(model, batch, outputs, 0), explicit(model, batch, outputs, 0)
            ), directions

    def test_the_anchor_decodes_what_the_other_spaces_say_when_the_pairs_run_to_it(self) -> None:
        model, batch, outputs = self._outputs()
        term = CrossReconstructionTerm(
            anchor="z_signal", others=["z_other", "z_third"], directions="to_anchor"
        )
        term.validateModel(model)
        term.bindTrainerSettings(_settings())
        samples = outputs["latent_samples"]
        expected = torch.stack(
            [
                F.mse_loss(model.decode({"z_signal": samples[f"z_{m}"]})["signal"], batch["signal"])
                for m in ("other", "third")
            ]
        ).sum()
        assert term(model, batch, outputs, 0).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_a_trainer_trains_three_spaces_aligned_to_one(self) -> None:
        terms = {
            "latent_alignment": LatentAlignmentTerm(
                distance="mean_mse", anchor="z_signal", others=["z_other", "z_third"]
            ),
            "cross_reconstruction": CrossReconstructionTerm(
                anchor="z_signal", others=["z_other", "z_third"]
            ),
        }
        trainer = Trainer(
            _threeModels(), device="cpu", loss_terms=terms, optimizer_kwargs={"lr": 0.02}
        )
        batches = [_batch(THREE, seed=s) for s in range(3)]
        trainer.fit(batches, num_epochs=25, val_dataloader=[_batch(THREE, seed=9)])
        first, last = trainer.history[0], trainer.history[-1]
        assert last["val/loss/latent_alignment"] < 0.5 * first["val/loss/latent_alignment"]
        assert "train/loss/cross_reconstruction" in last


class TestAnchorModeFromTheConfig:
    """The kwargs of a `training.loss_terms` entry in YAML build the same terms."""

    def test_the_pair_align_file_turned_to_anchor_mode_builds_the_terms(self) -> None:
        config = loadExperimentConfig(
            "experiment/pair_align",
            overrides=[
                "data.loader_factory=unused:unused",
                "data.train_path=unused",
                "~training.loss_terms.0.kwargs.pairs",
                "+training.loss_terms.0.kwargs.anchor=z_signal",
                "+training.loss_terms.0.kwargs.others=[z_image]",
                "~training.loss_terms.1.kwargs.pairs",
                "+training.loss_terms.1.kwargs.anchor=z_signal",
                "+training.loss_terms.1.kwargs.others=[z_image]",
            ],
        )
        terms, weights = buildLossTerms(config.training)
        alignment, cross = terms["latent_alignment"], terms["cross_reconstruction"]
        assert isinstance(alignment, LatentAlignmentTerm)
        assert isinstance(cross, CrossReconstructionTerm)
        assert alignment.pairs == (("z_image", "z_signal"),)
        assert cross.pairs == (("z_image", "z_signal"), ("z_signal", "z_image"))
        assert set(weights) == {"latent_alignment", "cross_reconstruction"}


class TestOnUnpaired:
    """`on_unpaired: skip` lets a stream mix paired batches with unpaired ones (ADR 0061)."""

    @pytest.mark.parametrize("make", [_alignment, _cross])
    def test_the_default_refuses_an_unpaired_batch_and_names_the_way_out(self, make: Any) -> None:
        model = _model()
        batch = _batch(("signal",))
        with pytest.raises(ValueError, match="needs paired batches") as raised:
            make(model)(model, batch, _outputs(model, batch), 0)
        assert "on_unpaired: skip" in str(raised.value)

    @pytest.mark.parametrize("make", [_alignment, _cross])
    def test_skip_returns_a_scalar_zero_and_counts_the_batch(self, make: Any) -> None:
        model = _model()
        term = make(model, on_unpaired="skip")
        batch = _batch(("signal",))
        value = term(model, batch, _outputs(model, batch), 0)
        assert value.dim() == 0 and float(value) == 0.0
        assert value.device == next(model.parameters()).device
        assert (term.num_skipped, term.num_applied) == (1, 0)

    @pytest.mark.parametrize("make", [_alignment, _cross])
    def test_skip_still_computes_a_paired_batch_as_before(self, make: Any) -> None:
        model = _model()
        batch = _batch()
        outputs = _outputs(model, batch)
        plain = make(model)(model, batch, outputs, 0)
        skipping = make(model, on_unpaired="skip")
        value = skipping(model, batch, outputs, 0)
        assert torch.equal(plain, value) and float(value.detach()) > 0.0
        assert (skipping.num_skipped, skipping.num_applied) == (0, 1)

    @pytest.mark.parametrize("make", [_alignment, _cross])
    def test_a_batch_with_only_one_modality_of_the_pair_is_skipped_either_way_round(
        self, make: Any
    ) -> None:
        model = _model()
        term = make(model, on_unpaired="skip")
        for modality in ("signal", "other"):
            batch = _batch((modality,))
            assert float(term(model, batch, _outputs(model, batch), 0)) == 0.0
        assert term.num_skipped == 2

    @pytest.mark.parametrize("make", [_alignment, _cross])
    @pytest.mark.parametrize("value", ["drop", "", None, True])
    def test_an_unknown_choice_is_refused_by_name(self, make: Any, value: Any) -> None:
        with pytest.raises(ValueError, match=r"on_unpaired must be one of \['raise', 'skip'\]"):
            make(on_unpaired=value)

    def test_a_mixed_stream_trains_every_batch_and_the_pair_term_sees_only_the_paired(self) -> None:
        terms = {
            "latent_alignment": LatentAlignmentTerm(
                pairs=[["z_signal", "z_other"]], distance="wasserstein2_diag", on_unpaired="skip"
            ),
            "cross_reconstruction": CrossReconstructionTerm(
                pairs=[["z_signal", "z_other"]], on_unpaired="skip"
            ),
        }
        trainer = _trainer(terms)
        stream = [
            _batch(seed=0),
            _batch(("signal",), seed=1),
            _batch(("other",), seed=2),
            _batch(seed=3),
        ]
        trainer.fit(stream, num_epochs=2)
        for name in terms:
            term = trainer.loss_terms[name]
            assert isinstance(term, (LatentAlignmentTerm, CrossReconstructionTerm))
            assert (term.num_applied, term.num_skipped) == (4, 4)
        assert trainer.global_step == 8

    def test_an_unpaired_batch_alone_still_trains_the_reconstruction_of_its_modality(self) -> None:
        trainer = _trainer({"latent_alignment": _alignment(on_unpaired="skip")})
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.fit([_batch(("signal",))], num_epochs=3)
        after = trainer.model.state_dict()
        moved = sorted(k for k in before if not torch.equal(before[k], after[k]))
        assert any(k.startswith("encoders.signal") for k in moved)
        assert any(k.startswith("decoders.signal") for k in moved)
        assert not any(k.startswith(("encoders.other", "decoders.other")) for k in moved)

    def test_the_config_passes_the_choice_through(self) -> None:
        config = loadExperimentConfig(
            "experiment/pair_align",
            overrides=[
                "data.loader_factory=unused:unused",
                "data.train_path=unused",
                "+training.loss_terms.0.kwargs.on_unpaired=skip",
                "+training.loss_terms.1.kwargs.on_unpaired=skip",
            ],
        )
        terms, _ = buildLossTerms(config.training)
        alignment, cross = terms["latent_alignment"], terms["cross_reconstruction"]
        assert isinstance(alignment, LatentAlignmentTerm)
        assert isinstance(cross, CrossReconstructionTerm)
        assert (alignment.on_unpaired, cross.on_unpaired) == ("skip", "skip")
