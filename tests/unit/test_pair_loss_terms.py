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
    STOP_GRADIENT_CHOICES,
    checkStopGradient,
    parsePairs,
    requireModalitiesInBatch,
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
