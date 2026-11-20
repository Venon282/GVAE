"""Adapters for the alignment term: spaces that keep their own widths (P6-8, ADR 0060).

What is pinned: how the `adapters` argument is read and refused, the heads that are built (width,
device, the state a checkpoint holds), the term that reads a space through its adapter (the value,
the gradient with and without `stop_gradient`, the widths it accepts and the message when they are
still unequal), the trainer that holds them (optimizer, checkpoint, resume), and the case the
extension is for: both VAEs frozen and only the adapter trained.
"""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from omegaconf import OmegaConf

from global_vae.heads.linear import LinearHead
from global_vae.losses.terms.adapters import (
    DEFAULT_ADAPTER_HEAD,
    AdapterSpec,
    applyAdapter,
    buildAdapter,
    buildAdapters,
    loadAdapters,
    parseAdapterSpecs,
    parseAdapterText,
)
from global_vae.losses.terms.base import TrainerSettings
from global_vae.losses.terms.cross_reconstruction import CrossReconstructionTerm
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks.freeze_modules import FreezeModules
from global_vae.training.checkpoint import readLossTermState
from global_vae.training.trainer import Trainer
from tests.integration._latent_alignment_fixtures import buildTripleBatches, buildTripleModel

WIDTHS = {"signal": 3, "image": 3, "text": 5}


def _model(widths: dict[str, int] | None = None) -> GlobalVae:
    return buildTripleModel(widths or WIDTHS)


def _settings() -> TrainerSettings:
    return TrainerSettings(
        reconstruction_loss_fn=F.mse_loss,
        reconstruction_weights=1.0,
        reconstruction_reduction="mean",
        modality_dropout_p=0.0,
    )


def _term(model: GlobalVae | None = None, **kwargs: Any) -> LatentAlignmentTerm:
    kwargs.setdefault("pairs", [["z_signal", "z_text"]])
    kwargs.setdefault("distance", "mean_mse")
    kwargs.setdefault("adapters", {"z_text": 3})
    term = LatentAlignmentTerm(**kwargs)
    term.validateModel(model or _model())
    term.bindTrainerSettings(_settings())
    return term


def _outputs(model: GlobalVae, batch: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(1)
    return dict(model(model.selectEncoderInputs(batch)))


def _linear(term: LatentAlignmentTerm, space: str = "z_text") -> LinearHead:
    """The adapter of a space of a term, as the linear head it is."""
    return cast(LinearHead, term.adapters[space])


def _termOf(trainer: Trainer) -> LatentAlignmentTerm:
    return cast(LatentAlignmentTerm, trainer.loss_terms["latent_alignment"])


def _batch() -> dict[str, Any]:
    return buildTripleBatches(1, seed=0)[0]


def _trainer(model: GlobalVae | None = None, **term_kwargs: Any) -> Trainer:
    term = LatentAlignmentTerm(
        **{
            "pairs": [["z_signal", "z_text"]],
            "distance": "mean_mse",
            "adapters": {"z_text": 3},
            **term_kwargs,
        }
    )
    return Trainer(
        model or _model(),
        device="cpu",
        loss_terms={"latent_alignment": term},
        optimizer_kwargs={"lr": 0.01},
    )


class TestParseAdapterSpecs:
    def test_nothing_is_no_adapter(self) -> None:
        assert parseAdapterSpecs(None, "t") == {}
        assert parseAdapterSpecs({}, "t") == {}

    def test_a_width_is_a_linear_head(self) -> None:
        assert parseAdapterSpecs({"z_a": 7}, "t") == {"z_a": AdapterSpec("linear", 7)}

    def test_a_mapping_names_the_head_and_its_arguments(self) -> None:
        spec = parseAdapterSpecs({"z_a": {"name": "linear", "out_dim": 4, "kwargs": {}}}, "t")
        assert spec["z_a"] == AdapterSpec(DEFAULT_ADAPTER_HEAD, 4, {})

    def test_the_head_defaults_to_linear_and_an_omegaconf_mapping_is_read(self) -> None:
        config = OmegaConf.create({"z_a": {"out_dim": 4}, "z_b": 6})
        specs = parseAdapterSpecs(config, "t")
        assert specs["z_a"].head == "linear" and specs["z_b"].out_dim == 6

    @pytest.mark.parametrize(
        ("adapters", "message"),
        [
            ([("z_a", 3)], "`adapters` must be a mapping"),
            ({"z_a": "3"}, "adapter of 'z_a' must be a width or a mapping"),
            ({"z_a": True}, "adapter of 'z_a' must be a width or a mapping"),
            ({"z_a": 0}, "positive `out_dim`"),
            ({"z_a": -2}, "positive `out_dim`"),
            ({"z_a": {"name": "linear"}}, "`out_dim` required"),
            (
                {"z_a": {"out_dim": 3, "width": 3}},
                r"takes the keys \['name', 'out_dim', 'kwargs'\]",
            ),
            ({"z_a": {"out_dim": 2.5}}, "positive `out_dim`"),
            ({"z_a": {"name": "", "out_dim": 3}}, "non-empty string"),
            ({"z.a": 3}, "without a dot"),
            ({"": 3}, "without a dot"),
        ],
    )
    def test_what_is_malformed_is_refused_by_name(self, adapters: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parseAdapterSpecs(adapters, "latent_alignment")

    def test_the_text_form_is_what_the_command_line_reads(self) -> None:
        assert parseAdapterText("z_a=16") == ("z_a", AdapterSpec("linear", 16))
        assert parseAdapterText("z_a=linear:16") == ("z_a", AdapterSpec("linear", 16))
        assert AdapterSpec("linear", 16).toText() == "16"
        assert AdapterSpec("other", 16).toText() == "other:16"

    @pytest.mark.parametrize("text", ["z_a", "=16", "z_a=", "z_a=x", "z_a=linear:", "z_a=1.5"])
    def test_a_malformed_text_is_refused_with_the_forms_it_reads(self, text: str) -> None:
        with pytest.raises(ValueError, match="SPACE=WIDTH or SPACE=HEAD:WIDTH"):
            parseAdapterText(text)

    def test_a_width_below_one_in_the_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive `out_dim`"):
            parseAdapterText("z_a=0")


class TestBuildAdapters:
    def test_the_head_reads_the_width_of_the_space(self) -> None:
        head = buildAdapter(_model(), "z_text", AdapterSpec("linear", 3), "t")
        assert isinstance(head, LinearHead)
        assert (head.in_dim, head.out_dim) == (5, 3)

    def test_it_returns_a_module_dict_keyed_by_space(self) -> None:
        adapters = buildAdapters(_model(), {"z_text": AdapterSpec("linear", 3)}, "t")
        assert list(adapters) == ["z_text"]

    def test_a_space_the_model_lacks_is_refused_with_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match=r"'z_nope', which is not a latent space.*z_image"):
            buildAdapter(_model(), "z_nope", AdapterSpec("linear", 3), "t")

    def test_an_unknown_head_lists_the_registered_ones(self) -> None:
        with pytest.raises(KeyError, match="Unknown latent head 'nope'.*linear"):
            buildAdapter(_model(), "z_text", AdapterSpec("nope", 3), "t")

    def test_the_identity_head_is_refused_when_the_widths_differ_and_the_message_says_why(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=r"adapter of 'z_text' \(identity, 5 -> 3\).*linear"):
            buildAdapter(_model(), "z_text", AdapterSpec("identity", 3), "t")

    def test_the_identity_head_is_an_adapter_when_the_widths_agree(self) -> None:
        head = buildAdapter(_model(), "z_signal", AdapterSpec("identity", 3), "t")
        mu, logvar = torch.randn(4, 3), torch.randn(4, 3)
        out_mu, out_logvar = applyAdapter(head, (mu, logvar))
        assert torch.equal(out_mu, mu) and torch.equal(out_logvar, logvar)

    def test_arguments_the_head_does_not_take_are_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"adapter of 'z_text' .*could not be built"):
            buildAdapter(_model(), "z_text", AdapterSpec("linear", 3, {"bias": False}), "t")

    def test_without_an_adapter_a_posterior_is_returned_as_it_is(self) -> None:
        posterior = (torch.randn(2, 3), torch.randn(2, 3))
        assert applyAdapter(None, posterior) is posterior


class TestLoadAdapters:
    def test_the_state_of_a_term_is_loaded_into_heads_in_evaluation_mode(self) -> None:
        trained = _term()
        torch.nn.init.constant_(_linear(trained).to_mu.weight, 0.5)
        state = trained.state_dict()
        heads = loadAdapters(_model(), {"z_text": AdapterSpec("linear", 3)}, state, "t")
        assert torch.equal(
            cast(LinearHead, heads["z_text"]).to_mu.weight, _linear(trained).to_mu.weight
        )
        assert not heads["z_text"].training

    def test_a_state_that_does_not_fit_the_head_says_what_to_give(self) -> None:
        state = _term().state_dict()
        with pytest.raises(ValueError, match=r"adapter of 'z_text' does not fit a 'linear' head"):
            loadAdapters(_model(), {"z_text": AdapterSpec("linear", 4)}, state, "t")

    def test_a_state_without_the_adapter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            loadAdapters(_model(), {"z_text": AdapterSpec("linear", 3)}, {}, "t")


class TestTheTerm:
    def test_the_adapter_is_built_when_the_term_is_checked_against_the_model(self) -> None:
        term = LatentAlignmentTerm(
            pairs=[["z_signal", "z_text"]], distance="mean_mse", adapters={"z_text": 3}
        )
        assert len(term.adapters) == 0 and list(term.parameters()) == []
        term.validateModel(_model())
        assert list(term.adapters) == ["z_text"]
        assert sorted(term.state_dict()) == [
            "adapters.z_text.to_logvar.bias",
            "adapters.z_text.to_logvar.weight",
            "adapters.z_text.to_mu.bias",
            "adapters.z_text.to_mu.weight",
        ]

    def test_checking_twice_keeps_the_same_adapter(self) -> None:
        model = _model()
        term = _term(model)
        before = term.adapters["z_text"]
        term.validateModel(model)
        assert term.adapters["z_text"] is before

    def test_the_value_is_the_distance_between_the_posteriors_as_the_adapter_gives_them(
        self,
    ) -> None:
        model, batch = _model(), _batch()
        term = _term(model)
        outputs = _outputs(model, batch)
        params = outputs["latent_params"]
        adapted = term.adapters["z_text"](*params["z_text"])
        expected = term.distance(params["z_signal"], adapted).mean()
        assert torch.allclose(term(model, batch, outputs, 0), expected)

    def test_a_space_without_an_adapter_is_read_as_it_is(self) -> None:
        model, batch = _model(), _batch()
        term = LatentAlignmentTerm(
            pairs=[["z_signal", "z_image"], ["z_signal", "z_text"]],
            distance="mean_mse",
            adapters={"z_text": 3},
        )
        term.validateModel(model)
        outputs = _outputs(model, batch)
        params = outputs["latent_params"]
        expected = (
            term.distance(params["z_signal"], params["z_image"]).mean()
            + term.distance(params["z_signal"], term.adapters["z_text"](*params["z_text"])).mean()
        )
        assert torch.allclose(term(model, batch, outputs, 0), expected)

    def test_both_sides_can_be_adapted_to_a_width_of_their_own_choosing(self) -> None:
        model = _model({"signal": 4, "image": 3, "text": 5})
        term = _term(model, pairs=[["z_signal", "z_text"]], adapters={"z_signal": 6, "z_text": 6})
        value = term(model, _batch(), _outputs(model, _batch()), 0)
        assert value.dim() == 0 and torch.isfinite(value)

    def test_the_gradient_reaches_the_adapter_and_the_encoders_of_both_spaces(self) -> None:
        model, batch = _model(), _batch()
        term = _term(model)
        value = term(model, batch, _outputs(model, batch), 0)
        value.backward()
        adapter = _linear(term)
        assert adapter.to_mu.weight.grad is not None and adapter.to_mu.weight.grad.abs().sum() > 0
        for name in ("signal", "text"):
            grads = [p.grad for p in model.encoders[name].parameters() if p.grad is not None]
            assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_stop_gradient_keeps_a_space_in_place_and_the_adapter_still_learns(self) -> None:
        model, batch = _model(), _batch()
        term = _term(model, stop_gradient="target")  # z_text is the second space
        value = term(model, batch, _outputs(model, batch), 0)
        value.backward()
        assert all(p.grad is None for p in model.encoders["text"].parameters())
        assert any(p.grad is not None for p in model.encoders["signal"].parameters())
        assert _linear(term).to_mu.weight.grad is not None

    def test_the_adapter_does_not_change_what_the_model_computes(self) -> None:
        model, batch = _model(), _batch()
        before = _outputs(model, batch)
        term = _term(model)
        term(model, batch, before, 0)
        after = _outputs(model, batch)
        for space in before["latent_params"]:
            assert torch.equal(before["latent_params"][space][0], after["latent_params"][space][0])

    def test_unequal_widths_without_adapters_are_refused_with_the_way_out(self) -> None:
        with pytest.raises(ValueError, match=r"widths 3 and 5.*roadmap D-16.*give `adapters`"):
            _term(adapters=None)

    def test_adapters_that_leave_the_widths_unequal_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"widths 3 and 4.*adapters leave them at different"):
            _term(adapters={"z_text": 4})

    def test_an_adapter_of_a_space_that_is_in_no_pair_is_refused_when_the_term_is_built(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=r"names \['z_image'\], which is in no pair"):
            LatentAlignmentTerm(
                pairs=[["z_signal", "z_text"]], distance="mean_mse", adapters={"z_image": 3}
            )

    def test_an_adapter_of_a_space_the_model_lacks_is_refused_when_it_is_checked(self) -> None:
        term = LatentAlignmentTerm(
            pairs=[["z_signal", "z_nope"]], distance="mean_mse", adapters={"z_nope": 3}
        )
        with pytest.raises(ValueError, match=r"adapter names 'z_nope'.*latent spaces"):
            term.validateModel(_model())

    def test_a_term_whose_adapters_were_never_built_says_so_when_it_runs(self) -> None:
        model, batch = _model(), _batch()
        term = LatentAlignmentTerm(
            pairs=[["z_signal", "z_text"]], distance="mean_mse", adapters={"z_text": 3}
        )
        with pytest.raises(RuntimeError, match="adapters that are not built"):
            term(model, batch, _outputs(model, batch), 0)

    def test_equal_widths_need_no_adapter_and_an_adapter_is_still_allowed(self) -> None:
        model = _model({"signal": 3, "image": 3, "text": 3})
        plain = _term(model, pairs=[["z_signal", "z_image"]], adapters=None)
        adapted = _term(model, pairs=[["z_signal", "z_image"]], adapters={"z_image": 3})
        assert list(plain.parameters()) == [] and len(list(adapted.parameters())) == 4

    def test_the_cross_reconstruction_still_refuses_unequal_widths_and_says_why(self) -> None:
        term = CrossReconstructionTerm(pairs=[["z_signal", "z_text"], ["z_text", "z_signal"]])
        with pytest.raises(ValueError, match=r"widths 3 and 5.*no adapter of `latent_alignment`"):
            term.validateModel(_model())

    def test_anchor_mode_shares_one_adapter_across_the_pairs_that_name_its_space(self) -> None:
        model = _model({"signal": 4, "image": 3, "text": 5})
        term = LatentAlignmentTerm(
            anchor="z_signal",
            others=["z_image", "z_text"],
            distance="mean_mse",
            adapters={"z_signal": 4, "z_image": 4, "z_text": 4},
        )
        term.validateModel(model)
        assert sorted(term.adapters) == ["z_image", "z_signal", "z_text"]
        value = term(model, _batch(), _outputs(model, _batch()), 0)
        assert torch.isfinite(value)


class TestWithTheTrainer:
    def test_the_adapter_is_in_the_optimizer_and_the_trainer_parameters(self) -> None:
        trainer = _trainer()
        held = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        adapter = _linear(_termOf(trainer))
        assert all(id(p) in held for p in adapter.parameters())

    def test_one_step_moves_the_adapter_and_the_losses_carry_the_term(self) -> None:
        trainer = _trainer()
        adapter = _linear(_termOf(trainer))
        before = adapter.to_mu.weight.detach().clone()
        losses = trainer.computeLosses(_batch(), step=0, apply_dropout=False)
        assert losses.terms["latent_alignment"].item() > 0
        trainer.optimizer.zero_grad()
        losses.total.backward()  # type: ignore[no-untyped-call]
        trainer.optimizer.step()
        assert not torch.equal(before, adapter.to_mu.weight)

    def test_both_vaes_frozen_and_only_the_adapter_trains(self) -> None:
        model = _model()
        trainer = _trainer(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter = _linear(_termOf(trainer))
        weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        adapter_before = adapter.to_mu.weight.detach().clone()
        trainer.model.eval()  # the batch norms stand still, as the freeze callback would have them
        for step in range(5):
            losses = trainer.computeLosses(_batch(), step=step, apply_dropout=False)
            trainer.optimizer.zero_grad()
            losses.total.backward()  # type: ignore[no-untyped-call]
            trainer.optimizer.step()
        assert not torch.equal(adapter_before, adapter.to_mu.weight)
        for name, parameter in model.named_parameters():
            assert torch.equal(weights[name], parameter), name

    def test_the_freeze_callback_runs_the_recipe_end_to_end_through_fit(self) -> None:
        model = _model()
        term = _term(model, distance="wasserstein2_diag")
        trainer = Trainer(
            model,
            device="cpu",
            loss_terms={"latent_alignment": term},
            optimizer_kwargs={"lr": 0.01},
            callbacks=[FreezeModules(["*"])],
        )
        weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        adapter_before = _linear(term).to_mu.weight.detach().clone()
        trainer.fit([_batch(), _batch()], num_epochs=3)
        assert not torch.equal(adapter_before, _linear(term).to_mu.weight)
        for name, parameter in model.named_parameters():
            assert torch.equal(weights[name], parameter), name
        for name, buffer in model.named_buffers():
            assert torch.equal(buffers[name], buffer), name

    def test_the_adapter_learns_to_bring_the_spaces_together_when_the_vaes_are_frozen(self) -> None:
        model = _model()
        trainer = _trainer(model, distance="wasserstein2_diag")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainer.model.eval()
        batch = _batch()
        first = trainer.computeLosses(batch, step=0, apply_dropout=False).terms["latent_alignment"]
        for step in range(150):
            losses = trainer.computeLosses(batch, step=step, apply_dropout=False)
            trainer.optimizer.zero_grad()
            losses.total.backward()  # type: ignore[no-untyped-call]
            trainer.optimizer.step()
        last = trainer.computeLosses(batch, step=150, apply_dropout=False).terms["latent_alignment"]
        assert last.item() < 0.5 * first.item()

    def test_the_adapter_is_saved_in_the_checkpoint_and_restored_by_a_resume(
        self, tmp_path: Path
    ) -> None:
        trainer = _trainer()
        adapter = _linear(_termOf(trainer))
        torch.nn.init.constant_(adapter.to_mu.weight, 0.25)
        trainer.saveCheckpoint(tmp_path / "run.pt")
        state = readLossTermState(tmp_path / "run.pt", "latent_alignment")
        assert torch.equal(state["adapters.z_text.to_mu.weight"], adapter.to_mu.weight)

        fresh = _trainer()
        fresh.loadCheckpoint(tmp_path / "run.pt")
        restored = _linear(_termOf(fresh))
        assert torch.equal(restored.to_mu.weight, adapter.to_mu.weight)

    def test_a_resume_with_other_adapters_is_refused_by_the_checkpoint_check(
        self, tmp_path: Path
    ) -> None:
        _trainer().saveCheckpoint(tmp_path / "run.pt")
        other = _trainer(adapters={"z_text": 3, "z_signal": 3})
        with pytest.raises(ValueError, match="loss"):
            other.loadCheckpoint(tmp_path / "run.pt")

    def test_a_trainer_whose_adapters_leave_the_widths_unequal_is_refused_when_built(self) -> None:
        with pytest.raises(ValueError, match="adapters leave them at different widths"):
            _trainer(adapters={"z_text": 4})


class TestReadLossTermState:
    def test_a_checkpoint_without_the_term_lists_the_terms_it_has(self, tmp_path: Path) -> None:
        _trainer().saveCheckpoint(tmp_path / "run.pt")
        with pytest.raises(ValueError, match=r"no parameters for the loss term 'nope'.*latent_al"):
            readLossTermState(tmp_path / "run.pt", "nope")

    def test_a_checkpoint_with_no_loss_state_says_none(self, tmp_path: Path) -> None:
        plain = Trainer(_model(), device="cpu")
        plain.saveCheckpoint(tmp_path / "plain.pt")
        with pytest.raises(ValueError, match="terms with parameters in the file: none"):
            readLossTermState(tmp_path / "plain.pt", "latent_alignment")

    def test_a_missing_file_is_a_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            readLossTermState(tmp_path / "absent.pt", "latent_alignment")
