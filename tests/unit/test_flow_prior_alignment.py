"""The `flow_prior_alignment` term: a latent space pulled toward a frozen flow prior (ADR 0061)."""

from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.training import buildLossTerms
from global_vae.losses.priors import FlowPrior, fitFlowPrior, saveFlowPrior
from global_vae.losses.terms.flow_prior_alignment import FlowPriorAlignmentTerm
from global_vae.losses.terms.registry import getLossTermClass, listRegisteredLossTerms
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from tests.unit._tiny_pair_model import (
    BATCH,
    IN_DIM,
    LATENT,
    buildTinyModel,
    buildTinyTrainer,
    snapshot,
    tinyBatches,
)


def _fit(seed: int = 0, dim: int = LATENT, steps: int = 60) -> FlowPrior:
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(300, dim, generator=generator) * 2.0 + 1.0
    return fitFlowPrior(points, steps=steps, hidden=16, layers=1, seed=seed).prior


@pytest.fixture
def prior_file(tmp_path: Path) -> Path:
    return saveFlowPrior(_fit(), tmp_path / "prior.pt")


def _term(path: Path, model: GlobalVae | None = None, **kwargs: Any) -> FlowPriorAlignmentTerm:
    kwargs.setdefault("space", "z_image")
    term = FlowPriorAlignmentTerm(prior=path, **kwargs)
    term.validateModel(model or buildTinyModel())
    return term


def _outputs(model: GlobalVae, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    torch.manual_seed(1)
    return dict(model(model.selectEncoderInputs(batch)))


class TestRegistration:
    def test_the_term_is_registered_under_its_name(self) -> None:
        assert "flow_prior_alignment" in listRegisteredLossTerms()
        assert getLossTermClass("flow_prior_alignment") is FlowPriorAlignmentTerm


class TestArguments:
    @pytest.mark.parametrize("space", ["", None, 3])
    def test_a_space_that_is_not_a_name_is_refused(self, prior_file: Path, space: Any) -> None:
        with pytest.raises(ValueError, match="`space` must be a latent space name"):
            FlowPriorAlignmentTerm(space=space, prior=prior_file)

    def test_an_unknown_read_is_refused_by_name(self, prior_file: Path) -> None:
        with pytest.raises(ValueError, match=r"read must be one of \['mu', 'sample'\]"):
            FlowPriorAlignmentTerm("z_image", prior_file, read="logvar")

    def test_an_unknown_on_missing_is_refused_by_name(self, prior_file: Path) -> None:
        with pytest.raises(ValueError, match=r"on_missing must be one of \['raise', 'skip'\]"):
            FlowPriorAlignmentTerm("z_image", prior_file, on_missing="ignore")

    @pytest.mark.parametrize("draws", [0, -1, 1.5, True])
    def test_a_number_of_draws_below_one_is_refused(self, prior_file: Path, draws: Any) -> None:
        with pytest.raises(ValueError, match="num_draws must be an integer of 1 or more"):
            FlowPriorAlignmentTerm("z_image", prior_file, num_draws=draws)

    def test_a_missing_prior_file_is_refused_when_the_term_is_built(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No flow prior file"):
            FlowPriorAlignmentTerm("z_image", tmp_path / "none.pt")

    def test_a_file_that_is_not_a_prior_is_refused_when_the_term_is_built(
        self, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.pt"
        torch.save({"weights": torch.zeros(1)}, other)
        with pytest.raises(ValueError, match="is not a flow prior file"):
            FlowPriorAlignmentTerm("z_image", other)

    def test_the_prior_is_frozen_inside_the_term(self, prior_file: Path) -> None:
        term = FlowPriorAlignmentTerm("z_image", prior_file)
        assert list(term.parameters()) and not any(p.requires_grad for p in term.parameters())


class TestValidateModel:
    def test_a_space_the_model_does_not_have_is_refused_with_the_ones_it_has(
        self, prior_file: Path
    ) -> None:
        term = FlowPriorAlignmentTerm("z_audio", prior_file)
        with pytest.raises(ValueError, match=r"'z_audio' is not a latent space.*z_image"):
            term.validateModel(buildTinyModel())

    def test_a_prior_of_another_width_is_refused_and_the_fix_is_named(self, tmp_path: Path) -> None:
        path = saveFlowPrior(_fit(dim=5), tmp_path / "wide.pt")
        term = FlowPriorAlignmentTerm("z_image", path)
        with pytest.raises(ValueError, match=rf"width {LATENT}.*over width 5.*--dim {LATENT}"):
            term.validateModel(buildTinyModel())

    def test_a_prior_fitted_with_a_projection_fits_a_space_of_its_width(
        self, tmp_path: Path
    ) -> None:
        points = torch.randn(200, 7)
        prior = fitFlowPrior(points, dim=LATENT, steps=10, hidden=8, layers=1).prior
        term = FlowPriorAlignmentTerm("z_image", saveFlowPrior(prior, tmp_path / "proj.pt"))
        term.validateModel(buildTinyModel())


class TestForward:
    def test_it_is_the_alignment_loss_of_the_posterior_mean(self, prior_file: Path) -> None:
        model = buildTinyModel()
        term = _term(prior_file, model)
        batch = tinyBatches(1)[0]
        outputs = _outputs(model, batch)
        torch.manual_seed(5)
        value = term(model, batch, outputs, 0)
        torch.manual_seed(5)
        expected = term.prior.alignmentLoss(outputs["latent_params"]["z_image"][0])
        assert value.dim() == 0 and torch.equal(value, expected)
        assert (term.num_applied, term.num_skipped) == (1, 0)

    def test_read_sample_pulls_the_sample_of_the_step(self, prior_file: Path) -> None:
        model = buildTinyModel()
        term = _term(prior_file, model, read="sample")
        batch = tinyBatches(1)[0]
        outputs = _outputs(model, batch)
        torch.manual_seed(5)
        value = term(model, batch, outputs, 0)
        torch.manual_seed(5)
        assert torch.equal(value, term.prior.alignmentLoss(outputs["latent_samples"]["z_image"]))

    def test_the_gradient_reaches_the_encoder_of_the_space_and_nothing_else(
        self, prior_file: Path
    ) -> None:
        model = buildTinyModel()
        term = _term(prior_file, model)
        batch = tinyBatches(1)[0]
        backward(term(model, batch, _outputs(model, batch), 0))
        reached = {n.split(".")[1] for n, p in model.named_parameters() if p.grad is not None}
        assert {n.split(".")[0] for n, p in model.named_parameters() if p.grad is not None} == {
            "encoders"
        }
        assert reached == {"image"}
        assert all(p.grad is None for p in term.parameters())

    def test_more_draws_average_the_loss(self, prior_file: Path) -> None:
        model = buildTinyModel()
        batch = tinyBatches(1)[0]
        outputs = _outputs(model, batch)
        spread = {}
        for draws in (1, 16):
            term = _term(prior_file, model, num_draws=draws)
            values = []
            for seed in range(15):
                torch.manual_seed(seed)
                values.append(float(term(model, batch, outputs, 0).detach()))
            spread[draws] = torch.tensor(values).std()
        assert spread[16] < spread[1]

    def test_a_batch_that_does_not_reach_the_space_is_refused_and_the_way_out_is_named(
        self, prior_file: Path
    ) -> None:
        model = buildTinyModel()
        term = _term(prior_file, model)
        batch = {"signal": tinyBatches(1)[0]["signal"]}
        with pytest.raises(ValueError, match=r"did not reach the latent space 'z_image'") as raised:
            term(model, batch, _outputs(model, batch), 0)
        assert "on_missing: skip" in str(raised.value)

    def test_skip_returns_a_scalar_zero_and_counts_the_batch(self, prior_file: Path) -> None:
        model = buildTinyModel()
        term = _term(prior_file, model, on_missing="skip")
        batch = {"signal": tinyBatches(1)[0]["signal"]}
        value = term(model, batch, _outputs(model, batch), 0)
        assert value.dim() == 0 and float(value) == 0.0
        assert (term.num_applied, term.num_skipped) == (0, 1)


def _trainer(path: Path, **term_kwargs: Any) -> Trainer:
    term_kwargs.setdefault("space", "z_image")
    term_kwargs.setdefault("on_missing", "skip")
    term = FlowPriorAlignmentTerm(prior=path, **term_kwargs)
    return buildTinyTrainer(
        loss_terms={"flow_prior_alignment": term}, loss_term_weights={"flow_prior_alignment": 0.1}
    )


class TestWithTheTrainer:
    def test_a_mixed_stream_trains_and_the_term_only_sees_batches_of_its_modality(
        self, prior_file: Path
    ) -> None:
        trainer = _trainer(prior_file)
        paired = tinyBatches(2)
        signal_only = [{"signal": b["signal"]} for b in tinyBatches(2, seed=3)]
        trainer.fit([*paired, *signal_only], num_epochs=2)
        term = trainer.loss_terms["flow_prior_alignment"]
        assert isinstance(term, FlowPriorAlignmentTerm)
        assert (term.num_applied, term.num_skipped) == (4, 4)
        assert "train/loss/flow_prior_alignment" in trainer.history[-1]

    def test_the_prior_does_not_move_and_the_encoder_of_the_space_does(
        self, prior_file: Path
    ) -> None:
        trainer = _trainer(prior_file)
        term = trainer.loss_terms["flow_prior_alignment"]
        assert isinstance(term, FlowPriorAlignmentTerm)
        before_prior = snapshot(term)
        before_model = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=3)
        for name, tensor in snapshot(term).items():
            assert torch.equal(tensor, before_prior[name]), name
        moved = [
            n for n, t in snapshot(trainer.model).items() if not torch.equal(t, before_model[n])
        ]
        assert any(n.startswith("encoders.image") for n in moved)

    def test_the_prior_is_in_the_checkpoint_state_of_the_term(
        self, prior_file: Path, tmp_path: Path
    ) -> None:
        trainer = _trainer(prior_file)
        trainer.saveCheckpoint(tmp_path / "run.pt")
        state = torch.load(tmp_path / "run.pt", weights_only=True)["loss_terms_state_dict"]
        assert any(k.startswith("flow_prior_alignment.prior.net.") for k in state)
        assert "flow_prior_alignment.prior.reference_loss" in state

    def test_a_resume_uses_the_prior_the_run_started_with_whatever_the_file_holds_now(
        self, prior_file: Path, tmp_path: Path
    ) -> None:
        trainer = _trainer(prior_file)
        trainer.fit(tinyBatches(2), num_epochs=1)
        trainer.saveCheckpoint(tmp_path / "run.pt")
        other = saveFlowPrior(_fit(seed=9), tmp_path / "other.pt")
        fresh = _trainer(other)
        fresh_term = fresh.loss_terms["flow_prior_alignment"]
        original = trainer.loss_terms["flow_prior_alignment"]
        assert isinstance(fresh_term, FlowPriorAlignmentTerm)
        assert isinstance(original, FlowPriorAlignmentTerm)
        assert not torch.equal(
            fresh_term.prior.state_dict()["net.0.weight"],
            original.prior.state_dict()["net.0.weight"],
        )
        fresh.loadCheckpoint(tmp_path / "run.pt")
        for name, tensor in original.prior.state_dict().items():
            assert torch.equal(tensor, fresh_term.prior.state_dict()[name]), name

    def test_the_config_builds_the_term(self, prior_file: Path) -> None:
        config = loadExperimentConfig(
            "experiment/pair_align",
            overrides=[
                "data.loader_factory=unused:unused",
                "data.train_path=unused",
                "training.loss_terms=[{name: flow_prior_alignment, weight: 0.01, kwargs: "
                f"{{space: z_image, prior: {prior_file}, on_missing: skip}}}}]",
            ],
        )
        terms, weights = buildLossTerms(config.training)
        term = terms["flow_prior_alignment"]
        assert isinstance(term, FlowPriorAlignmentTerm)
        assert (term.space, term.on_missing, weights["flow_prior_alignment"]) == (
            "z_image",
            "skip",
            0.01,
        )


def test_the_batch_size_and_width_of_the_helpers_are_what_the_tests_assume() -> None:
    assert (BATCH, IN_DIM, LATENT) == (8, 8, 3)
