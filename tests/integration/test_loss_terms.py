"""Extra loss terms at the single loss site (roadmap P5-6, ADR 0047).

`Trainer.computeLosses` adds a reconstruction term and a regularization term, and the later
phases bring more objectives. This file pins what makes a term safe to add:

- a dummy term with one parameter trains: the parameter is in the optimizer and moves, the term is
  logged as `train/loss/<name>` and `val/loss/<name>` (the CSV logger needs no change), it is
  saved in the checkpoint under a key of its own, and a run resumed after half of its epochs gives
  the curve and the weights of the uninterrupted run (the guarantee of ADR 0032);
- the weight is a number or a beta schedule, the term is scaled by the accumulation like the total,
  computed in float32 under mixed precision, and runs under `torch.compile` (the model is the
  trainer's `GlobalVae`, not the compiled wrapper);
- the terms and their weights are validated when the trainer is built, at the right place;
- **a run with no term is the 2.0.0 run**: a copy of the old `computeLosses` is kept below and a
  full run with it is compared with the trainer's, history and weights, bit for bit, in four
  configurations (plain, bf16 with accumulation and clipping, modality dropout, a beta schedule).

Uses its own dummy encoder and decoder, registered under a `_loss_terms_test` suffix, and its own
terms, registered under unique names (see `test_trainer.py` for why sibling test files do not share
fixtures). The registry contract is in `tests/unit/test_registry_contract.py`.
"""

import csv
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from global_vae.config.training import (
    BetaScheduleConfig,
    LoggerEntryConfig,
    LossTermConfig,
    TrainingConfig,
    buildLossTerms,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.losses.terms.base import AbstractLossTerm
from global_vae.losses.terms.registry import registerLossTerm
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.trainer import RESERVED_LOSS_TERM_NAMES, StepLosses, Trainer

INPUT_DIM = 12
LATENT_DIM = 3
HIDDEN_DIM = 10
BATCH_SIZE = 6
TERM = "scale_term"
LATENT = "z_fused"


@registerEncoder("dummy_signal_encoder_loss_terms_test")
class _DummySignalEncoder(AbstractEncoder):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.hidden = nn.Linear(input_dim, HIDDEN_DIM)
        self.to_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.to_logvar = nn.Linear(HIDDEN_DIM, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.hidden(x))
        return self.to_mu(h), self.to_logvar(h)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return "signal"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_signal_decoder_loss_terms_test")
class _DummySignalDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.hidden = nn.Linear(latent_dim, HIDDEN_DIM)
        self.project = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(torch.tanh(self.hidden(z)))
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


@registerEncoder("dummy_other_encoder_loss_terms_test")
class _DummyOtherEncoder(_DummySignalEncoder):
    @property
    def modality_name(self) -> str:
        return "other"


@registerDecoder("dummy_other_decoder_loss_terms_test")
class _DummyOtherDecoder(_DummySignalDecoder):
    @property
    def modality_name(self) -> str:
        return "other"


@registerLossTerm("dummy_scale_loss_terms_test")
class _ScaleTerm(AbstractLossTerm):
    """A term with one parameter: `(scale * mean(mu ** 2) - 0.5) ** 2`.

    It records what the trainer handed it at every call, so the tests can check the contract.
    """

    def __init__(self, initial: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(initial)))
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        mu = outputs["latent_params"][LATENT][0]
        self.calls.append(
            {
                "model": model,
                "batch_keys": set(batch),
                "step": step,
                "training": self.training,
                "mu_dtype": mu.dtype,
                "sample_dtype": outputs["latent_samples"][LATENT].dtype,
                "output_keys": set(outputs),
            }
        )
        value: torch.Tensor = (self.scale * mu.pow(2).mean() - 0.5).pow(2)
        return value


@registerLossTerm("dummy_anchor_loss_terms_test")
class _AnchorTerm(AbstractLossTerm):
    """A term that depends on its parameter only: `(anchor - 3) ** 2`, so no sampling noise."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        return (self.anchor - 3.0).pow(2)


class _FixedTerm(AbstractLossTerm):
    """A term with no parameter that returns a given tensor."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        value: torch.Tensor = self.value
        return value


class _PickyTerm(AbstractLossTerm):
    """A term that refuses every model, in `validateModel`."""

    def validateModel(self, model: GlobalVae) -> None:
        raise ValueError(
            f"picky: no latent space named 'z_missing' in {sorted(model.latent_spaces)}"
        )

    def forward(
        self,
        model: GlobalVae,
        batch: Mapping[str, torch.Tensor],
        outputs: Mapping[str, Any],
        step: int,
    ) -> torch.Tensor:
        return torch.zeros(())


def _buildModel(seed: int = 0, two_modalities: bool = False) -> GlobalVae:
    torch.manual_seed(seed)
    modality_configs = {
        "signal": {
            "encoder": "dummy_signal_encoder_loss_terms_test",
            "decoder": "dummy_signal_decoder_loss_terms_test",
        }
    }
    if not two_modalities:
        return GlobalVae.createSingleLatent(
            modality_configs=modality_configs, latent_dim=LATENT_DIM
        )
    modality_configs["other"] = {
        "encoder": "dummy_other_encoder_loss_terms_test",
        "decoder": "dummy_other_decoder_loss_terms_test",
    }
    return GlobalVae.createSingleLatent(
        modality_configs=modality_configs, latent_dim=LATENT_DIM, fusion_strategy="poe"
    )


def _batches(
    num_batches: int, seed: int = 0, two_modalities: bool = False
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    names = ["signal", "other"] if two_modalities else ["signal"]
    return [
        {name: torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator) for name in names}
        for _ in range(num_batches)
    ]


def _trainer(
    terms: dict[str, AbstractLossTerm] | None = None,
    weights: dict[str, Any] | None = None,
    model: GlobalVae | None = None,
    **kwargs: Any,
) -> Trainer:
    kwargs.setdefault("optimizer_kwargs", {"lr": 0.05})
    return Trainer(
        model if model is not None else _buildModel(),
        device="cpu",
        loss_terms=terms,
        loss_term_weights=weights,
        **kwargs,
    )


def _term(trainer: Trainer, name: str = TERM) -> Any:
    return trainer.loss_terms[name]


def _needsCpuGradScaler() -> None:
    try:
        torch.amp.GradScaler("cpu")
    except (AttributeError, RuntimeError, ValueError):  # pragma: no cover - old PyTorch
        pytest.skip("torch.amp.GradScaler does not support the CPU in this PyTorch")


class TestConfig:
    """`training.loss_terms` becomes terms and weights through the registry."""

    def test_a_config_without_terms_builds_nothing(self) -> None:
        assert buildLossTerms(TrainingConfig()) == ({}, {})

    def test_entries_become_terms_with_their_kwargs_and_weights(self) -> None:
        config = TrainingConfig(
            loss_terms=[
                LossTermConfig(name="dummy_scale_loss_terms_test", kwargs={"initial": 2.5}),
                LossTermConfig(name="dummy_anchor_loss_terms_test", weight=0.25),
            ]
        )
        terms, weights = buildLossTerms(config)
        assert list(terms) == ["dummy_scale_loss_terms_test", "dummy_anchor_loss_terms_test"]
        scale_term = terms["dummy_scale_loss_terms_test"]
        assert isinstance(scale_term, _ScaleTerm) and scale_term.scale.item() == 2.5
        assert weights == {"dummy_anchor_loss_terms_test": 0.25}

    def test_a_schedule_is_resolved_through_the_beta_schedule_registry(self) -> None:
        config = TrainingConfig(
            loss_terms=[
                LossTermConfig(
                    name="dummy_anchor_loss_terms_test",
                    schedule=BetaScheduleConfig(
                        strategy="linear_warmup", kwargs={"warmup_steps": 10, "end_value": 0.4}
                    ),
                )
            ]
        )
        _, weights = buildLossTerms(config)
        schedule = weights["dummy_anchor_loss_terms_test"]
        assert isinstance(schedule, LinearWarmupBetaSchedule)
        assert schedule(0) == 0.0 and schedule(5) == pytest.approx(0.2) and schedule(99) == 0.4

    def test_an_unknown_term_is_a_key_error_that_lists_the_registered_ones(self) -> None:
        config = TrainingConfig(loss_terms=[LossTermConfig(name="no_such_term")])
        with pytest.raises(KeyError, match="Unknown loss term 'no_such_term'.*dummy_scale"):
            buildLossTerms(config)

    def test_a_term_listed_twice_is_refused_and_the_entry_is_named(self) -> None:
        config = TrainingConfig(
            loss_terms=[
                LossTermConfig(name="dummy_anchor_loss_terms_test"),
                LossTermConfig(name="dummy_anchor_loss_terms_test", weight=2.0),
            ]
        )
        with pytest.raises(ValueError, match=r"training\.loss_terms\[1\].*listed twice"):
            buildLossTerms(config)

    def test_a_weight_and_a_schedule_together_are_refused(self) -> None:
        config = TrainingConfig(
            loss_terms=[
                LossTermConfig(
                    name="dummy_anchor_loss_terms_test",
                    weight=1.0,
                    schedule=BetaScheduleConfig(strategy="constant"),
                )
            ]
        )
        with pytest.raises(ValueError, match=r"`weight` or a `schedule`, not both"):
            buildLossTerms(config)

    @pytest.mark.parametrize("weight", [-0.1, math.nan, math.inf])
    def test_a_negative_or_non_finite_weight_is_refused(self, weight: float) -> None:
        config = TrainingConfig(
            loss_terms=[LossTermConfig(name="dummy_anchor_loss_terms_test", weight=weight)]
        )
        with pytest.raises(ValueError, match=r"loss_terms\[0\]\.weight.*at least 0"):
            buildLossTerms(config)

    def test_yaml_builds_a_trainer_with_the_term_and_its_ramp(self) -> None:
        yaml = """
        num_epochs: 2
        loss_terms:
          - name: dummy_scale_loss_terms_test
            kwargs: {initial: 2.0}
            schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 4}}
          - name: dummy_anchor_loss_terms_test
            weight: 0.5
        """
        merged = OmegaConf.merge(OmegaConf.structured(TrainingConfig), OmegaConf.create(yaml))
        config = OmegaConf.to_object(merged)
        assert isinstance(config, TrainingConfig)
        trainer = buildTrainerFromConfig(_buildModel(), config)

        assert list(trainer.loss_terms) == [
            "dummy_scale_loss_terms_test",
            "dummy_anchor_loss_terms_test",
        ]
        assert _term(trainer, "dummy_scale_loss_terms_test").scale.item() == 2.0
        assert trainer._computeLossTermWeights(2) == {
            "dummy_scale_loss_terms_test": 0.5,
            "dummy_anchor_loss_terms_test": 0.5,
        }

    def test_a_config_without_terms_gives_a_trainer_without_terms(self) -> None:
        trainer = buildTrainerFromConfig(_buildModel(), TrainingConfig())
        assert len(trainer.loss_terms) == 0 and trainer.loss_term_weights == {}


class TestTrainerValidation:
    """The names and weights are checked when the trainer is built."""

    @pytest.mark.parametrize("name", RESERVED_LOSS_TERM_NAMES)
    def test_a_reserved_name_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"cannot be named '{name}'"):
            _trainer({name: _FixedTerm(torch.zeros(()))})

    def test_a_dotted_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot contain a dot"):
            _trainer({"a.b": _FixedTerm(torch.zeros(()))})

    def test_a_weight_for_a_name_that_is_not_a_term_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"loss_term_weights names \['scale_trem'\]"):
            _trainer({TERM: _ScaleTerm()}, {"scale_trem": 0.5})

    @pytest.mark.parametrize("weight", [-1.0, math.nan])
    def test_a_bad_weight_is_refused(self, weight: float) -> None:
        with pytest.raises(ValueError, match="finite number of at least 0"):
            _trainer({TERM: _ScaleTerm()}, {TERM: weight})

    def test_validate_model_runs_when_the_trainer_is_built(self) -> None:
        with pytest.raises(ValueError, match="picky: no latent space named 'z_missing'"):
            _trainer({"picky": _PickyTerm()})

    @pytest.mark.parametrize(
        "value",
        [torch.ones(BATCH_SIZE), torch.ones(1), 1.5],
        ids=["vector", "one_element", "float"],
    )
    def test_a_term_that_does_not_return_a_scalar_tensor_is_refused_by_name(
        self, value: Any
    ) -> None:
        trainer = _trainer({"flat": _FixedTerm(value)})
        with pytest.raises(ValueError, match="Loss term 'flat' must return a scalar tensor"):
            trainer.computeLosses(_batches(1)[0], step=0)


class TestTheSingleLossSite:
    """`computeLosses` adds each weighted term, and training and validation share it."""

    def _losses(self, trainer: Trainer, step: int = 0) -> StepLosses:
        torch.manual_seed(3)
        return trainer.computeLosses(_batches(1)[0], step=step)

    def test_the_total_is_the_reconstruction_plus_the_regularization_plus_the_terms(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm(), "anchor": _AnchorTerm()}, {"anchor": 0.5})
        losses = self._losses(trainer)
        assert set(losses.terms) == {TERM, "anchor"}
        expected = (
            losses.reconstruction
            + losses.regularization
            + losses.terms[TERM]
            + losses.terms["anchor"]
        )
        assert losses.total.item() == pytest.approx(expected.item(), rel=1e-6)
        assert losses.terms["anchor"].item() == pytest.approx(0.5 * 4.0)

    def test_the_value_is_the_weighted_term_and_the_weight_is_a_number(self) -> None:
        full = self._losses(_trainer({TERM: _ScaleTerm()}))
        quarter = self._losses(_trainer({TERM: _ScaleTerm()}, {TERM: 0.25}))
        assert quarter.terms[TERM].item() == pytest.approx(0.25 * full.terms[TERM].item())

    def test_the_weight_follows_a_schedule_at_the_step(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=10, start_value=0.0, end_value=1.0)
        trainer = _trainer({TERM: _ScaleTerm()}, {TERM: schedule})
        full = self._losses(_trainer({TERM: _ScaleTerm()}))
        assert self._losses(trainer, step=0).terms[TERM].item() == 0.0
        assert self._losses(trainer, step=5).terms[TERM].item() == pytest.approx(
            0.5 * full.terms[TERM].item()
        )
        assert self._losses(trainer, step=50).terms[TERM].item() == pytest.approx(
            full.terms[TERM].item()
        )

    def test_the_metrics_have_the_three_losses_then_one_key_per_term(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm(), "anchor": _AnchorTerm()})
        assert list(self._losses(trainer).asMetrics()) == [
            "loss/total",
            "loss/reconstruction",
            "loss/regularization",
            f"loss/{TERM}",
            "loss/anchor",
        ]

    def test_the_term_receives_the_model_the_batch_the_outputs_and_the_step(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()})
        self._losses(trainer, step=7)
        call = _term(trainer).calls[-1]
        assert call["model"] is trainer.model
        assert call["batch_keys"] == {"signal"}
        assert call["output_keys"] == {"reconstructions", "latent_params", "latent_samples"}
        assert call["step"] == 7

    def test_validation_goes_through_the_same_function_in_evaluation_mode(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()})
        trainer.fit(_batches(3), num_epochs=2, val_dataloader=_batches(2, seed=9))
        calls = _term(trainer).calls
        training_calls = [c for c in calls if c["training"]]
        evaluation_calls = [c for c in calls if not c["training"]]
        assert len(training_calls) == 6 and len(evaluation_calls) == 4
        assert {c["step"] for c in evaluation_calls} == {3, 6}  # the step is not advanced

    def test_a_trainer_without_terms_has_empty_terms(self) -> None:
        losses = self._losses(_trainer())
        assert losses.terms == {}
        assert list(losses.asMetrics()) == [
            "loss/total",
            "loss/reconstruction",
            "loss/regularization",
        ]
        assert losses.total.item() == pytest.approx(
            (losses.reconstruction + losses.regularization).item()
        )


class TestATermWithAParameterTrains:
    """The parameter is in the optimizer, moves, is logged and reaches the existing loggers."""

    def test_the_parameter_is_in_the_one_optimizer_group_and_moves(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()})
        optimizer_params = [p for g in trainer.optimizer.param_groups for p in g["params"]]
        assert len(trainer.optimizer.param_groups) == 1
        assert any(p is _term(trainer).scale for p in optimizer_params)
        assert len(optimizer_params) == len(list(trainer.model.parameters())) + 1

        before = _term(trainer).scale.item()
        history = trainer.fit(_batches(4), num_epochs=3)
        assert _term(trainer).scale.item() != before
        assert _term(trainer).scale.grad is not None
        assert all(math.isfinite(entry[f"train/loss/{TERM}"]) for entry in history)

    def test_the_term_is_in_the_history_for_training_and_validation(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()})
        history = trainer.fit(_batches(3), num_epochs=2, val_dataloader=_batches(2, seed=9))
        for entry in history:
            assert f"train/loss/{TERM}" in entry and f"val/loss/{TERM}" in entry

    def test_the_csv_logger_writes_the_term_without_a_change(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            num_epochs=2,
            loss_terms=[LossTermConfig(name="dummy_scale_loss_terms_test")],
            loggers=[LoggerEntryConfig(name="csv", kwargs={"path": str(tmp_path / "m.csv")})],
        )
        trainer = buildTrainerFromConfig(_buildModel(), config)
        trainer.fit(_batches(3), num_epochs=2, val_dataloader=_batches(2, seed=9))
        for callback in trainer.callbacks:
            close = getattr(callback, "close", None)
            if callable(close):
                close()
        with open(tmp_path / "m.csv", newline="") as handle:
            metrics = {row["metric"] for row in csv.DictReader(handle)}
        assert "train/loss/dummy_scale_loss_terms_test" in metrics
        assert "val/loss/dummy_scale_loss_terms_test" in metrics

    def test_an_optimizer_instance_gets_the_missing_parameters_as_a_group(self) -> None:
        model = _buildModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, optimizer=optimizer, device="cpu", loss_terms={TERM: _ScaleTerm()})
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[1]["params"] == [_term(trainer).scale]
        before = _term(trainer).scale.item()
        trainer.fit(_batches(3), num_epochs=2)
        assert _term(trainer).scale.item() != before

    def test_parameters_the_optimizer_already_holds_are_not_added_again(self) -> None:
        model = _buildModel()
        term = _ScaleTerm()
        optimizer = torch.optim.SGD([*model.parameters(), *term.parameters()], lr=0.1)
        Trainer(model, optimizer=optimizer, device="cpu", loss_terms={TERM: term})
        assert len(optimizer.param_groups) == 1

    def test_clipping_covers_the_parameters_of_the_terms(self) -> None:
        trainer = _trainer(
            {"anchor": _AnchorTerm()},
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.0},
            grad_clip_norm=1e-3,
        )
        trainer.fit(_batches(2), num_epochs=1)
        gradients = [
            p.grad.flatten()
            for p in [*trainer.model.parameters(), *trainer.loss_terms.parameters()]
            if p.grad is not None
        ]
        norm = torch.cat(gradients).norm().item()
        assert norm <= 1e-3 * (1 + 1e-4)
        assert _term(trainer, "anchor").anchor.grad is not None

    def test_the_terms_follow_the_train_and_eval_modes_of_the_trainer(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()})
        trainer.fitEpoch(_batches(2), epoch=0)
        assert trainer.loss_terms.training
        trainer.evaluate(_batches(2, seed=9))
        assert not trainer.loss_terms.training


class TestPrecisionAccumulationAndCompile:
    """The paths of ADR 0031 with a term."""

    def test_a_term_sees_float32_outputs_under_bf16(self) -> None:
        trainer = _trainer({TERM: _ScaleTerm()}, mixed_precision="bf16")
        history = trainer.fit(_batches(3), num_epochs=2)
        assert {c["mu_dtype"] for c in _term(trainer).calls} == {torch.float32}
        assert {c["sample_dtype"] for c in _term(trainer).calls} == {torch.float32}
        assert _term(trainer).scale.dtype == torch.float32
        assert all(math.isfinite(entry[f"train/loss/{TERM}"]) for entry in history)

    def test_fp16_trains_the_parameter_through_the_scaler(self) -> None:
        _needsCpuGradScaler()
        trainer = _trainer({TERM: _ScaleTerm()}, mixed_precision="fp16")
        before = _term(trainer).scale.item()
        history = trainer.fit(_batches(3), num_epochs=2)
        assert _term(trainer).scale.item() != before
        assert all(math.isfinite(entry[f"train/loss/{TERM}"]) for entry in history)

    def test_a_term_is_scaled_by_the_accumulation_like_the_total(self) -> None:
        """N=2 over (a, a) moves the parameter exactly as one step on `a`."""
        a = _batches(1)[0]

        def run(accumulate: int, batches: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
            trainer = _trainer(
                {"anchor": _AnchorTerm()},
                optimizer=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.1},
                accumulate_grad_batches=accumulate,
            )
            history = trainer.fit(batches, num_epochs=1)
            return _term(trainer, "anchor").anchor.item(), history[0]["train/loss/anchor"]

        accumulated, accumulated_metric = run(2, [a, a])
        single, single_metric = run(1, [a])
        # d/da (a - 3) ** 2 = -4 at a = 1, so one SGD step of 0.1 gives 1.4.
        assert single == pytest.approx(1.4)
        assert accumulated == pytest.approx(single)
        # The metric is the mean over the batches, so it does not grow with the accumulation.
        assert accumulated_metric == pytest.approx(single_metric) == pytest.approx(4.0)

    def test_a_compiled_model_trains_with_a_term_that_receives_the_plain_model(self) -> None:
        trainer = _trainer(
            {TERM: _ScaleTerm()}, compile_model=True, compile_kwargs={"backend": "eager"}
        )
        before = _term(trainer).scale.item()
        history = trainer.fit(_batches(3), num_epochs=2)
        assert _term(trainer).scale.item() != before
        assert all(call["model"] is trainer.model for call in _term(trainer).calls)
        assert trainer.model is not trainer._forward
        assert all(math.isfinite(entry[f"train/loss/{TERM}"]) for entry in history)

    def test_it_works_with_clipping_and_modality_dropout(self) -> None:
        trainer = _trainer(
            {TERM: _ScaleTerm()},
            model=_buildModel(two_modalities=True),
            mixed_precision="bf16",
            grad_clip_norm=1.0,
            modality_dropout_p=0.5,
        )
        history = trainer.fit(_batches(4, two_modalities=True), num_epochs=3)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)


class TestCheckpointAndResume:
    """The parameters of a term are saved under a key of their own and restored exactly."""

    def _saved(self, path: Path, **kwargs: Any) -> Trainer:
        trainer = _trainer({TERM: _ScaleTerm()}, **kwargs)
        trainer.fit(_batches(3), num_epochs=2)
        trainer.saveCheckpoint(path)
        return trainer

    def test_the_state_is_in_a_key_of_its_own_and_readable_in_the_safe_mode(
        self, tmp_path: Path
    ) -> None:
        trainer = self._saved(tmp_path / "c.pt")
        checkpoint = torch.load(tmp_path / "c.pt", weights_only=True)
        assert list(checkpoint["loss_terms_state_dict"]) == [f"{TERM}.scale"]
        assert torch.equal(
            checkpoint["loss_terms_state_dict"][f"{TERM}.scale"], _term(trainer).scale.detach()
        )
        assert not any(name.startswith(TERM) for name in checkpoint["model_state_dict"])

    def test_it_is_restored_with_the_optimizer_state_of_the_term(self, tmp_path: Path) -> None:
        saved = self._saved(tmp_path / "c.pt", optimizer=torch.optim.Adam)
        fresh = _trainer({TERM: _ScaleTerm(initial=9.0)}, model=_buildModel(1))
        fresh.loadCheckpoint(tmp_path / "c.pt")
        assert torch.equal(_term(fresh).scale, _term(saved).scale)
        saved_state = saved.optimizer.state[_term(saved).scale]
        fresh_state = fresh.optimizer.state[_term(fresh).scale]
        assert torch.equal(saved_state["exp_avg"], fresh_state["exp_avg"])
        assert torch.equal(saved_state["exp_avg_sq"], fresh_state["exp_avg_sq"])

    @pytest.mark.parametrize(
        "extras", [{}, {"mixed_precision": "bf16", "accumulate_grad_batches": 2}]
    )
    def test_a_resumed_run_gives_the_curve_and_the_weights_of_the_uninterrupted_one(
        self, tmp_path: Path, extras: dict[str, Any]
    ) -> None:
        batches = _batches(4)
        schedule = LinearWarmupBetaSchedule(warmup_steps=6, end_value=1.0)

        def build(seed: int, initial: float) -> Trainer:
            return _trainer(
                {TERM: _ScaleTerm(initial=initial)},
                {TERM: schedule},
                model=_buildModel(seed),
                optimizer=torch.optim.Adam,
                **extras,
            )

        torch.manual_seed(7)
        whole = build(0, 1.0)
        history_whole = whole.fit(batches, num_epochs=4)

        torch.manual_seed(7)
        first = build(0, 1.0)
        first.fit(batches, num_epochs=2)
        first.saveCheckpoint(tmp_path / "half.pt")

        second = build(5, 9.0)  # another model and another initial value: only the file survives
        second.loadCheckpoint(tmp_path / "half.pt")
        history_resumed = second.fit(batches, num_epochs=2)

        assert len(history_resumed) == len(history_whole) == 4
        for expected, observed in zip(history_whole, history_resumed, strict=True):
            assert expected.keys() == observed.keys()
            for key, value in expected.items():
                assert observed[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
        assert _term(second).scale.item() == pytest.approx(_term(whole).scale.item(), rel=1e-6)
        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                second.model.state_dict()[name], tensor, rtol=1e-6, atol=1e-6
            )

    def test_the_resume_helper_of_the_config_restores_the_terms(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            num_epochs=4,
            loss_terms=[LossTermConfig(name="dummy_scale_loss_terms_test", weight=0.5)],
        )
        first = buildTrainerFromConfig(_buildModel(), config)
        first.fit(_batches(3), num_epochs=2)
        first.saveCheckpoint(tmp_path / "c.pt")
        config.resume_from = str(tmp_path / "c.pt")

        second = buildTrainerFromConfig(_buildModel(3), config)
        remaining = resumeTrainerFromConfig(second, config)
        assert remaining == 2
        name = "dummy_scale_loss_terms_test"
        assert torch.equal(_term(second, name).scale, _term(first, name).scale)

    def test_a_run_without_terms_saves_no_term_key(self, tmp_path: Path) -> None:
        trainer = _trainer()
        trainer.fit(_batches(2), num_epochs=1)
        trainer.saveCheckpoint(tmp_path / "c.pt")
        assert "loss_terms_state_dict" not in torch.load(tmp_path / "c.pt", weights_only=True)

    def test_a_checkpoint_of_a_run_without_terms_is_refused_by_a_trainer_with_a_term(
        self, tmp_path: Path
    ) -> None:
        plain = _trainer()
        plain.fit(_batches(2), num_epochs=1)
        plain.saveCheckpoint(tmp_path / "plain.pt")

        with pytest.raises(ValueError, match=r"no state for the loss term parameter.*scale"):
            _trainer({TERM: _ScaleTerm()}).loadCheckpoint(tmp_path / "plain.pt")

    def test_the_weights_alone_load_and_the_new_term_keeps_its_initial_parameters(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        plain = _trainer()
        plain.fit(_batches(2), num_epochs=1)
        plain.saveCheckpoint(tmp_path / "plain.pt")

        fresh = _trainer({TERM: _ScaleTerm(initial=4.0)})
        with caplog.at_level(logging.WARNING):
            fresh.loadCheckpoint(tmp_path / "plain.pt", restore_optimizer=False)
        assert _term(fresh).scale.item() == 4.0
        assert "keep their initial values" in caplog.text

    def test_a_checkpoint_with_a_term_is_refused_by_a_trainer_without_it(
        self, tmp_path: Path
    ) -> None:
        self._saved(tmp_path / "c.pt")
        with pytest.raises(ValueError, match=rf"holds loss term state \['{TERM}.scale'\]"):
            _trainer().loadCheckpoint(tmp_path / "c.pt")

    def test_a_non_strict_load_drops_a_term_the_trainer_does_not_have(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._saved(tmp_path / "c.pt")
        with caplog.at_level(logging.WARNING):
            _trainer().loadCheckpoint(tmp_path / "c.pt", restore_optimizer=False, strict=False)
        assert "Dropped" in caplog.text


class _LegacyTrainer(Trainer):
    """A trainer whose `computeLosses` is the one of 2.0.0, before loss terms existed."""

    def computeLosses(
        self, batch: dict[str, torch.Tensor], step: int, apply_dropout: bool = True
    ) -> StepLosses:
        if not batch:
            raise ValueError("Trainer.computeLosses received an empty batch.")

        encoder_inputs = self.model.selectEncoderInputs(batch)
        inputs = self._applyModalityDropout(encoder_inputs) if apply_dropout else encoder_inputs
        with self._autocast():
            outputs = self._forward(inputs)
        if self.mixed_precision is not None:
            outputs = {
                **outputs,
                "reconstructions": {k: v.float() for k, v in outputs["reconstructions"].items()},
                "latent_params": {
                    k: (mu.float(), logvar.float())
                    for k, (mu, logvar) in outputs["latent_params"].items()
                },
            }

        reconstruction_loss = computeTotalReconstructionLoss(
            outputs["reconstructions"],
            batch,
            weights=self.reconstruction_weights,
            loss_fn=self.reconstruction_loss_fn,
            reduction=self.reconstruction_reduction,
        )
        regularization_loss = self.model.computeRegularizationLoss(
            outputs["latent_params"], beta=self._computeBeta(step)
        )
        return StepLosses(
            total=reconstruction_loss + regularization_loss,
            reconstruction=reconstruction_loss,
            regularization=regularization_loss,
        )


_LEGACY_CASES: dict[str, dict[str, Any]] = {
    "plain": {},
    "bf16_accumulation_clipping": {
        "mixed_precision": "bf16",
        "accumulate_grad_batches": 2,
        "grad_clip_norm": 1.0,
    },
    "modality_dropout": {
        "two_modalities": True,
        "modality_dropout_p": 0.5,
        "grad_clip_norm": 1.0,
    },
    "beta_schedule": {
        "beta_schedules": {LATENT: LinearWarmupBetaSchedule(warmup_steps=5, end_value=0.5)},
        "optimizer": torch.optim.Adam,
    },
}


class TestARunWithoutTermsIsTheLegacyRun:
    """No `loss_terms`: the same run as 2.0.0, bit for bit."""

    @pytest.mark.parametrize("case", list(_LEGACY_CASES), ids=list(_LEGACY_CASES))
    def test_the_history_and_the_weights_equal_those_of_the_2_0_0_loop(self, case: str) -> None:
        options = dict(_LEGACY_CASES[case])
        two = bool(options.pop("two_modalities", False))
        batches = _batches(5, two_modalities=two)
        validation = _batches(2, seed=9, two_modalities=two)

        def run(trainer_class: type[Trainer]) -> Trainer:
            trainer = trainer_class(
                _buildModel(two_modalities=two),
                device="cpu",
                optimizer_kwargs={"lr": 0.05},
                **options,
            )
            torch.manual_seed(11)
            trainer.history_of_run = trainer.fit(  # type: ignore[attr-defined]
                batches, num_epochs=3, val_dataloader=validation
            )
            return trainer

        legacy, current = run(_LegacyTrainer), run(Trainer)

        assert legacy.history_of_run == current.history_of_run  # type: ignore[attr-defined]
        assert legacy.global_step == current.global_step
        for name, tensor in legacy.model.state_dict().items():
            assert torch.equal(tensor, current.model.state_dict()[name]), name

    def test_the_optimizer_and_the_clipping_see_exactly_the_models_parameters(self) -> None:
        trainer = _trainer()
        model_parameters = list(trainer.model.parameters())
        groups = trainer.optimizer.param_groups
        assert len(groups) == 1
        assert len(groups[0]["params"]) == len(model_parameters)
        assert all(a is b for a, b in zip(groups[0]["params"], model_parameters, strict=True))
        trained = trainer._trainedParameters()
        assert all(a is b for a, b in zip(trained, model_parameters, strict=True))
        assert len(trained) == len(model_parameters)

    def test_the_metrics_and_the_checkpoint_have_no_new_key(self, tmp_path: Path) -> None:
        trainer = _trainer()
        history = trainer.fit(_batches(2), num_epochs=1, val_dataloader=_batches(1, seed=9))
        assert {key for key in history[0] if "loss/" in key} == {
            "train/loss/total",
            "train/loss/reconstruction",
            "train/loss/regularization",
            "val/loss/total",
            "val/loss/reconstruction",
            "val/loss/regularization",
        }
        trainer.saveCheckpoint(tmp_path / "c.pt")
        checkpoint = torch.load(tmp_path / "c.pt", weights_only=True)
        assert "loss_terms_state_dict" not in checkpoint
