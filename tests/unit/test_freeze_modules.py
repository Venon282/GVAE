"""The `freeze_modules` callback (`training/callbacks/freeze_modules.py`, roadmap P6-5, ADR 0055).

What is pinned, on a tiny model whose every module has weights **and** `BatchNorm` buffers:

- the arguments: patterns (a non-empty list of non-empty strings), `until_epoch` (`None` or an
  integer of at least 1), the callback registered under `freeze_modules`;
- the guarantee of the roadmap: the parameters **and the buffers** of a frozen module (the running
  mean and variance and the batch counter of its `BatchNorm`) are bit-identical after training,
  whatever the optimizer, its weight decay, gradient clipping, gradient accumulation or a
  validation pass between epochs, while the rest of the model does train, and without the callback
  the same modules do change (so the check is not vacuous);
- the part that is easy to forget: a frozen module is in evaluation mode in every epoch, also the
  ones that follow a validation pass, which leaves the model in evaluation mode;
- the release: frozen in epochs `0 .. until_epoch - 1`, training from `until_epoch`, `requires_grad`
  put back as it was at the end, and a parameter that was already not trainable left alone;
- the resume: whether an epoch is a frozen one is a function of its number, so a run resumed
  before the release epoch and one resumed after it both give the uninterrupted run;
- what it refuses: a pattern that matches no module and a setup that leaves nothing to train, each
  by a message that says so, before the first step.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.freeze_modules import FreezeModules
from global_vae.training.callbacks.registry import getCallbackClass, listRegisteredCallbacks
from global_vae.training.param_groups import ParamGroupSpec
from global_vae.training.trainer import Trainer
from tests.unit._tiny_pair_model import (
    buildTinyModel,
    buildTinyTrainer,
    changedNames,
    namesUnder,
    snapshot,
    tinyBatches,
)

SIGNAL = ["encoders.signal", "decoders.signal"]


class _Spy(TrainerCallback):
    """Records, in the epoch's first and last moments, what the freeze callback left behind.

    It must come **after** `FreezeModules` in the callback list, so that `onEpochBegin` sees the
    state the epoch will train in.
    """

    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.training: dict[int, dict[str, bool]] = {}
        self.trainable: dict[int, dict[str, bool]] = {}
        self.states: dict[int, dict[str, torch.Tensor]] = {}

    def onEpochBegin(self, trainer: Trainer, epoch: int) -> None:
        self.epochs.append(epoch)
        modules = dict(trainer.model.named_modules())
        self.training[epoch] = {
            name: modules[name].training for name in ("encoders.signal", "encoders.image")
        }
        self.trainable[epoch] = {
            name: p.requires_grad for name, p in trainer.model.named_parameters()
        }

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.states[epoch] = snapshot(trainer.model)


def _frozenTrainer(
    modules: list[str] = SIGNAL, until_epoch: int | None = None, **kwargs: Any
) -> tuple[Trainer, _Spy]:
    spy = _Spy()
    callbacks = [FreezeModules(modules, until_epoch=until_epoch), spy]
    return buildTinyTrainer(callbacks=callbacks, **kwargs), spy


class TestArguments:
    def test_it_is_registered_under_its_name(self) -> None:
        assert "freeze_modules" in listRegisteredCallbacks()
        assert getCallbackClass("freeze_modules") is FreezeModules

    def test_the_patterns_are_kept_as_a_tuple_and_until_epoch_defaults_to_never(self) -> None:
        callback = FreezeModules(["decoders.*", "encoders.signal"])
        assert callback.modules == ("decoders.*", "encoders.signal")
        assert callback.until_epoch is None

    @pytest.mark.parametrize("modules", ["encoders.signal", [], [""], [3], None])
    def test_patterns_that_are_not_a_list_of_names_are_refused(self, modules: Any) -> None:
        with pytest.raises(ValueError, match="freeze_modules: `modules`"):
            FreezeModules(modules)

    @pytest.mark.parametrize("until_epoch", [0, -1, 1.5, True, "3"])
    def test_an_until_epoch_that_is_not_an_integer_of_at_least_one_is_refused(
        self, until_epoch: Any
    ) -> None:
        with pytest.raises(ValueError, match="`until_epoch` must be None .* at least 1"):
            FreezeModules(["a"], until_epoch=until_epoch)

    @pytest.mark.parametrize("until_epoch", [1, 5])
    def test_an_integer_of_at_least_one_is_accepted(self, until_epoch: int) -> None:
        assert FreezeModules(["a"], until_epoch=until_epoch).until_epoch == until_epoch

    def test_the_epochs_of_a_freeze_are_the_ones_below_until_epoch(self) -> None:
        forever = FreezeModules(["a"])
        assert all(forever.isFrozenAt(epoch) for epoch in (0, 1, 1000))
        until_three = FreezeModules(["a"], until_epoch=3)
        assert [until_three.isFrozenAt(epoch) for epoch in range(5)] == [True] * 3 + [False] * 2

    def test_it_holds_no_state_to_save(self) -> None:
        callback = FreezeModules(["a"], until_epoch=2)
        assert callback.stateDict() == {}


class TestFrozenModulesAreBitIdentical:
    def test_parameters_and_buffers_of_the_frozen_modules_do_not_change_and_the_rest_trains(
        self,
    ) -> None:
        trainer, _ = _frozenTrainer()
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=3)
        after = snapshot(trainer.model)
        assert changedNames(before, after, "encoders.signal") == []
        assert changedNames(before, after, "decoders.signal") == []
        image = namesUnder(trainer.model, "encoders.image") + namesUnder(
            trainer.model, "decoders.image"
        )
        assert sorted(changedNames(before, after, "")) == sorted(image)

    def test_the_buffers_of_a_frozen_batch_norm_are_among_what_was_checked(self) -> None:
        names = namesUnder(buildTinyModel(), "encoders.signal.norm")
        assert {n.rsplit(".", 1)[-1] for n in names} >= {
            "running_mean",
            "running_var",
            "num_batches_tracked",
        }

    def test_without_the_callback_the_same_modules_change_buffers_included(self) -> None:
        trainer = buildTinyTrainer()
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=3)
        changed = changedNames(before, snapshot(trainer.model), "encoders.signal.norm")
        assert {n.rsplit(".", 1)[-1] for n in changed} >= {
            "weight",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        }

    def test_a_pattern_may_name_a_layer_inside_a_module(self) -> None:
        trainer, _ = _frozenTrainer(["encoders.*.norm"])
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=2)
        changed = changedNames(before, snapshot(trainer.model), "encoders")
        assert not any(".norm." in name for name in changed)
        assert any(name.endswith("hidden.weight") for name in changed)
        assert any(
            ".norm." in name for name in changedNames(before, snapshot(trainer.model), "decoders")
        )

    def test_the_gradients_of_frozen_parameters_are_dropped(self) -> None:
        trainer, _ = _frozenTrainer()
        trainer.fit(tinyBatches(2), num_epochs=1)
        model = trainer.model
        assert all(p.grad is None for p in model.encoders["signal"].parameters())
        assert all(p.grad is not None for p in model.encoders["image"].parameters())

    def test_a_gradient_a_parameter_still_holds_when_it_is_frozen_is_dropped(self) -> None:
        trainer, _ = _frozenTrainer(["encoders.signal"])
        trainer.fit(tinyBatches(1), num_epochs=1)  # leaves the last step's gradients in place
        callback = trainer.callbacks[0]
        assert isinstance(callback, FreezeModules)
        for parameter in trainer.model.parameters():
            parameter.requires_grad_(True)
            parameter.grad = torch.ones_like(parameter)
        callback.onTrainBegin(trainer)
        callback.onEpochBegin(trainer, 0)
        assert all(p.grad is None for p in trainer.model.encoders["signal"].parameters())
        assert all(p.grad is not None for p in trainer.model.encoders["image"].parameters())

    @pytest.mark.parametrize(
        ("optimizer", "kwargs"),
        [
            (torch.optim.Adam, {"lr": 0.01, "weight_decay": 0.5}),
            (torch.optim.AdamW, {"lr": 0.01, "weight_decay": 0.5}),
            (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.5}),
        ],
    )
    def test_no_optimizer_and_no_weight_decay_moves_a_frozen_parameter(
        self, optimizer: type[torch.optim.Optimizer], kwargs: dict[str, Any]
    ) -> None:
        trainer, _ = _frozenTrainer(optimizer=optimizer, optimizer_kwargs=kwargs)
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=2)
        assert changedNames(before, snapshot(trainer.model), "encoders.signal") == []
        assert changedNames(before, snapshot(trainer.model), "encoders.image") != []

    def test_clipping_accumulating_and_a_validation_pass_change_nothing_about_it(self) -> None:
        trainer, _ = _frozenTrainer(grad_clip_norm=0.5, accumulate_grad_batches=2)
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=3, val_dataloader=tinyBatches(2, seed=9))
        after = snapshot(trainer.model)
        assert changedNames(before, after, "encoders.signal") == []
        assert changedNames(before, after, "decoders.signal") == []

    def test_a_learning_rate_group_on_a_frozen_module_does_not_unfreeze_it(self) -> None:
        trainer, _ = _frozenTrainer(param_groups=[ParamGroupSpec(tuple(SIGNAL), 0.5)])
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(2), num_epochs=2)
        assert changedNames(before, snapshot(trainer.model), "encoders.signal") == []

    def test_frozen_modules_still_pass_the_gradient_to_the_modules_before_them(self) -> None:
        """The decoder is frozen and the encoder is not: the encoder learns through it."""
        trainer, _ = _frozenTrainer(["decoders.*"])
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=2)
        after = snapshot(trainer.model)
        assert changedNames(before, after, "decoders") == []
        assert changedNames(before, after, "encoders.signal.to_mu") != []


class TestFrozenModulesAreInEvaluationMode:
    def test_in_every_epoch_even_the_ones_after_a_validation_pass(self) -> None:
        trainer, spy = _frozenTrainer()
        trainer.fit(tinyBatches(2), num_epochs=3, val_dataloader=tinyBatches(1, seed=5))
        assert spy.epochs == [0, 1, 2]
        for epoch in spy.epochs:
            assert spy.training[epoch] == {"encoders.signal": False, "encoders.image": True}

    def test_every_module_inside_a_frozen_module_is_in_evaluation_mode(self) -> None:
        trainer, _ = _frozenTrainer(["encoders.signal"])
        callback = trainer.callbacks[0]
        assert isinstance(callback, FreezeModules)
        callback.onTrainBegin(trainer)
        trainer.model.train()
        callback.onEpochBegin(trainer, 0)
        inside = [m for n, m in trainer.model.named_modules() if n.startswith("encoders.signal")]
        assert inside and not any(m.training for m in inside)
        assert trainer.model.encoders["image"].training

    def test_a_pattern_naming_a_batch_norm_alone_puts_only_it_in_evaluation_mode(self) -> None:
        trainer, _ = _frozenTrainer(["encoders.signal.norm"])
        before = snapshot(trainer.model)
        trainer.fit(tinyBatches(2), num_epochs=1)
        changed = changedNames(before, snapshot(trainer.model), "encoders.signal")
        assert not any(".norm." in name for name in changed)
        assert "encoders.signal.hidden.weight" in changed


class TestTheRelease:
    def test_frozen_in_the_epochs_below_until_epoch_and_training_from_it(self) -> None:
        trainer, spy = _frozenTrainer(until_epoch=2)
        initial = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=4)
        signal = "encoders.signal.to_mu.weight"
        assert torch.equal(spy.states[0][signal], initial[signal])
        assert torch.equal(spy.states[1][signal], initial[signal])
        assert not torch.equal(spy.states[2][signal], initial[signal])
        assert not torch.equal(spy.states[3][signal], spy.states[2][signal])

    def test_requires_grad_and_the_training_mode_follow_the_epoch(self) -> None:
        trainer, spy = _frozenTrainer(until_epoch=2)
        trainer.fit(tinyBatches(2), num_epochs=4)
        flag = "encoders.signal.to_mu.weight"
        assert [spy.trainable[e][flag] for e in range(4)] == [False, False, True, True]
        assert [spy.training[e]["encoders.signal"] for e in range(4)] == [False, False, True, True]

    def test_the_buffers_start_moving_again_when_a_module_is_released(self) -> None:
        trainer, spy = _frozenTrainer(until_epoch=1)
        initial = snapshot(trainer.model)
        trainer.fit(tinyBatches(3), num_epochs=2)
        counter = "encoders.signal.norm.num_batches_tracked"
        assert torch.equal(spy.states[0][counter], initial[counter])
        assert spy.states[1][counter].item() == 3

    def test_the_flags_are_put_back_when_the_run_ends(self) -> None:
        for until_epoch in (None, 2):
            trainer, _ = _frozenTrainer(until_epoch=until_epoch)
            trainer.fit(tinyBatches(1), num_epochs=2)
            assert all(p.requires_grad for p in trainer.model.parameters())

    def test_a_parameter_that_was_not_trainable_to_begin_with_is_left_alone(self) -> None:
        trainer, spy = _frozenTrainer(["encoders.signal"], until_epoch=1)
        model = trainer.model
        kept = model.get_parameter("encoders.signal.to_mu.bias")
        kept.requires_grad_(False)
        trainer.fit(tinyBatches(2), num_epochs=3)
        assert spy.trainable[2]["encoders.signal.to_mu.bias"] is False
        assert spy.trainable[2]["encoders.signal.to_mu.weight"] is True
        assert kept.requires_grad is False
        assert model.get_parameter("encoders.signal.to_mu.weight").requires_grad is True

    def test_a_second_fit_goes_on_from_where_the_first_stopped(self) -> None:
        trainer, spy = _frozenTrainer(until_epoch=2)
        trainer.fit(tinyBatches(2), num_epochs=1)
        trainer.fit(tinyBatches(2), num_epochs=2)
        assert spy.epochs == [0, 1, 2]
        assert spy.trainable[2]["encoders.signal.to_mu.weight"] is True


class TestResume:
    @pytest.mark.parametrize("interrupted_after", [1, 2, 3, 5])
    def test_a_resumed_run_is_the_uninterrupted_one_before_or_after_the_release_epoch(
        self, tmp_path: Path, interrupted_after: int
    ) -> None:
        """`until_epoch=3`: the interruption falls in the frozen epochs, at the edge, or after."""
        total, until = 6, 3

        def build(seed: int) -> tuple[Trainer, _Spy]:
            return _frozenTrainer(until_epoch=until, model=buildTinyModel(seed))

        whole, whole_spy = build(1)
        torch.manual_seed(7)
        whole.fit(tinyBatches(3), num_epochs=total)

        first, _ = build(1)
        torch.manual_seed(7)
        first.fit(tinyBatches(3), num_epochs=interrupted_after)
        first.saveCheckpoint(tmp_path / "ckpt.pt")

        resumed, spy = build(99)
        resumed.loadCheckpoint(tmp_path / "ckpt.pt")
        resumed.fit(tinyBatches(3), num_epochs=total - interrupted_after)

        assert spy.epochs == list(range(interrupted_after, total))
        flag = "encoders.signal.to_mu.weight"
        for epoch in spy.epochs:
            assert spy.trainable[epoch][flag] is whole_spy.trainable[epoch][flag]
        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name], tensor, rtol=1e-6, atol=1e-7, msg=name
            )

    def test_a_frozen_module_is_bit_identical_across_the_resume(self, tmp_path: Path) -> None:
        first, _ = _frozenTrainer()
        initial = snapshot(first.model)
        first.fit(tinyBatches(3), num_epochs=2)
        first.saveCheckpoint(tmp_path / "ckpt.pt")

        resumed, _ = _frozenTrainer(model=buildTinyModel(99))
        resumed.loadCheckpoint(tmp_path / "ckpt.pt")
        resumed.fit(tinyBatches(3), num_epochs=2)
        final = snapshot(resumed.model)
        assert changedNames(initial, final, "encoders.signal") == []
        assert changedNames(initial, final, "decoders.signal") == []
        assert changedNames(initial, final, "encoders.image") != []


class TestWhatItRefuses:
    def test_a_pattern_that_matches_no_module_stops_the_run_before_its_first_step(self) -> None:
        trainer, spy = _frozenTrainer(["encoders.signal", "decoders.signl"])
        before = snapshot(trainer.model)
        with pytest.raises(
            ValueError, match=r"freeze_modules: the pattern 'decoders.signl' matches"
        ):
            trainer.fit(tinyBatches(2), num_epochs=1)
        assert spy.epochs == []
        assert changedNames(before, snapshot(trainer.model)) == []

    def test_modules_that_have_no_parameter_leave_nothing_to_freeze(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        trainer = SimpleNamespace(model=model)
        with pytest.raises(ValueError, match="have no parameter, so there is nothing to freeze"):
            FreezeModules(["1"]).onTrainBegin(trainer)  # type: ignore[arg-type]

    @pytest.mark.parametrize("until_epoch", [None, 1])
    def test_freezing_every_parameter_is_refused_in_the_epoch_it_would_happen(
        self, until_epoch: int | None
    ) -> None:
        trainer, spy = _frozenTrainer(["*"], until_epoch=until_epoch)
        with pytest.raises(ValueError, match=r"in epoch 0, every parameter .* is frozen") as caught:
            trainer.fit(tinyBatches(2), num_epochs=2)
        assert "give `latent_alignment` `adapters`" in str(caught.value)
        assert "ADR 0060" in str(caught.value)
        assert trainer.history == []

    def test_a_run_whose_only_trainable_parameters_are_released_later_is_still_refused_at_the_start(
        self,
    ) -> None:
        trainer, _ = _frozenTrainer(["encoders.*", "decoders.*"], until_epoch=2)
        with pytest.raises(ValueError, match="every parameter .* is frozen"):
            trainer.fit(tinyBatches(2), num_epochs=3)

    def test_a_batch_that_only_the_frozen_side_can_use_says_so_and_the_flags_are_put_back(
        self,
    ) -> None:
        trainer, _ = _frozenTrainer()
        trainer.fit(tinyBatches(2), num_epochs=1)
        only_the_frozen_side = [{"signal": torch.zeros(8, 8)}]
        with pytest.raises(RuntimeError, match="does not depend on any parameter that can train"):
            trainer.fit(only_the_frozen_side, num_epochs=1)
        assert all(p.requires_grad for p in trainer.model.parameters())
