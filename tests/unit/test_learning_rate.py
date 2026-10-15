"""Unit tests for `training.learning_rate` and the `acts_on_learning_rate` declaration (ADR 0028).

The helpers are checked on a bare torch optimizer, with no `Trainer`: how the key is spelled for
one parameter group and for several, which keys count as a learning rate, and which built-in
callbacks declare that they act on it. What `Trainer` does with the declaration (history, log line,
loggers, checkpoint) is in `tests/integration/test_learning_rate_logging.py`.
"""

import pytest
import torch
from torch import nn

import global_vae.training.callbacks  # noqa: F401  (registers the built-in callbacks)
import global_vae.training.loggers  # noqa: F401  (registers the built-in loggers)
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.callbacks.registry import getCallbackClass, listRegisteredCallbacks
from global_vae.training.learning_rate import (
    LEARNING_RATE_KEY,
    isLearningRateKey,
    learningRateMetrics,
)
from global_vae.training.loggers.registry import getLoggerClass, listRegisteredLoggers


def _twoGroupOptimizer() -> torch.optim.Optimizer:
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    return torch.optim.SGD(
        [{"params": first.parameters(), "lr": 0.1}, {"params": second.parameters(), "lr": 0.01}]
    )


class TestLearningRateMetrics:
    """`learningRateMetrics` reads the rate the next `optimizer.step()` uses."""

    def test_one_parameter_group_gives_a_single_lr_key(self) -> None:
        optimizer = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=3e-4)
        assert learningRateMetrics(optimizer) == {"lr": 3e-4}
        assert LEARNING_RATE_KEY == "lr"

    def test_several_parameter_groups_give_one_key_each_in_order(self) -> None:
        metrics = learningRateMetrics(_twoGroupOptimizer())
        assert metrics == {"lr/group0": 0.1, "lr/group1": 0.01}
        assert list(metrics) == ["lr/group0", "lr/group1"]

    def test_the_value_is_read_at_call_time(self) -> None:
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.5)
        optimizer.param_groups[0]["lr"] = 0.05
        assert learningRateMetrics(optimizer) == {"lr": 0.05}

    def test_a_tensor_learning_rate_becomes_a_float(self) -> None:
        """Some optimizers accept a 0-d tensor as `lr`; a metric is always a plain float."""
        optimizer = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=torch.tensor(1e-3))
        value = learningRateMetrics(optimizer)["lr"]
        assert type(value) is float
        assert value == pytest.approx(1e-3)


class TestIsLearningRateKey:
    """Any key with an `lr` path component is a learning rate, and nothing else is."""

    @pytest.mark.parametrize(
        "key", ["lr", "lr/group0", "lr/group12", "train/lr", "train/lr/group1"]
    )
    def test_learning_rate_keys(self, key: str) -> None:
        assert isLearningRateKey(key)

    @pytest.mark.parametrize(
        "key",
        [
            "loss/total",
            "train/loss/total",
            "val/loss/regularization",
            "train/lrate",
            "train/loss/lr_like",
            "lrs",
            "",
        ],
    )
    def test_other_keys(self, key: str) -> None:
        assert not isLearningRateKey(key)

    def test_every_key_learningRateMetrics_writes_is_recognised(self) -> None:
        single = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        for optimizer in (single, _twoGroupOptimizer()):
            for key in learningRateMetrics(optimizer):
                assert isLearningRateKey(key)
                assert isLearningRateKey(f"train/{key}")


class TestDeclaration:
    """`TrainerCallback.acts_on_learning_rate` is `False` unless a callback says otherwise."""

    def test_the_base_class_does_not_act_on_the_learning_rate(self) -> None:
        assert TrainerCallback.acts_on_learning_rate is False
        assert TrainerCallback().acts_on_learning_rate is False

    def test_reduce_lr_on_plateau_declares_it(self) -> None:
        assert ReduceLrOnPlateau.acts_on_learning_rate is True
        assert getCallbackClass("reduce_lr_on_plateau") is ReduceLrOnPlateau

    @pytest.mark.parametrize("name", ["checkpoint", "best_checkpoint", "early_stopping"])
    def test_callbacks_that_leave_the_optimizer_alone_do_not(self, name: str) -> None:
        assert name in listRegisteredCallbacks()
        assert getCallbackClass(name).acts_on_learning_rate is False

    def test_every_logger_leaves_the_learning_rate_alone(self) -> None:
        """A logger only records what it is given, so it must not turn the recording on."""
        assert listRegisteredLoggers()
        for name in listRegisteredLoggers():
            assert getLoggerClass(name).acts_on_learning_rate is False, name

    def test_a_subclass_can_declare_it(self) -> None:
        class _Warmup(TrainerCallback):
            acts_on_learning_rate = True

        assert _Warmup.acts_on_learning_rate is True
        assert TrainerCallback.acts_on_learning_rate is False
