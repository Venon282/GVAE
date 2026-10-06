"""Integration tests for `training.callbacks.registry` (spec §10, ADR 0022).

Covers the registry mechanism itself: registration, lookup, the duplicate/
unknown-name error paths (spec §10's own testing checklist item, "Unit tests
for every self-registration registry"), and that every built-in callback
(`training/callbacks/__init__.py`'s own import side effect) actually ends up
registered under the name its module docstring/config examples advertise.
Each concrete callback's own behavior is tested in its own file
(`test_checkpoint_callbacks.py`, `test_early_stopping.py`,
`test_reduce_lr_on_plateau.py`).

Experiment loggers (`training/loggers/`) are a separate, pre-existing
subsystem with their own registry (`training.loggers.registry`, covered by
`test_loggers.py`), not part of this one: a `Logger` is a journalling
service, not a `TrainerCallback` *strategy* selected from this registry,
even though `AbstractExperimentLogger` happens to subclass `TrainerCallback`
for the hook plumbing (ADR 0008). See `docs/adr/0022-callback-registry.md`
for why the two stay apart.
"""

import pytest

import global_vae.training.callbacks  # noqa: F401  (registers every built-in callback)
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.best_checkpoint import BestCheckpointCallback
from global_vae.training.callbacks.checkpoint import CheckpointCallback
from global_vae.training.callbacks.early_stopping import EarlyStopping
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.callbacks.registry import (
    getCallbackClass,
    listRegisteredCallbacks,
    registerCallback,
)


class TestBuiltInCallbacksAreRegistered:
    """Every built-in callback registers under the name its own docs advertise."""

    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [
            ("checkpoint", CheckpointCallback),
            ("best_checkpoint", BestCheckpointCallback),
            ("early_stopping", EarlyStopping),
            ("reduce_lr_on_plateau", ReduceLrOnPlateau),
        ],
    )
    def test_registered_under_expected_name(self, name, expected_cls) -> None:
        assert name in listRegisteredCallbacks()
        assert getCallbackClass(name) is expected_cls

    def test_list_contains_exactly_the_expected_names_at_minimum(self) -> None:
        expected = {
            "checkpoint",
            "best_checkpoint",
            "early_stopping",
            "reduce_lr_on_plateau",
        }
        assert expected <= set(listRegisteredCallbacks())

    def test_loggers_are_not_registered_here(self) -> None:
        """A logger is not a callback *selection*: see this module's docstring."""
        assert "csv" not in listRegisteredCallbacks()
        assert "tensorboard" not in listRegisteredCallbacks()


class TestRegistryMechanism:
    def test_unknown_callback_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="does_not_exist"):
            getCallbackClass("does_not_exist")

    def test_duplicate_registration_raises_value_error(self) -> None:
        @registerCallback("dummy_callback_duplicate_check")
        class _First(TrainerCallback):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registerCallback("dummy_callback_duplicate_check")
            class _Second(TrainerCallback):
                pass

    def test_list_registered_callbacks_is_sorted(self) -> None:
        names = listRegisteredCallbacks()
        assert names == sorted(names)

    def test_registered_class_is_returned_unchanged(self) -> None:
        @registerCallback("dummy_callback_identity_check")
        class _Dummy(TrainerCallback):
            pass

        assert getCallbackClass("dummy_callback_identity_check") is _Dummy
