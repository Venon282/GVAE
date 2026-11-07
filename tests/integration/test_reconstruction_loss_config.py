"""The forms of `training.reconstruction_loss` and what they build (roadmap P5-3, ADR 0048).

A YAML file names a registered loss in one of three forms: a bare name, an entry `{name, kwargs}`
or a mapping reconstruction -> name or entry. These tests resolve each against real models, through
Hydra overrides and through OmegaConf containers, and check the errors that tell a mistyped name
from a valid one.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    buildTrainerFromConfig,
    listSupportedReconstructionLossNames,
    resolveReconstructionLoss,
    resolveReconstructionLossFn,
    validateReconstructionLoss,
)
from global_vae.losses.reconstruction import listRegisteredReconstructionLosses
from global_vae.losses.reconstruction.mse import MseLoss
from global_vae.losses.reconstruction.smooth_l1 import SmoothL1Loss
from global_vae.losses.reconstruction.tabular_mixed import TabularMixedLoss
from global_vae.models.global_vae import GlobalVae
from tests.integration._script_fixtures import buildTwoModalityModelForScript

_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]


def _tabularModel(sizes: list[int] | None = None) -> GlobalVae:
    config = loadExperimentConfig("experiment/tabular_vae", overrides=_DATA_OVERRIDES)
    if sizes is not None:
        config.model.modalities["tabular"].decoder.kwargs["categorical_sizes"] = sizes
    return buildModelFromConfig(config.model)


def _twoModels() -> GlobalVae:
    return buildTwoModalityModelForScript()


def _modelWithAReconstructionCalledName() -> GlobalVae:
    """Only `output_modalities` and `decoders` are read to resolve a loss."""
    return cast(GlobalVae, SimpleNamespace(output_modalities=("name",), decoders={}))


class TestTheNamesOfTheRegistry:
    def test_the_config_lists_the_registered_losses(self) -> None:
        assert listSupportedReconstructionLossNames() == listRegisteredReconstructionLosses()
        assert "masked_mse" in listSupportedReconstructionLossNames()

    def test_the_four_names_of_2_0_0_are_still_there(self) -> None:
        assert {"mse", "l1", "bce", "smooth_l1"} <= set(listSupportedReconstructionLossNames())

    def test_a_name_builds_a_new_instance_every_time(self) -> None:
        assert resolveReconstructionLossFn("mse") is not resolveReconstructionLossFn("mse")

    def test_the_arguments_reach_the_constructor(self) -> None:
        loss = resolveReconstructionLossFn("smooth_l1", beta=0.25)
        assert isinstance(loss, SmoothL1Loss)
        assert loss.beta == 0.25

    def test_an_unknown_name_lists_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown loss 'huber'.*Available: .*smooth_l1"):
            validateReconstructionLoss("huber")


class TestTheThreeForms:
    def test_a_bare_name_is_one_instance_shared_by_every_reconstruction(self) -> None:
        loss = resolveReconstructionLoss("smooth_l1", _twoModels())
        assert isinstance(loss, SmoothL1Loss)

    def test_an_entry_is_one_loss_with_its_arguments(self) -> None:
        loss = resolveReconstructionLoss(
            {"name": "smooth_l1", "kwargs": {"beta": 0.3}}, _twoModels()
        )
        assert isinstance(loss, SmoothL1Loss)
        assert loss.beta == 0.3

    def test_an_entry_may_leave_kwargs_out(self) -> None:
        assert isinstance(resolveReconstructionLoss({"name": "mse"}, _twoModels()), MseLoss)

    def test_a_mapping_gives_each_reconstruction_its_own_instance(self) -> None:
        losses = resolveReconstructionLoss({"x": "l1", "y": "l1"}, _twoModels())
        assert isinstance(losses, dict)
        assert losses["x"] is not losses["y"]

    def test_a_mapping_takes_names_and_entries_together(self) -> None:
        losses = resolveReconstructionLoss(
            {"x": {"name": "smooth_l1", "kwargs": {"beta": 0.1}}, "y": "mse"}, _twoModels()
        )
        assert isinstance(losses, dict)
        assert isinstance(losses["x"], SmoothL1Loss) and losses["x"].beta == 0.1
        assert isinstance(losses["y"], MseLoss)

    def test_a_reconstruction_the_mapping_leaves_out_gets_mse(self) -> None:
        losses = resolveReconstructionLoss({"x": "l1"}, _twoModels())
        assert isinstance(losses, dict)
        assert isinstance(losses["y"], MseLoss)

    def test_a_mapping_of_an_entry_for_each_of_two_reconstructions_is_not_an_entry(self) -> None:
        """Keys `name` and `kwargs` alone make an entry. Any other key makes a mapping."""
        losses = resolveReconstructionLoss({"x": "mse", "y": "l1"}, _twoModels())
        assert isinstance(losses, dict)
        assert set(losses) == {"x", "y"}

    def test_a_mapping_naming_a_reconstruction_the_model_lacks_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"names decoder\(s\) \['z'\]"):
            resolveReconstructionLoss({"z": "mse"}, _twoModels())


class TestAnEntryThatCouldBeAMapping:
    """A model with a reconstruction called `name` makes `{name: l1}` mean two things."""

    def test_it_is_refused_with_the_way_out(self) -> None:
        with pytest.raises(ValueError, match="ambiguous.*kwargs: \\{\\}"):
            resolveReconstructionLoss({"name": "l1"}, _modelWithAReconstructionCalledName())

    def test_writing_kwargs_makes_it_an_entry(self) -> None:
        loss = resolveReconstructionLoss(
            {"name": "l1", "kwargs": {}}, _modelWithAReconstructionCalledName()
        )
        assert type(loss).__name__ == "L1Loss"

    def test_another_model_has_no_ambiguity(self) -> None:
        assert type(resolveReconstructionLoss({"name": "l1"}, _twoModels())).__name__ == "L1Loss"


class TestMistakes:
    @pytest.mark.parametrize("value", [3, ["mse"], None, 1.5])
    def test_a_value_of_no_form_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="expected a loss name"):
            validateReconstructionLoss(value)

    def test_an_entry_whose_name_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected a loss name or an entry"):
            validateReconstructionLoss({"name": 3})

    def test_kwargs_must_be_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="kwargs must be a mapping"):
            validateReconstructionLoss({"name": "smooth_l1", "kwargs": [0.3]})

    def test_a_mapping_value_that_is_not_a_loss_is_refused_with_its_key(self) -> None:
        with pytest.raises(ValueError, match=r"reconstruction_loss\['x'\]"):
            validateReconstructionLoss({"x": 3})

    def test_an_argument_the_loss_does_not_have_is_a_value_error_naming_the_loss(self) -> None:
        with pytest.raises(ValueError, match="cannot build loss 'mse'"):
            resolveReconstructionLoss({"name": "mse", "kwargs": {"beta": 1.0}}, _twoModels())


class TestOmegaConfContainers:
    """A YAML file reaches the resolver as `DictConfig` and `ListConfig`, not as dict and list."""

    def test_an_entry_of_a_yaml_file_builds_with_plain_arguments(self) -> None:
        value = OmegaConf.create({"name": "smooth_l1", "kwargs": {"beta": 0.4}})
        loss = resolveReconstructionLoss(value, _twoModels())
        assert isinstance(loss, SmoothL1Loss)
        assert loss.beta == 0.4 and type(loss.beta) is float

    def test_a_list_argument_becomes_a_list(self) -> None:
        value = OmegaConf.create({"name": "tabular_mixed", "kwargs": {"categorical_sizes": [3, 2]}})
        loss = resolveReconstructionLoss(value, _twoModels())
        assert isinstance(loss, TabularMixedLoss)
        assert type(loss.categorical_sizes) in (list, tuple)
        assert list(loss.categorical_sizes) == [3, 2]

    def test_a_mapping_of_a_yaml_file_resolves(self) -> None:
        value = OmegaConf.create({"x": "l1", "y": {"name": "smooth_l1", "kwargs": {"beta": 0.2}}})
        losses = resolveReconstructionLoss(value, _twoModels())
        assert isinstance(losses, dict)
        assert type(losses["x"]).__name__ == "L1Loss"
        assert isinstance(losses["y"], SmoothL1Loss) and losses["y"].beta == 0.2


class TestHydraOverrides:
    def _trainer(self, tmp_path: Path, *overrides: str) -> Any:
        config = loadExperimentConfig(
            "experiment/time_series_vae",
            overrides=[*_DATA_OVERRIDES, f"output_dir={tmp_path}", *overrides],
        )
        return buildTrainerFromConfig(buildModelFromConfig(config.model), config.training)

    def test_a_name_on_the_command_line_replaces_the_one_of_the_file(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path, "training.reconstruction_loss=l1")
        assert type(trainer.reconstruction_loss_fn).__name__ == "L1Loss"

    def test_an_entry_on_the_command_line_carries_its_arguments(self, tmp_path: Path) -> None:
        trainer = self._trainer(
            tmp_path, "training.reconstruction_loss={name:smooth_l1,kwargs:{beta:0.25}}"
        )
        assert isinstance(trainer.reconstruction_loss_fn, SmoothL1Loss)
        assert trainer.reconstruction_loss_fn.beta == 0.25

    def test_a_mapping_on_the_command_line_names_the_reconstruction(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path, "training.reconstruction_loss={time_series:l1}")
        assert isinstance(trainer.reconstruction_loss_fn, dict)
        assert type(trainer.reconstruction_loss_fn["time_series"]).__name__ == "L1Loss"

    def test_a_wrong_name_stops_the_build_with_the_available_ones(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="Available: .*masked_mse"):
            self._trainer(tmp_path, "training.reconstruction_loss=rmse")


class TestWhatALossReadsFromItsDecoder:
    def test_tabular_mixed_takes_the_categorical_sizes_of_the_decoder(self) -> None:
        loss = resolveReconstructionLoss("tabular_mixed", _tabularModel([3, 4]))
        assert isinstance(loss, TabularMixedLoss)
        assert list(loss.categorical_sizes) == [3, 4]

    def test_a_table_with_no_categorical_column_gives_a_loss_with_none(self) -> None:
        loss = resolveReconstructionLoss("tabular_mixed", _tabularModel())
        assert isinstance(loss, TabularMixedLoss)
        assert list(loss.categorical_sizes) == []

    def test_the_kwargs_of_the_file_win_over_the_decoder(self) -> None:
        loss = resolveReconstructionLoss(
            {"name": "tabular_mixed", "kwargs": {"categorical_sizes": [2]}}, _tabularModel([3, 4])
        )
        assert isinstance(loss, TabularMixedLoss)
        assert list(loss.categorical_sizes) == [2]

    def test_a_mapping_reads_it_from_the_decoder_of_that_reconstruction(self) -> None:
        loss = resolveReconstructionLoss({"tabular": "tabular_mixed"}, _tabularModel([5]))
        assert isinstance(loss, dict)
        tabular = loss["tabular"]
        assert isinstance(tabular, TabularMixedLoss)
        assert list(tabular.categorical_sizes) == [5]

    def test_one_loss_for_several_reconstructions_cannot_read_a_single_decoder(self) -> None:
        with pytest.raises(ValueError, match=r"reads \['categorical_sizes'\] from the decoder"):
            resolveReconstructionLoss("tabular_mixed", _twoModels())

    def test_the_same_loss_with_its_sizes_written_out_is_accepted_for_several(self) -> None:
        loss = resolveReconstructionLoss(
            {"name": "tabular_mixed", "kwargs": {"categorical_sizes": [3]}}, _twoModels()
        )
        assert isinstance(loss, TabularMixedLoss)

    def test_the_experiment_of_the_table_builds_with_the_sizes_of_its_decoder(
        self, tmp_path: Path
    ) -> None:
        config = loadExperimentConfig(
            "experiment/tabular_vae",
            overrides=[
                *_DATA_OVERRIDES,
                f"output_dir={tmp_path}",
                "+model.modalities.tabular.decoder.kwargs.categorical_sizes=[3,3]",
            ],
        )
        trainer = buildTrainerFromConfig(buildModelFromConfig(config.model), config.training)
        assert isinstance(trainer.reconstruction_loss_fn, TabularMixedLoss)
        assert list(trainer.reconstruction_loss_fn.categorical_sizes) == [3, 3]
