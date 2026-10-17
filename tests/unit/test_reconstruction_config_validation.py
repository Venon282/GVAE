"""Unit tests for the validators of `TrainingConfig.reconstruction_loss` and `_weight`.

Roadmap P2-2, `docs/adr/0030-per-modality-reconstruction-loss-and-weight.md`. Both fields accept
a single value for every modality or a mapping decoder name -> value. The validators need no
model, so they cannot tell whether a key is a decoder (the resolvers, tested in
`tests/integration/test_per_modality_loss_and_weight.py`, do); they check the shape and the
values.
"""

import math
from types import MappingProxyType

import pytest
from omegaconf import OmegaConf

from global_vae.config.training import (
    DEFAULT_RECONSTRUCTION_LOSS,
    DEFAULT_RECONSTRUCTION_WEIGHT,
    TrainingConfig,
    listSupportedReconstructionLossNames,
    validateReconstructionLoss,
    validateReconstructionWeight,
)


class TestDefaults:
    def test_the_defaults_are_the_ones_the_fields_always_had(self) -> None:
        assert DEFAULT_RECONSTRUCTION_LOSS == "mse"
        assert DEFAULT_RECONSTRUCTION_WEIGHT == 1.0

    def test_the_dataclass_fields_use_them(self) -> None:
        config = TrainingConfig()
        assert config.reconstruction_loss == DEFAULT_RECONSTRUCTION_LOSS
        assert config.reconstruction_weight == DEFAULT_RECONSTRUCTION_WEIGHT

    def test_the_default_is_a_supported_loss(self) -> None:
        validateReconstructionLoss(DEFAULT_RECONSTRUCTION_LOSS)
        validateReconstructionWeight(DEFAULT_RECONSTRUCTION_WEIGHT)


class TestValidateReconstructionLoss:
    @pytest.mark.parametrize("name", listSupportedReconstructionLossNames())
    def test_every_supported_name_is_accepted(self, name: str) -> None:
        validateReconstructionLoss(name)
        validateReconstructionLoss({"signal": name})

    def test_a_mapping_of_names_is_accepted(self) -> None:
        validateReconstructionLoss({"signal": "mse", "mask": "bce"})

    def test_an_empty_mapping_is_accepted_and_means_every_decoder_keeps_the_default(self) -> None:
        validateReconstructionLoss({})

    def test_a_read_only_mapping_and_an_omegaconf_mapping_are_accepted(self) -> None:
        validateReconstructionLoss(MappingProxyType({"signal": "l1"}))
        validateReconstructionLoss(OmegaConf.create({"signal": "l1"}))

    def test_an_unknown_name_is_a_key_error_that_lists_the_supported_ones(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            validateReconstructionLoss("huber")
        message = str(excinfo.value.args[0])
        assert message.startswith("training.reconstruction_loss: unknown loss 'huber'")
        for name in listSupportedReconstructionLossNames():
            assert name in message

    def test_an_unknown_name_in_a_mapping_names_its_decoder(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            validateReconstructionLoss({"signal": "mse", "mask": "dice"})
        assert str(excinfo.value.args[0]).startswith(
            "training.reconstruction_loss['mask']: unknown loss 'dice'"
        )

    def test_the_field_name_in_the_message_can_be_changed(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            validateReconstructionLoss({"y": "dice"}, "--reconstruction-loss")
        assert str(excinfo.value.args[0]).startswith("--reconstruction-loss['y']: unknown loss")

    @pytest.mark.parametrize("value", [3, 1.5, None, True, ["mse"], ("mse",)])
    def test_a_value_that_is_neither_a_name_nor_a_mapping_is_a_value_error(
        self, value: object
    ) -> None:
        with pytest.raises(ValueError, match=r"^training\.reconstruction_loss: expected a loss"):
            validateReconstructionLoss(value)

    @pytest.mark.parametrize("name", [3, None, ["mse"], True])
    def test_a_mapping_value_that_is_not_a_string_is_a_value_error(self, name: object) -> None:
        with pytest.raises(ValueError, match=r"^training\.reconstruction_loss\['signal'\]"):
            validateReconstructionLoss({"signal": name})

    def test_a_mapping_key_that_is_not_a_string_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="a modality name must be a string, got 0"):
            validateReconstructionLoss({0: "mse"})

    def test_a_name_with_other_case_or_spaces_is_not_normalised(self) -> None:
        """A config key is exact, like the optimizer names: `MSE` is not `mse`."""
        for text in ("MSE", " mse", "mse "):
            with pytest.raises(KeyError):
                validateReconstructionLoss(text)


class TestValidateReconstructionWeight:
    @pytest.mark.parametrize("weight", [0, 0.0, 1, 0.5, 2.0, 1e-9, 1e9])
    def test_a_finite_number_at_least_zero_is_accepted(self, weight: float) -> None:
        validateReconstructionWeight(weight)
        validateReconstructionWeight({"signal": weight})

    def test_a_mapping_of_numbers_is_accepted(self) -> None:
        validateReconstructionWeight({"signal": 0.5, "image": 2})

    def test_an_empty_mapping_is_accepted_and_means_every_decoder_keeps_the_default(self) -> None:
        validateReconstructionWeight({})

    def test_a_read_only_mapping_and_an_omegaconf_mapping_are_accepted(self) -> None:
        validateReconstructionWeight(MappingProxyType({"signal": 0.5}))
        validateReconstructionWeight(OmegaConf.create({"signal": 0.5}))

    @pytest.mark.parametrize("weight", [-1, -1e-9, -math.inf, math.inf, math.nan])
    def test_a_negative_or_non_finite_number_is_refused(self, weight: float) -> None:
        with pytest.raises(ValueError, match=r"^training\.reconstruction_weight: must be a finite"):
            validateReconstructionWeight(weight)

    def test_the_refused_number_is_named_with_its_decoder_in_a_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"^training\.reconstruction_weight\['image'\]: must"):
            validateReconstructionWeight({"signal": 1.0, "image": -0.5})

    @pytest.mark.parametrize("weight", ["0.5", None, [1.0], True, False, 1 + 2j])
    def test_a_value_that_is_not_a_number_is_refused(self, weight: object) -> None:
        """A `bool` is an `int` to Python, and `reconstruction_weight=true` is a typo, not 1."""
        with pytest.raises(
            ValueError, match=r"^training\.reconstruction_weight: expected a number"
        ):
            validateReconstructionWeight(weight)

    def test_a_string_inside_a_mapping_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\['signal'\]: expected a number, got '0.5'"):
            validateReconstructionWeight({"signal": "0.5"})

    def test_a_mapping_key_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a modality name must be a string, got 1"):
            validateReconstructionWeight({1: 0.5})

    def test_the_field_name_in_the_message_can_be_changed(self) -> None:
        with pytest.raises(ValueError, match=r"^--reconstruction-weight: must be"):
            validateReconstructionWeight(-1.0, "--reconstruction-weight")
