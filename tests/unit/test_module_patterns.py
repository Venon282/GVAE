"""Selecting modules of a model by name patterns (`training/module_patterns.py`, roadmap P6-5).

`freeze_modules` and `training.param_groups` read module names the same way, so the rule is pinned
here once: a shell-style pattern against the whole dotted name, `*` standing for any characters
dots included, a module standing for its own parameters and those of every module inside it, and
a pattern that matches nothing being an error.
"""

from typing import Any

import pytest
from omegaconf import OmegaConf
from torch import nn

from global_vae.training.module_patterns import checkPatterns, matchModules, parametersOf


class _Model(nn.Module):
    """`encoders.{signal,image}`, `decoders.{signal,image}` (each a small Sequential) and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3)) for m in ("signal", "image")}
        )
        self.decoders = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(3, 2)) for m in ("signal", "image")}
        )
        self.head = nn.Linear(3, 1)
        self.activation = nn.ReLU()


class TestCheckPatterns:
    def test_a_list_is_returned_as_a_tuple_in_order(self) -> None:
        assert checkPatterns(["b", "a"], "w") == ("b", "a")

    def test_an_omegaconf_list_and_a_tuple_are_accepted(self) -> None:
        assert checkPatterns(OmegaConf.create(["encoders.*"]), "w") == ("encoders.*",)
        assert checkPatterns(("a",), "w") == ("a",)

    @pytest.mark.parametrize("patterns", ["encoders.signal", None, 3, {"a": 1}])
    def test_anything_that_is_not_a_list_is_refused_and_named(self, patterns: Any) -> None:
        with pytest.raises(ValueError, match="my field must be a list of module name patterns"):
            checkPatterns(patterns, "my field")

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="my field is empty"):
            checkPatterns([], "my field")

    @pytest.mark.parametrize("entry", ["", None, 4, ["a"]])
    def test_an_entry_that_is_not_a_non_empty_string_is_refused_by_index(self, entry: Any) -> None:
        with pytest.raises(ValueError, match=r"my field\[1\] must be a non-empty module name"):
            checkPatterns(["a", entry], "my field")


class TestMatchModules:
    def test_a_dotted_name_selects_that_module_and_every_module_inside_it(self) -> None:
        matched = matchModules(_Model(), ["encoders.signal"], "w")
        assert list(matched) == ["encoders.signal"]

    def test_a_star_stands_for_dots_too(self) -> None:
        names = set(matchModules(_Model(), ["decoders.*"], "w"))
        assert names == {
            "decoders.signal",
            "decoders.signal.0",
            "decoders.image",
            "decoders.image.0",
        }

    def test_the_container_has_no_dot_after_it_so_decoders_star_leaves_it_out(self) -> None:
        assert "decoders" not in matchModules(_Model(), ["decoders.*"], "w")
        assert "decoders" in matchModules(_Model(), ["decoders"], "w")

    def test_a_star_alone_is_the_whole_model_root_included(self) -> None:
        model = _Model()
        matched = matchModules(model, ["*"], "w")
        assert matched[""] is model
        assert len(matched) == len(list(model.named_modules()))

    def test_a_star_in_the_middle_selects_a_layer_of_every_modality(self) -> None:
        assert set(matchModules(_Model(), ["encoders.*.1"], "w")) == {
            "encoders.signal.1",
            "encoders.image.1",
        }

    def test_a_pattern_is_matched_against_the_whole_name_and_is_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="matches no module"):
            matchModules(_Model(), ["signal"], "w")
        with pytest.raises(ValueError, match="matches no module"):
            matchModules(_Model(), ["Encoders.signal"], "w")

    def test_several_patterns_are_a_union_each_module_once_in_model_order(self) -> None:
        model = _Model()
        matched = matchModules(model, ["head", "encoders.image", "encoders.*"], "w")
        assert list(matched) == [name for name, _ in model.named_modules() if name in matched]
        assert len(matched) == len(set(matched))
        assert "head" in matched and "encoders.signal.0" in matched

    def test_the_modules_returned_are_the_modules_of_the_model(self) -> None:
        model = _Model()
        assert matchModules(model, ["head"], "w")["head"] is model.head

    def test_a_pattern_that_matches_nothing_is_an_error_that_names_it_and_lists_real_names(
        self,
    ) -> None:
        with pytest.raises(ValueError) as caught:
            matchModules(_Model(), ["encoders.signal", "decodrs.*"], "freeze_modules")
        message = str(caught.value)
        assert message.startswith("freeze_modules: the pattern 'decodrs.*' matches no module")
        assert "encoders.signal" in message and "decoders.image" in message

    def test_only_the_first_names_are_listed_when_there_are_many(self) -> None:
        model = nn.Sequential(*[nn.Linear(1, 1) for _ in range(30)])
        with pytest.raises(ValueError, match=r"modules: 0, 1, 2.*, \.\.\."):
            matchModules(model, ["nope"], "w")


class TestParametersOf:
    def test_a_module_stands_for_its_parameters_and_those_inside_it(self) -> None:
        model = _Model()
        parameters = parametersOf([model.encoders["signal"]])
        assert len(parameters) == 4  # Linear weight and bias, BatchNorm weight and bias
        assert {id(p) for p in parameters} == {id(p) for p in model.encoders["signal"].parameters()}

    def test_a_module_inside_another_adds_nothing(self) -> None:
        model = _Model()
        assert len(parametersOf([model.encoders, model.encoders["signal"]])) == 8

    def test_a_shared_parameter_is_listed_once(self) -> None:
        layer = nn.Linear(2, 2)
        holder = nn.ModuleList([layer, layer])
        assert len(parametersOf(holder)) == 2

    def test_a_module_without_a_parameter_gives_nothing_and_the_order_is_kept(self) -> None:
        model = _Model()
        assert parametersOf([model.activation]) == []
        both = parametersOf([model.head, model.decoders["signal"]])
        assert both[0] is model.head.weight
