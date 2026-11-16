"""Learning-rate groups by module: `training.param_groups` (roadmap P6-5, ADR 0055).

What is pinned:

- the entry (`ParamGroupSpec`): patterns, a factor above zero (zero is refused with a pointer to
  `freeze_modules`), a name that is not `default`;
- the split (`buildParamGroups`): a partition of the parameters, a parameter going to the first
  entry that matches it, the parameters no entry names (and those of the loss terms) in one last
  group named `default`, and an entry with nothing of its own refused;
- the optimizer (`Trainer(param_groups=...)`): one group per entry at `lr_factor` times the rate
  (the optimizer's default rate included), the default of no groups unchanged, an optimizer
  instance refused, and a schedule callback keeping the factors;
- the checkpoint: the groups reach the file with their names and factors, a resume with other
  groups is refused by a message that says why, and one with the same groups gives the run the
  uninterrupted one would have;
- the config (`ParamGroupConfig` in `TrainingConfig`): built into specs, an invalid entry named by
  its index, and `buildTrainerFromConfig` passing them on.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from global_vae.config.training import (
    ParamGroupConfig,
    TrainingConfig,
    buildParamGroupSpecs,
    buildTrainerFromConfig,
)
from global_vae.losses.terms.base import AbstractLossTerm
from global_vae.losses.terms.registry import registerLossTerm
from global_vae.training.callbacks.cosine_lr import CosineLr
from global_vae.training.param_groups import (
    DEFAULT_GROUP_NAME,
    ParamGroupSpec,
    applyLrFactors,
    buildParamGroups,
)
from global_vae.training.trainer import Trainer
from tests.unit._tiny_pair_model import (
    buildTinyModel,
    buildTinyTrainer,
    namesUnder,
    tinyBatches,
)

SIGNAL = ("encoders.signal", "decoders.signal")
IMAGE = ("encoders.image", "decoders.image")


def _ids(parameters: Any) -> set[int]:
    return {id(p) for p in parameters}


class TestParamGroupSpec:
    def test_a_list_of_patterns_is_kept_as_a_tuple_and_the_label_is_the_patterns_joined(
        self,
    ) -> None:
        spec = ParamGroupSpec(modules=["encoders.signal", "decoders.*"], lr_factor=0.1)  # type: ignore[arg-type]
        assert spec.modules == ("encoders.signal", "decoders.*")
        assert spec.label == "encoders.signal+decoders.*"

    def test_a_name_replaces_the_label(self) -> None:
        assert ParamGroupSpec(("a",), 0.5, name="slow").label == "slow"

    @pytest.mark.parametrize("factor", [0.1, 1.0, 2, 1e-6])
    def test_a_finite_factor_above_zero_is_accepted(self, factor: float) -> None:
        assert ParamGroupSpec(("a",), factor).lr_factor == factor

    @pytest.mark.parametrize("factor", [0.0, -0.1, float("nan"), float("inf"), True, "0.1", None])
    def test_any_other_factor_is_refused(self, factor: Any) -> None:
        with pytest.raises(ValueError, match="lr_factor must be a finite number above zero"):
            ParamGroupSpec(("a",), factor)

    def test_the_refusal_of_a_zero_factor_points_to_the_callback_that_freezes(self) -> None:
        with pytest.raises(ValueError, match="freeze_modules"):
            ParamGroupSpec(("a",), 0.0)

    @pytest.mark.parametrize("patterns", [[], "encoders.signal", [""], None])
    def test_bad_patterns_are_refused(self, patterns: Any) -> None:
        with pytest.raises(ValueError, match="param_groups modules"):
            ParamGroupSpec(patterns, 0.5)

    @pytest.mark.parametrize("name", ["", 3])
    def test_a_name_that_is_not_a_non_empty_string_is_refused(self, name: Any) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            ParamGroupSpec(("a",), 0.5, name=name)

    def test_the_name_of_the_remainder_group_is_reserved(self) -> None:
        with pytest.raises(ValueError, match=f"'{DEFAULT_GROUP_NAME}' is the name of the group"):
            ParamGroupSpec(("a",), 0.5, name=DEFAULT_GROUP_NAME)

    def test_an_entry_cannot_be_changed(self) -> None:
        spec = ParamGroupSpec(("a",), 0.5)
        with pytest.raises(AttributeError):
            spec.lr_factor = 1.0  # type: ignore[misc]


class TestBuildParamGroups:
    def test_an_entry_gets_the_parameters_of_its_modules_and_the_rest_go_to_default(self) -> None:
        model = buildTinyModel()
        groups = buildParamGroups(model, [ParamGroupSpec(SIGNAL, 0.1)])
        assert [g["name"] for g in groups] == ["encoders.signal+decoders.signal", "default"]
        signal = [p for n, p in model.named_parameters() if ".signal." in f".{n}"]
        assert _ids(groups[0]["params"]) == _ids(signal)
        assert _ids(groups[1]["params"]) == _ids(model.parameters()) - _ids(signal)
        assert [g["lr_factor"] for g in groups] == [0.1, 1.0]

    def test_a_group_has_no_lr_so_the_optimizer_s_own_rate_applies(self) -> None:
        groups = buildParamGroups(buildTinyModel(), [ParamGroupSpec(SIGNAL, 0.1)])
        assert all("lr" not in group for group in groups)

    def test_the_groups_partition_the_parameters_each_exactly_once(self) -> None:
        model = buildTinyModel()
        groups = buildParamGroups(
            model,
            [ParamGroupSpec(("encoders.*",), 0.1), ParamGroupSpec(("decoders.signal",), 0.5)],
        )
        listed = [id(p) for g in groups for p in g["params"]]
        assert len(listed) == len(set(listed))
        assert set(listed) == _ids(model.parameters())

    def test_the_entries_keep_the_order_given_and_the_default_group_is_last(self) -> None:
        groups = buildParamGroups(
            buildTinyModel(),
            [
                ParamGroupSpec(("encoders.image",), 0.3, name="image"),
                ParamGroupSpec(("encoders.signal",), 0.2, name="signal"),
            ],
        )
        assert [g["name"] for g in groups] == ["image", "signal", "default"]

    def test_a_parameter_belongs_to_the_first_entry_that_matches_it(self) -> None:
        model = buildTinyModel()
        groups = buildParamGroups(
            model,
            [
                ParamGroupSpec(("encoders.signal",), 0.1, name="first"),
                ParamGroupSpec(("encoders.*",), 0.5, name="second"),
            ],
        )
        by_name = {g["name"]: g for g in groups}
        assert _ids(by_name["first"]["params"]) == _ids(model.encoders["signal"].parameters())
        assert _ids(by_name["second"]["params"]) == _ids(model.encoders["image"].parameters())

    def test_when_the_entries_take_everything_there_is_no_default_group(self) -> None:
        groups = buildParamGroups(buildTinyModel(), [ParamGroupSpec(("*",), 0.5)])
        assert [g["name"] for g in groups] == ["*"]

    def test_no_entry_is_one_default_group_of_every_parameter(self) -> None:
        model = buildTinyModel()
        groups = buildParamGroups(model, [])
        assert len(groups) == 1 and groups[0]["name"] == "default"
        assert _ids(groups[0]["params"]) == _ids(model.parameters())

    def test_extra_parameters_go_to_the_default_group_once(self) -> None:
        model = buildTinyModel()
        extra = nn.Parameter(torch.zeros(2))
        shared = model.get_parameter("encoders.signal.hidden.weight")
        groups = buildParamGroups(model, [ParamGroupSpec(SIGNAL, 0.1)], [extra, shared])
        default = groups[-1]
        assert any(p is extra for p in default["params"])
        assert not any(p is shared for p in default["params"])  # the signal entry holds it
        assert sum(p is extra for g in groups for p in g["params"]) == 1

    def test_the_same_name_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"name\(s\) \['x'\] are used by more than one entry"):
            buildParamGroups(
                buildTinyModel(),
                [ParamGroupSpec(SIGNAL, 0.1, name="x"), ParamGroupSpec(IMAGE, 0.1, name="x")],
            )

    def test_the_same_patterns_twice_is_refused_as_the_same_name(self) -> None:
        with pytest.raises(ValueError, match="more than one entry"):
            buildParamGroups(
                buildTinyModel(), [ParamGroupSpec(SIGNAL, 0.1), ParamGroupSpec(SIGNAL, 0.2)]
            )

    def test_a_pattern_that_matches_no_module_is_refused_with_the_entry_named(self) -> None:
        with pytest.raises(ValueError) as caught:
            buildParamGroups(
                buildTinyModel(),
                [ParamGroupSpec(SIGNAL, 0.1), ParamGroupSpec(("encoders.imge",), 0.1)],
            )
        assert "training.param_groups[1] ('encoders.imge')" in str(caught.value)
        assert "matches no module" in str(caught.value)

    def test_an_entry_whose_parameters_an_earlier_one_took_is_refused_and_says_whose(self) -> None:
        with pytest.raises(ValueError) as caught:
            buildParamGroups(
                buildTinyModel(),
                [
                    ParamGroupSpec(("encoders.*",), 0.1, name="all_encoders"),
                    ParamGroupSpec(("encoders.signal",), 0.5, name="signal_encoder"),
                ],
            )
        message = str(caught.value)
        assert "training.param_groups[1] ('signal_encoder')" in message
        assert "already in the group(s) ['all_encoders']" in message
        assert "first entry that matches it" in message

    def test_an_entry_that_matches_only_modules_without_a_parameter_is_refused(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        with pytest.raises(ValueError, match=r"\('1'\): the modules it matches .* no parameter"):
            buildParamGroups(model, [ParamGroupSpec(("1",), 0.5)])


class TestApplyLrFactors:
    def test_each_group_s_rate_is_multiplied_by_its_factor(self) -> None:
        groups: list[dict[str, Any]] = [
            {"lr": 0.01, "lr_factor": 0.1},
            {"lr": 0.01, "lr_factor": 1.0},
            {"lr": 0.01, "lr_factor": 3.0},
        ]
        applyLrFactors(groups)
        assert [g["lr"] for g in groups] == pytest.approx([0.001, 0.01, 0.03])

    def test_a_tensor_rate_stays_a_tensor(self) -> None:
        groups: list[dict[str, Any]] = [{"lr": torch.tensor(0.01), "lr_factor": 0.5}]
        applyLrFactors(groups)
        assert isinstance(groups[0]["lr"], torch.Tensor)
        assert groups[0]["lr"].item() == pytest.approx(0.005)


def _signalGroups() -> list[ParamGroupSpec]:
    return [ParamGroupSpec(SIGNAL, 0.1, name="signal")]


class TestTrainerBuildsTheOptimizerFromGroups:
    def test_each_entry_is_a_group_at_its_factor_times_the_rate_and_the_rest_at_the_rate(
        self,
    ) -> None:
        trainer = buildTinyTrainer(param_groups=_signalGroups())
        groups = trainer.optimizer.param_groups
        assert [(g["name"], g["lr"]) for g in groups] == [
            ("signal", pytest.approx(0.001)),
            ("default", pytest.approx(0.01)),
        ]
        assert [g["lr_factor"] for g in groups] == [0.1, 1.0]

    def test_without_groups_the_optimizer_has_one_group_with_no_name_as_before(self) -> None:
        variants: list[dict[str, Any]] = [{}, {"param_groups": None}, {"param_groups": []}]
        for kwargs in variants:
            trainer = buildTinyTrainer(**kwargs)
            (group,) = trainer.optimizer.param_groups
            assert "name" not in group and "lr_factor" not in group

    def test_the_optimizer_s_default_rate_is_scaled_when_none_is_given(self) -> None:
        trainer = Trainer(
            buildTinyModel(),
            device="cpu",
            optimizer=torch.optim.Adam,
            param_groups=_signalGroups(),
        )
        rates = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
        assert rates["signal"] == pytest.approx(0.1 * rates["default"])
        assert rates["default"] == pytest.approx(1e-3)

    @pytest.mark.parametrize("optimizer", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD])
    def test_every_optimizer_takes_the_groups_and_trains_with_them(
        self, optimizer: type[torch.optim.Optimizer]
    ) -> None:
        trainer = buildTinyTrainer(optimizer=optimizer, param_groups=_signalGroups())
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.fit(tinyBatches(2), num_epochs=1)
        assert any(not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())

    def test_with_sgd_a_step_is_the_group_s_own_rate_times_its_gradient(self) -> None:
        trainer = buildTinyTrainer(
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.05},
            param_groups=[ParamGroupSpec(SIGNAL, 0.2, name="signal")],
        )
        model = trainer.model
        weights = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if n in ("encoders.signal.to_mu.weight", "encoders.image.to_mu.weight")
        }
        trainer.fit(tinyBatches(1), num_epochs=1)  # one batch: one step, the gradient is kept
        for name, factor in (
            ("encoders.signal.to_mu.weight", 0.2),
            ("encoders.image.to_mu.weight", 1.0),
        ):
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is not None
            expected = weights[name] - 0.05 * factor * parameter.grad
            torch.testing.assert_close(parameter.detach(), expected, rtol=1e-6, atol=1e-8)

    def test_an_optimizer_instance_cannot_be_given_groups(self) -> None:
        model = buildTinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="param_groups is for the optimizer that Trainer"):
            Trainer(model, optimizer=optimizer, device="cpu", param_groups=_signalGroups())

    def test_a_bad_entry_stops_the_trainer_when_it_is_built(self) -> None:
        with pytest.raises(ValueError, match=r"param_groups\[0\] \('nope'\).*matches no module"):
            buildTinyTrainer(param_groups=[ParamGroupSpec(("nope",), 0.5)])

    def test_the_parameters_of_a_loss_term_train_in_the_default_group(self) -> None:
        @registerLossTerm("p65_test_term_with_a_parameter")
        class _TermWithParameter(AbstractLossTerm):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, model: Any, batch: Any, outputs: Any, step: int) -> torch.Tensor:
                return (self.scale**2).sum()

        term = _TermWithParameter()
        trainer = buildTinyTrainer(
            param_groups=_signalGroups(), loss_terms={"p65_test_term_with_a_parameter": term}
        )
        by_name = {g["name"]: g for g in trainer.optimizer.param_groups}
        assert any(p is term.scale for p in by_name["default"]["params"])
        assert not any(p is term.scale for p in by_name["signal"]["params"])

    def test_a_schedule_callback_keeps_each_group_s_factor(self) -> None:
        trainer = buildTinyTrainer(
            param_groups=_signalGroups(), callbacks=[CosineLr(total_steps=100, warmup_steps=2)]
        )
        for _ in range(3):
            trainer.fit(tinyBatches(3), num_epochs=1)
            slow, fast = (g["lr"] for g in trainer.optimizer.param_groups)
            assert 0.0 < fast < 0.01 * 1.0000001
            assert slow == pytest.approx(0.1 * fast)

    def test_several_groups_are_logged_apart_when_a_callback_logs_the_rate(self) -> None:
        trainer = buildTinyTrainer(
            param_groups=_signalGroups(), callbacks=[CosineLr(total_steps=10)]
        )
        history = trainer.fit(tinyBatches(2), num_epochs=1)
        assert history[0]["train/lr/group0"] == pytest.approx(0.1 * history[0]["train/lr/group1"])

    def test_gradient_clipping_covers_every_group(self) -> None:
        trainer = buildTinyTrainer(
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.05},
            param_groups=_signalGroups(),
            grad_clip_norm=1e-3,
        )
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.fit(tinyBatches(1), num_epochs=1)
        for name in ("encoders.signal.to_mu.weight", "encoders.image.to_mu.weight"):
            moved = (trainer.model.state_dict()[name] - before[name]).abs().max().item()
            # Clipped to a global norm of 1e-3, so no element of the gradient is above 1e-3.
            assert 0.0 < moved <= 0.05 * 1e-3 * 1.0001, name


def _optimizerGroups(path: Path) -> list[dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    groups: list[dict[str, Any]] = state["optimizer_state_dict"]["param_groups"]
    return groups


class TestTheGroupsReachTheCheckpoint:
    def test_the_file_holds_each_group_s_name_factor_and_rate(self, tmp_path: Path) -> None:
        trainer = buildTinyTrainer(param_groups=_signalGroups())
        trainer.fit(tinyBatches(2), num_epochs=1)
        trainer.saveCheckpoint(tmp_path / "ckpt.pt")
        groups = _optimizerGroups(tmp_path / "ckpt.pt")
        assert [(g["name"], g["lr_factor"]) for g in groups] == [("signal", 0.1), ("default", 1.0)]
        assert [g["lr"] for g in groups] == pytest.approx([0.001, 0.01])

    def test_a_run_without_groups_writes_the_file_it_always_did(self, tmp_path: Path) -> None:
        trainer = buildTinyTrainer()
        trainer.fit(tinyBatches(2), num_epochs=1)
        trainer.saveCheckpoint(tmp_path / "ckpt.pt")
        (group,) = _optimizerGroups(tmp_path / "ckpt.pt")
        assert "name" not in group and "lr_factor" not in group

    def test_a_resume_with_the_same_groups_gives_the_uninterrupted_run(
        self, tmp_path: Path
    ) -> None:
        whole = buildTinyTrainer(buildTinyModel(1), param_groups=_signalGroups())
        torch.manual_seed(5)
        whole.fit(tinyBatches(3), num_epochs=4)

        first = buildTinyTrainer(buildTinyModel(1), param_groups=_signalGroups())
        torch.manual_seed(5)
        first.fit(tinyBatches(3), num_epochs=2)
        first.saveCheckpoint(tmp_path / "ckpt.pt")

        resumed = buildTinyTrainer(buildTinyModel(99), param_groups=_signalGroups())
        resumed.loadCheckpoint(tmp_path / "ckpt.pt")
        resumed.fit(tinyBatches(3), num_epochs=2)

        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name], tensor, rtol=1e-6, atol=1e-7, msg=name
            )

    @pytest.mark.parametrize(
        ("saved", "resumed"),
        [
            (
                [ParamGroupSpec(SIGNAL, 0.1, name="signal")],
                [ParamGroupSpec(SIGNAL, 0.5, name="signal")],
            ),
            (
                [ParamGroupSpec(SIGNAL, 0.1, name="signal")],
                [ParamGroupSpec(SIGNAL, 0.1, name="other")],
            ),
            ([ParamGroupSpec(SIGNAL, 0.1)], []),
            ([], [ParamGroupSpec(SIGNAL, 0.1)]),
            (
                [ParamGroupSpec(SIGNAL, 0.1, name="s")],
                [ParamGroupSpec(SIGNAL, 0.1, name="s"), ParamGroupSpec(IMAGE, 0.1, name="i")],
            ),
        ],
    )
    def test_a_resume_with_other_groups_is_refused_and_the_config_to_change_is_named(
        self, tmp_path: Path, saved: list[ParamGroupSpec], resumed: list[ParamGroupSpec]
    ) -> None:
        first = buildTinyTrainer(param_groups=saved)
        first.fit(tinyBatches(1), num_epochs=1)
        first.saveCheckpoint(tmp_path / "ckpt.pt")
        with pytest.raises(ValueError, match=r"`training.param_groups`.*restore_optimizer=False"):
            buildTinyTrainer(param_groups=resumed).loadCheckpoint(tmp_path / "ckpt.pt")

    def test_the_refusal_shows_both_layouts(self, tmp_path: Path) -> None:
        first = buildTinyTrainer(param_groups=_signalGroups())
        first.fit(tinyBatches(1), num_epochs=1)
        first.saveCheckpoint(tmp_path / "ckpt.pt")
        with pytest.raises(ValueError) as caught:
            buildTinyTrainer().loadCheckpoint(tmp_path / "ckpt.pt")
        assert "[('signal', 0.1), ('default', 1.0)]" in str(caught.value)
        assert "[(None, None)]" in str(caught.value)

    def test_the_weights_alone_can_still_be_loaded_into_other_groups(self, tmp_path: Path) -> None:
        first = buildTinyTrainer(param_groups=_signalGroups())
        first.fit(tinyBatches(1), num_epochs=1)
        first.saveCheckpoint(tmp_path / "ckpt.pt")
        other = buildTinyTrainer(buildTinyModel(7), param_groups=[ParamGroupSpec(IMAGE, 0.3)])
        other.loadCheckpoint(tmp_path / "ckpt.pt", restore_optimizer=False)
        assert [g["lr"] for g in other.optimizer.param_groups] == pytest.approx([0.003, 0.01])
        for name, tensor in first.model.state_dict().items():
            assert torch.equal(other.model.state_dict()[name], tensor)

    def test_without_groups_on_either_side_pytorch_s_own_message_stands(
        self, tmp_path: Path
    ) -> None:
        first = buildTinyTrainer()
        first.fit(tinyBatches(1), num_epochs=1)
        first.saveCheckpoint(tmp_path / "ckpt.pt")
        model = buildTinyModel()
        encoders = list(model.encoders.parameters())
        others = [p for p in model.parameters() if all(p is not q for q in encoders)]
        optimizer = torch.optim.Adam([{"params": encoders}, {"params": others}], lr=0.01)
        second = Trainer(model, optimizer=optimizer, device="cpu")
        with pytest.raises(ValueError, match="parameter group"):
            second.loadCheckpoint(tmp_path / "ckpt.pt")


class TestTheConfig:
    def _config(self, entries: list[dict[str, Any]]) -> TrainingConfig:
        merged = OmegaConf.merge(
            OmegaConf.structured(TrainingConfig), OmegaConf.create({"param_groups": entries})
        )
        config = OmegaConf.to_object(merged)
        assert isinstance(config, TrainingConfig)
        return config

    def test_the_default_is_no_group(self) -> None:
        assert TrainingConfig().param_groups == []
        assert buildParamGroupSpecs(TrainingConfig()) == []

    def test_yaml_entries_become_specs_in_order(self) -> None:
        config = self._config(
            [
                {"modules": ["encoders.signal", "decoders.signal"], "lr_factor": 0.1},
                {"modules": ["encoders.image"], "lr_factor": 0.5, "name": "img"},
            ]
        )
        assert isinstance(config.param_groups[0], ParamGroupConfig)
        assert buildParamGroupSpecs(config) == [
            ParamGroupSpec(("encoders.signal", "decoders.signal"), 0.1),
            ParamGroupSpec(("encoders.image",), 0.5, name="img"),
        ]

    def test_an_entry_without_a_factor_or_patterns_is_not_a_valid_config(self) -> None:
        with pytest.raises(Exception, match="lr_factor|modules"):
            self._config([{"modules": ["a"]}])
        with pytest.raises(Exception, match="lr_factor|modules"):
            self._config([{"lr_factor": 0.1}])

    def test_an_invalid_entry_is_named_by_its_index(self) -> None:
        config = self._config(
            [
                {"modules": ["encoders.signal"], "lr_factor": 0.1},
                {"modules": ["encoders.image"], "lr_factor": 0.0},
            ]
        )
        with pytest.raises(ValueError, match=r"training\.param_groups\[1\]: .*lr_factor"):
            buildParamGroupSpecs(config)

    def test_an_empty_pattern_list_is_named_by_its_index(self) -> None:
        config = self._config([{"modules": [], "lr_factor": 0.1}])
        with pytest.raises(ValueError, match=r"training\.param_groups\[0\]: .*modules"):
            buildParamGroupSpecs(config)

    def test_the_trainer_built_from_a_config_has_the_groups(self) -> None:
        config = self._config([{"modules": list(SIGNAL), "lr_factor": 0.1, "name": "signal"}])
        config.device = "cpu"
        trainer = buildTrainerFromConfig(buildTinyModel(), config)
        assert [(g["name"], g["lr"]) for g in trainer.optimizer.param_groups] == [
            ("signal", pytest.approx(0.0001)),
            ("default", pytest.approx(0.001)),
        ]

    def test_the_names_a_config_can_use_exist_in_the_model(self) -> None:
        assert namesUnder(buildTinyModel(), "encoders.signal.norm")
