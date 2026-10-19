"""Checkpoint loading is safe by default (roadmap P2-3, ADR 0034).

`loadCheckpoint` reads a file with `torch.load(weights_only=True)`, PyTorch's restricted
unpickler, unless the caller asks for `weights_only=False`. That is only possible because a
checkpoint written now holds nothing but tensors, numbers, strings, `None` and plain
containers: the NumPy RNG state is stored as a list, a dataclass `config` is stored as a dict.
These tests pin that, the failure a file from an earlier version gets (an error that names the
remedy), and the remedies: `upgradeCheckpoint`, `weights_only=False`, `--trust-checkpoint`.
"""

import importlib.util
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from global_vae.training.callbacks.checkpoint import CheckpointCallback
from global_vae.training.checkpoint import (
    loadCheckpoint,
    saveCheckpoint,
    upgradeCheckpoint,
)
from global_vae.training.trainer import Trainer
from tests.integration._script_fixtures import buildModelForScript

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_MODEL_FACTORY = "tests.integration._script_fixtures:buildModelForScript"
_DATALOADER_FACTORY = "tests.integration._script_fixtures:buildDataloaderForScript"


def _loadScript(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"global_vae_{name}_safe_test", _SCRIPTS / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _writeLegacyCheckpoint(path: Path) -> None:
    """Write a checkpoint the way versions before ADR 0034 did: a NumPy array in the RNG state."""
    saveCheckpoint(path, model=buildModelForScript())
    raw = torch.load(path, map_location="cpu", weights_only=True)
    name, keys, position, has_gauss, cached = raw["rng_state"]["numpy"]
    raw["rng_state"]["numpy"] = (
        name,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        cached,
    )
    torch.save(raw, path)


@dataclass
class _Config:
    learning_rate: float = 0.1
    name: str = "demo"


class TestACheckpointWrittenNowIsMadeOfPlainValues:
    def test_a_bare_checkpoint_loads_with_weights_only(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.pt"
        saveCheckpoint(path, model=buildModelForScript())
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert {"model_state_dict", "rng_state", "history", "config"} <= set(raw)

    def test_a_checkpoint_with_everything_loads_with_weights_only(self, tmp_path: Path) -> None:
        trainer = Trainer(
            buildModelForScript(),
            device="cpu",
            callbacks=[CheckpointCallback(tmp_path, config={"seed": 3, "lr": [0.1, 0.01]})],
            mixed_precision="bf16",
        )
        trainer.fit([{"signal": torch.randn(4, 16)}], num_epochs=1, val_dataloader=None)
        raw = torch.load(next(tmp_path.glob("*.pt")), map_location="cpu", weights_only=True)
        assert raw["config"] == {"seed": 3, "lr": [0.1, 0.01]}
        assert raw["optimizer_state_dict"]["param_groups"]
        assert raw["callbacks_state"][0]["class"] == "CheckpointCallback"

    def test_the_numpy_rng_state_is_a_list_not_an_array(self, tmp_path: Path) -> None:
        path = tmp_path / "rng.pt"
        saveCheckpoint(path, model=buildModelForScript())
        raw = torch.load(path, map_location="cpu", weights_only=True)
        keys = raw["rng_state"]["numpy"][1]
        assert isinstance(keys, list)
        assert len(keys) == 624

    def test_the_rng_state_round_trips_through_the_safe_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "rng.pt"
        saveCheckpoint(path, model=buildModelForScript())
        expected = (np.random.rand(3), torch.rand(3))
        np.random.rand(10)
        torch.rand(10)
        metadata = loadCheckpoint(path, model=buildModelForScript())
        assert metadata.rng_state_restored
        observed = (np.random.rand(3), torch.rand(3))
        np.testing.assert_array_equal(observed[0], expected[0])
        assert torch.equal(observed[1], expected[1])

    def test_a_dataclass_config_is_stored_as_a_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "dataclass.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=_Config())
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert raw["config"] == {"learning_rate": 0.1, "name": "demo"}
        metadata = loadCheckpoint(path, model=buildModelForScript())
        assert metadata.config == {"learning_rate": 0.1, "name": "demo"}

    def test_a_nested_dataclass_config_is_converted_too(self, tmp_path: Path) -> None:
        @dataclass
        class Outer:
            inner: _Config
            sizes: tuple[int, ...] = (1, 2)

        path = tmp_path / "nested.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=Outer(inner=_Config()))
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert raw["config"]["inner"] == {"learning_rate": 0.1, "name": "demo"}
        assert tuple(raw["config"]["sizes"]) == (1, 2)

    def test_a_dataclass_type_is_not_converted(self, tmp_path: Path) -> None:
        """Only an instance is a snapshot; the class itself is written as given (and refused)."""
        path = tmp_path / "type.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=_Config)
        with pytest.raises(pickle.UnpicklingError):
            torch.load(path, map_location="cpu", weights_only=True)

    @pytest.mark.parametrize("config", [None, {"a": 1}, [1, 2, 3], "text", 3.5])
    def test_other_configs_are_stored_unchanged(self, tmp_path: Path, config: Any) -> None:
        path = tmp_path / "other.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=config)
        assert loadCheckpoint(path, model=buildModelForScript()).config == config


class TestAnObjectInTheConfigNeedsTheUnsafeMode:
    class _Opaque:
        pass

    def test_it_is_refused_with_a_message_that_names_the_remedies(self, tmp_path: Path) -> None:
        path = tmp_path / "opaque.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=self._Opaque())
        with pytest.raises(pickle.UnpicklingError, match="upgradeCheckpoint") as excinfo:
            loadCheckpoint(path, model=buildModelForScript())
        message = str(excinfo.value)
        assert "weights_only=False" in message
        assert "--trust-checkpoint" in message

    def test_it_loads_when_the_caller_trusts_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "opaque.pt"
        saveCheckpoint(path, model=buildModelForScript(), config=self._Opaque())
        metadata = loadCheckpoint(path, model=buildModelForScript(), weights_only=False)
        assert isinstance(metadata.config, self._Opaque)


class TestAFileFromAnEarlierVersion:
    def test_it_is_refused_in_the_safe_mode_with_a_helpful_error(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        with pytest.raises(pickle.UnpicklingError) as excinfo:
            loadCheckpoint(path, model=buildModelForScript())
        message = str(excinfo.value)
        assert str(path) in message
        assert "NumPy array" in message
        assert "upgradeCheckpoint" in message

    def test_it_loads_when_the_caller_trusts_it_and_restores_the_rng(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        metadata = loadCheckpoint(path, model=buildModelForScript(), weights_only=False)
        assert metadata.rng_state_restored

    def test_a_trainer_forwards_the_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        trainer = Trainer(buildModelForScript(), device="cpu")
        with pytest.raises(pickle.UnpicklingError):
            trainer.loadCheckpoint(path, restore_optimizer=False)
        trainer.loadCheckpoint(path, restore_optimizer=False, weights_only=False)

    def test_a_file_without_the_new_keys_still_loads_in_either_mode(self, tmp_path: Path) -> None:
        """A checkpoint with no callback or scaler state, as every earlier one, is fine."""
        path = tmp_path / "plain.pt"
        saveCheckpoint(path, model=buildModelForScript(), include_rng_state=False)
        for weights_only in (True, False):
            metadata = loadCheckpoint(path, model=buildModelForScript(), weights_only=weights_only)
            assert metadata.callbacks_state == []
            assert metadata.scaler_state_dict is None


class TestUpgradeCheckpoint:
    def test_it_makes_a_legacy_file_readable_in_the_safe_mode_in_place(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        returned = upgradeCheckpoint(path)
        assert returned == path
        metadata = loadCheckpoint(path, model=buildModelForScript())
        assert metadata.rng_state_restored

    def test_it_can_write_to_another_path_and_leaves_the_original_alone(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        upgraded = upgradeCheckpoint(path, tmp_path / "out" / "safe.pt")
        assert upgraded == tmp_path / "out" / "safe.pt"
        with pytest.raises(pickle.UnpicklingError):
            loadCheckpoint(path, model=buildModelForScript())
        loadCheckpoint(upgraded, model=buildModelForScript())

    def test_it_keeps_everything_else(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        trainer = Trainer(buildModelForScript(), device="cpu")
        trainer.fit([{"signal": torch.randn(4, 16)}], num_epochs=2)
        trainer.saveCheckpoint(path, config={"seed": 1})
        raw = torch.load(path, map_location="cpu", weights_only=True)
        name, keys, position, has_gauss, cached = raw["rng_state"]["numpy"]
        raw["rng_state"]["numpy"] = (
            name,
            np.asarray(keys, dtype=np.uint32),
            position,
            has_gauss,
            cached,
        )
        torch.save(raw, path)

        upgradeCheckpoint(path)
        upgraded = torch.load(path, map_location="cpu", weights_only=True)
        assert upgraded["global_step"] == raw["global_step"] == 2
        assert upgraded["start_epoch"] == 2
        assert upgraded["history"] == raw["history"]
        assert upgraded["config"] == {"seed": 1}
        for key, tensor in raw["model_state_dict"].items():
            assert torch.equal(upgraded["model_state_dict"][key], tensor)
        assert upgraded["rng_state"]["numpy"][1] == keys

    def test_a_file_that_is_already_safe_is_left_equivalent(self, tmp_path: Path) -> None:
        path = tmp_path / "new.pt"
        saveCheckpoint(path, model=buildModelForScript())
        before = torch.load(path, map_location="cpu", weights_only=True)
        upgradeCheckpoint(path)
        after = torch.load(path, map_location="cpu", weights_only=True)
        assert after["rng_state"]["numpy"] == before["rng_state"]["numpy"]

    def test_a_file_with_an_object_in_its_config_is_rewritten_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "opaque.pt"
        saveCheckpoint(
            path,
            model=buildModelForScript(),
            config=TestAnObjectInTheConfigNeedsTheUnsafeMode._Opaque(),
        )
        with caplog.at_level("WARNING"):
            upgradeCheckpoint(path)
        assert "still not readable in the safe mode" in caplog.text

    def test_a_missing_file_is_a_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            upgradeCheckpoint(tmp_path / "nope.pt")


class TestTheScriptsHaveATrustFlag:
    @pytest.mark.parametrize("name", ["evaluate.py", "visualize_latent.py"])
    def test_the_flag_is_documented_in_the_help(self, name: str) -> None:
        parser = _loadScript(name)._buildArgumentParser()
        help_text = "".join(parser.format_help().split())
        assert "--trust-checkpoint" in help_text
        assert "ADR0034" in help_text or "upgradeCheckpoint" in help_text

    @pytest.mark.parametrize("name", ["evaluate.py", "visualize_latent.py"])
    def test_it_is_off_by_default(self, name: str) -> None:
        parser = _loadScript(name)._buildArgumentParser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "x.pt",
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
            ]
        )
        assert args.trust_checkpoint is False

    def test_evaluate_refuses_a_legacy_file_unless_told_to_trust_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _loadScript("evaluate.py")
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        argv = [
            "--checkpoint",
            str(path),
            "--model-factory",
            _MODEL_FACTORY,
            "--dataloader-factory",
            _DATALOADER_FACTORY,
            "--device",
            "cpu",
        ]
        with pytest.raises(pickle.UnpicklingError, match="upgradeCheckpoint"):
            script.main(argv)
        script.main([*argv, "--trust-checkpoint"])
        assert "Evaluation over" in capsys.readouterr().out

    def test_visualize_refuses_a_legacy_file_unless_told_to_trust_it(self, tmp_path: Path) -> None:
        script = _loadScript("visualize_latent.py")
        path = tmp_path / "legacy.pt"
        _writeLegacyCheckpoint(path)
        argv = [
            "--checkpoint",
            str(path),
            "--model-factory",
            _MODEL_FACTORY,
            "--dataloader-factory",
            _DATALOADER_FACTORY,
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "viz"),
        ]
        with pytest.raises(pickle.UnpicklingError, match="upgradeCheckpoint"):
            script.main(argv)
        script.main([*argv, "--trust-checkpoint"])
        assert (tmp_path / "viz" / "latent_z_fused.png").exists()
