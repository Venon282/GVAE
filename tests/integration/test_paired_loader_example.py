"""Integration tests for `examples/04_paired_loader.py` (roadmap P3-2, spec §6.2).

The example is the reference `loader_factory` for paired (signal, image) data: it pairs two
folders of files by sample ID, splits the pairs, applies the per-modality transforms and hands
`scripts/train.py` a `DataloaderBundle`. Most of it is small and testable on its own, so the
module is loaded in-process through `importlib` (its file name starts with a digit, so it
cannot be imported normally) and tested directly: the pairing, the seeded split, the loaders
and the checks. Two tests then run `scripts/train.py` as a real subprocess on a paired
dataset, which is what P3-2 asks for, one through the example's own `--train` flag and one with
explicit `val_path` and `test_path` folders and two loader worker processes.

The example imports nothing but `global_vae` and third-party packages, which is what makes
loading it in-process safe here.
"""

import importlib.util
import logging
import os
import subprocess
import sys
from collections.abc import Sized
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from global_vae.config.data import DataConfig, TransformConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "04_paired_loader.py"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"

_Loader = DataLoader[dict[str, torch.Tensor]]


def _loadExample() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_example_04", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _loadExample()


def _writeMarkedDataset(
    example: ModuleType,
    root: Path,
    both: list[str],
    signal_only: list[str] | None = None,
    image_only: list[str] | None = None,
    length: int = 8,
) -> None:
    """Write files whose content is the sample's number, so a wrong pairing is visible.

    The signal of sample `s0007` is `7.0` everywhere and so is its image. A batch whose signal
    and image disagree was paired wrongly.
    """
    signal_dir = root / example.SIGNAL_DIRECTORY
    image_dir = root / example.IMAGE_DIRECTORY
    signal_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in [*both, *(signal_only or [])]:
        value = float(sample_id[1:])
        np.save(signal_dir / f"{sample_id}.npy", np.full(length, value, dtype=np.float32))
    for sample_id in [*both, *(image_only or [])]:
        value = float(sample_id[1:])
        np.save(image_dir / f"{sample_id}.npy", np.full((4, 4), value, dtype=np.float32))


def _ids(count: int, start: int = 0) -> list[str]:
    return [f"s{index:04d}" for index in range(start, start + count)]


def _config(root: Path, **overrides: object) -> DataConfig:
    fields: dict[str, object] = {
        "loader_factory": "unused:unused",
        "train_path": str(root),
        "batch_size": 4,
        "val_split": None,
        "test_split": None,
        "shuffle_train": False,
        "seed": 0,
    }
    fields.update(overrides)
    return DataConfig(**fields)  # type: ignore[arg-type]


def _sampleIds(loader: object) -> list[str]:
    ids: list[str] = cast(_Loader, loader).dataset.sample_ids  # type: ignore[attr-defined]
    return ids


def _allBatches(loader: object) -> list[dict[str, torch.Tensor]]:
    return list(cast(_Loader, loader))


class TestPairing:
    def test_pairs_by_id_not_by_position(self, example: ModuleType) -> None:
        """Unpaired samples interleaved among the pairs must not shift the pairs after them."""
        signal_files = {f"s{i}": Path(f"signal/s{i}.npy") for i in (1, 2, 3, 5, 6)}
        image_files = {f"s{i}": Path(f"image/s{i}.npy") for i in (1, 3, 4, 5, 6)}
        result = example.pairBySampleId(signal_files, image_files)
        assert result.sampleIds == ["s1", "s3", "s5", "s6"]
        for sample_id in result.sampleIds:
            assert result.signal_files[sample_id].stem == sample_id
            assert result.image_files[sample_id].stem == sample_id

    def test_reports_what_is_left_out(self, example: ModuleType) -> None:
        result = example.pairBySampleId(
            {"a": Path("a"), "b": Path("b")}, {"b": Path("b"), "c": Path("c")}
        )
        assert result.signal_only == ["a"]
        assert result.image_only == ["c"]
        assert result.sampleIds == ["b"]

    def test_paired_ids_come_out_sorted(self, example: ModuleType) -> None:
        files = {name: Path(name) for name in ("c", "a", "b")}
        assert example.pairBySampleId(files, files).sampleIds == ["a", "b", "c"]

    def test_indexing_ignores_other_files(self, example: ModuleType, tmp_path: Path) -> None:
        np.save(tmp_path / "s0.npy", np.zeros(3))
        (tmp_path / "notes.txt").write_text("not data")
        (tmp_path / ".DS_Store").write_text("")
        assert list(example.indexModality(tmp_path, "signal")) == ["s0"]

    def test_indexing_a_missing_folder_is_an_error(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="signal folder"):
            example.indexModality(tmp_path / "absent", "signal")

    def test_indexing_an_empty_folder_is_an_error(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="holds no"):
            example.indexModality(tmp_path, "image")

    def test_two_files_with_one_id_are_an_error(
        self, example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        np.save(tmp_path / "sig_1.npy", np.zeros(3))
        np.save(tmp_path / "sig_2.npy", np.zeros(3))
        monkeypatch.setattr(example, "sampleIdFromPath", lambda path, modality: "same")
        with pytest.raises(ValueError, match="one file per modality"):
            example.indexModality(tmp_path, "signal")

    def test_a_dataset_whose_ids_never_match_is_an_error(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, both=[], signal_only=_ids(2), image_only=_ids(2, 5))
        with pytest.raises(ValueError, match="sampleIdFromPath"):
            example.pairDirectory(tmp_path)

    def test_samples_left_out_are_warned_about(
        self, example: ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _writeMarkedDataset(
            example, tmp_path, both=_ids(3), signal_only=["s0100"], image_only=["s0200"]
        )
        with caplog.at_level(logging.WARNING, logger="global_vae.examples.paired_loader"):
            example.pairDirectory(tmp_path)
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "s0100" in messages and "s0200" in messages


class TestFileReaders:
    def test_signal_must_be_one_dimensional(self, example: ModuleType, tmp_path: Path) -> None:
        np.save(tmp_path / "x.npy", np.zeros((2, 3)))
        with pytest.raises(ValueError, match="1D"):
            example.loadSignal(tmp_path / "x.npy")

    def test_image_must_be_two_dimensional(self, example: ModuleType, tmp_path: Path) -> None:
        np.save(tmp_path / "x.npy", np.zeros(3))
        with pytest.raises(ValueError, match="2D"):
            example.loadImage(tmp_path / "x.npy")

    def test_readers_return_float32_tensors(self, example: ModuleType, tmp_path: Path) -> None:
        np.save(tmp_path / "s.npy", np.arange(5, dtype=np.float64))
        np.save(tmp_path / "i.npy", np.zeros((3, 2), dtype=np.float64))
        signal = example.loadSignal(tmp_path / "s.npy")
        image = example.loadImage(tmp_path / "i.npy")
        assert signal.dtype == torch.float32 and signal.shape == (5,)
        assert image.dtype == torch.float32 and image.shape == (3, 2)


class TestSplit:
    def test_splits_are_disjoint_and_cover_every_sample(self, example: ModuleType) -> None:
        ids = _ids(50)
        train, val, test = example.splitSampleIds(ids, 0.2, 0.1, seed=3)
        assert (len(train), len(val), len(test)) == (35, 10, 5)
        assert sorted([*train, *val, *test]) == ids

    def test_the_split_is_reproducible_for_a_seed(self, example: ModuleType) -> None:
        first = example.splitSampleIds(_ids(40), 0.25, 0.25, seed=7)
        second = example.splitSampleIds(_ids(40), 0.25, 0.25, seed=7)
        assert first == second

    def test_another_seed_gives_another_split(self, example: ModuleType) -> None:
        first = example.splitSampleIds(_ids(40), 0.25, 0.25, seed=1)
        second = example.splitSampleIds(_ids(40), 0.25, 0.25, seed=2)
        assert first != second

    def test_the_split_does_not_depend_on_the_order_the_ids_are_given_in(
        self, example: ModuleType
    ) -> None:
        ids = _ids(40)
        assert example.splitSampleIds(ids, 0.25, 0.25, 5) == example.splitSampleIds(
            list(reversed(ids)), 0.25, 0.25, 5
        )

    def test_an_unset_fraction_gives_an_empty_split(self, example: ModuleType) -> None:
        train, val, test = example.splitSampleIds(_ids(10), None, 0.0, seed=0)
        assert (len(train), val, test) == (10, [], [])

    def test_a_small_dataset_still_gets_a_non_empty_split(self, example: ModuleType) -> None:
        _, val, test = example.splitSampleIds(_ids(5), 0.1, 0.1, seed=0)
        assert len(val) == 1 and len(test) == 1

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_an_invalid_fraction_is_an_error(self, example: ModuleType, fraction: float) -> None:
        with pytest.raises(ValueError, match="val_split"):
            example.splitSampleIds(_ids(10), fraction, None, seed=0)

    def test_the_splits_must_leave_a_training_sample(self, example: ModuleType) -> None:
        with pytest.raises(ValueError, match="at least one for training"):
            example.splitSampleIds(_ids(2), 0.5, 0.5, seed=0)


class TestBuildPairedDataloaders:
    def test_batches_have_the_keys_and_shapes_the_experiments_read(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        example.writeSyntheticPairedDataset(tmp_path, num_pairs=20, num_signal_only=2, seed=1)
        config = _config(tmp_path, batch_size=8, sequence_length={"signal": example.SIGNAL_LENGTH})
        bundle = example.buildPairedDataloaders(config)
        batch = next(iter(bundle.train))
        assert set(batch) == {"signal", "image"}
        assert batch["signal"].shape == (8, example.SIGNAL_LENGTH)
        assert batch["image"].shape == (8, example.IMAGE_SIZE, example.IMAGE_SIZE)
        assert batch["signal"].dtype == batch["image"].dtype == torch.float32

    def test_the_synthetic_signals_are_positive(self, example: ModuleType, tmp_path: Path) -> None:
        """The `log` step of `configs/data/signal_image.yaml` needs positive signals."""
        example.writeSyntheticPairedDataset(tmp_path, num_pairs=10, seed=2)
        batch = next(iter(example.buildPairedDataloaders(_config(tmp_path)).train))
        assert float(batch["signal"].min()) > 0.0

    def test_every_batch_pairs_the_right_signal_with_the_right_image(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        """Samples with one modality only, mixed in, must not shift the pairs after them."""
        both = ["s0001", "s0004", "s0005", "s0008", "s0009", "s0012"]
        _writeMarkedDataset(
            example, tmp_path, both, signal_only=["s0002", "s0006"], image_only=["s0003", "s0010"]
        )
        bundle = example.buildPairedDataloaders(_config(tmp_path))
        seen: list[float] = []
        for batch in _allBatches(bundle.train):
            assert torch.equal(batch["signal"][:, 0], batch["image"][:, 0, 0])
            seen.extend(batch["signal"][:, 0].tolist())
        assert seen == [1.0, 4.0, 5.0, 8.0, 9.0, 12.0]

    def test_unpaired_samples_are_in_no_split(self, example: ModuleType, tmp_path: Path) -> None:
        both = _ids(20)
        _writeMarkedDataset(
            example, tmp_path, both, signal_only=_ids(3, 100), image_only=_ids(3, 200)
        )
        bundle = example.buildPairedDataloaders(_config(tmp_path, val_split=0.2, test_split=0.1))
        used = [*_sampleIds(bundle.train), *_sampleIds(bundle.val), *_sampleIds(bundle.test)]
        assert sorted(used) == both

    def test_split_sizes_follow_the_fractions_and_never_overlap(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(40))
        bundle = example.buildPairedDataloaders(_config(tmp_path, val_split=0.25, test_split=0.1))
        train, val, test = (
            _sampleIds(loader) for loader in (bundle.train, bundle.val, bundle.test)
        )
        assert (len(train), len(val), len(test)) == (26, 10, 4)
        assert not set(train) & set(val) | set(train) & set(test) | set(val) & set(test)

    def test_the_split_follows_the_data_seed(self, example: ModuleType, tmp_path: Path) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(40))
        runs = [
            _sampleIds(
                example.buildPairedDataloaders(_config(tmp_path, val_split=0.25, seed=seed)).val
            )
            for seed in (4, 4, 5)
        ]
        assert runs[0] == runs[1]
        assert runs[0] != runs[2]

    def test_a_split_left_unset_gives_no_loader(self, example: ModuleType, tmp_path: Path) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(10))
        bundle = example.buildPairedDataloaders(_config(tmp_path))
        assert bundle.val is None and bundle.test is None
        assert len(_sampleIds(bundle.train)) == 10

    def test_explicit_val_and_test_folders_replace_the_split_fractions(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        train_root, val_root, test_root = (tmp_path / name for name in ("train", "val", "test"))
        _writeMarkedDataset(example, train_root, _ids(10))
        _writeMarkedDataset(example, val_root, _ids(3, 50), signal_only=["s0099"])
        _writeMarkedDataset(example, test_root, _ids(2, 70))
        config = _config(
            train_root,
            val_path=str(val_root),
            test_path=str(test_root),
            val_split=0.5,
            test_split=0.5,
        )
        bundle = example.buildPairedDataloaders(config)
        assert _sampleIds(bundle.train) == _ids(10)
        assert _sampleIds(bundle.val) == _ids(3, 50)
        assert _sampleIds(bundle.test) == _ids(2, 70)

    def test_only_the_training_loader_shuffles(self, example: ModuleType, tmp_path: Path) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(40))
        config = _config(
            tmp_path, batch_size=40, val_split=0.25, test_split=0.25, shuffle_train=True
        )
        bundle = example.buildPairedDataloaders(config)
        train_values = _allBatches(bundle.train)[0]["signal"][:, 0].tolist()
        assert train_values != sorted(train_values)
        for loader in (bundle.val, bundle.test):
            values = _allBatches(loader)[0]["signal"][:, 0].tolist()
            assert values == sorted(values)

    def test_training_order_is_fixed_by_the_seed_and_changes_between_epochs(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(30))

        def epochs(seed: int) -> list[list[float]]:
            config = _config(tmp_path, batch_size=30, shuffle_train=True, seed=seed)
            loader = example.buildPairedDataloaders(config).train
            return [_allBatches(loader)[0]["signal"][:, 0].tolist() for _ in range(2)]

        first_run, second_run, other_seed = epochs(1), epochs(1), epochs(2)
        assert first_run == second_run
        assert first_run[0] != first_run[1]
        assert first_run != other_seed

    def test_without_shuffling_the_order_is_by_sample_id(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(12))
        loader = example.buildPairedDataloaders(_config(tmp_path, batch_size=12)).train
        values = _allBatches(loader)[0]["signal"][:, 0].tolist()
        assert values == [float(index) for index in range(12)]

    def test_batch_size_is_honoured(self, example: ModuleType, tmp_path: Path) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(10))
        loader = example.buildPairedDataloaders(_config(tmp_path, batch_size=4)).train
        assert [batch["signal"].shape[0] for batch in _allBatches(loader)] == [4, 4, 2]
        assert len(cast(Sized, loader)) == 3

    def test_the_configured_transforms_run_on_each_modality(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, ["s0003", "s0007"])
        transforms = {
            "signal": [TransformConfig(name="standardize", kwargs={"mean": 1.0, "std": 2.0})],
            "image": [TransformConfig(name="log", kwargs={"eps": 1e-6})],
        }
        bundle = example.buildPairedDataloaders(_config(tmp_path, transforms=transforms))
        batch = _allBatches(bundle.train)[0]
        assert torch.allclose(batch["signal"][:, 0], torch.tensor([1.0, 3.0]))
        assert torch.allclose(batch["image"][:, 0, 0], (torch.tensor([3.0, 7.0]) + 1e-6).log())

    def test_a_modality_without_transforms_is_used_as_stored(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, ["s0003"])
        transforms = {
            "signal": [TransformConfig(name="standardize", kwargs={"mean": 1.0, "std": 2.0})]
        }
        batch = _allBatches(
            example.buildPairedDataloaders(_config(tmp_path, transforms=transforms)).train
        )[0]
        assert float(batch["image"][0, 0, 0]) == 3.0

    def test_a_signal_of_the_wrong_length_fails_before_training(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(4), length=8)
        config = _config(tmp_path, sequence_length={"signal": 16})
        with pytest.raises(ValueError, match=r"8 point\(s\).*sequence_length.signal is 16"):
            example.buildPairedDataloaders(config)

    def test_a_resample_step_brings_the_signal_to_the_configured_length(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(4), length=8)
        transforms = {"signal": [TransformConfig(name="resample", kwargs={"target_size": 16})]}
        config = _config(tmp_path, transforms=transforms, sequence_length={"signal": 16})
        batch = next(iter(example.buildPairedDataloaders(config).train))
        assert batch["signal"].shape == (4, 16)

    def test_no_sequence_length_means_no_check(self, example: ModuleType, tmp_path: Path) -> None:
        _writeMarkedDataset(example, tmp_path, _ids(4), length=8)
        bundle = example.buildPairedDataloaders(_config(tmp_path, sequence_length=None))
        assert next(iter(bundle.train))["signal"].shape == (4, 8)


# Every test below runs `scripts/train.py` in a subprocess.
@pytest.mark.slow
class TestTrainingOnPairedData:
    def _environment(self) -> dict[str, str]:
        python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
        return {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}

    def test_the_example_trains_through_scripts_train(self, tmp_path: Path) -> None:
        """The whole P3-2 loop, from the example's own command line."""
        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--output-dir",
                str(tmp_path),
                "--num-pairs",
                "40",
                "--num-signal-only",
                "3",
                "--num-image-only",
                "2",
                "--train",
                "--num-epochs",
                "1",
            ],
            cwd=_REPO_ROOT,
            env=self._environment(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "Wrote 40 pairs, 3 signal-only and 2 image-only" in output
        assert "signal (" in output and "image (" in output
        assert (
            "python " in output and "train.py --config-name experiment/signal_image_vae" in output
        )
        assert "Training complete" in output
        assert (tmp_path / "run" / "checkpoints" / "best.pt").is_file()
        assert (tmp_path / "run" / "metrics.csv").is_file()

    def test_scripts_train_reads_explicit_validation_and_test_folders(
        self, example: ModuleType, tmp_path: Path
    ) -> None:
        for name, seed in (("train", 0), ("val", 1), ("test", 2)):
            example.writeSyntheticPairedDataset(
                tmp_path / name, num_pairs=12, num_signal_only=1, seed=seed
            )
        completed = subprocess.run(
            [
                sys.executable,
                str(_TRAIN_SCRIPT),
                "--config-name",
                "experiment/signal_image_vae",
                "data.loader_factory=examples.04_paired_loader:buildPairedDataloaders",
                f"data.train_path={tmp_path / 'train'}",
                f"data.val_path={tmp_path / 'val'}",
                f"data.test_path={tmp_path / 'test'}",
                "data.batch_size=6",
                "data.num_workers=2",
                f"output_dir={tmp_path / 'run'}",
                "training.num_epochs=1",
                "training.beta_schedules.z_fused.kwargs.warmup_steps=2",
                f"hydra.run.dir={tmp_path / 'hydra'}",
            ],
            cwd=_REPO_ROOT,
            env=self._environment(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "Splits: 12 train, 12 validation, 12 test sample(s)." in output
        assert "val/loss/total" in output
        assert "Training complete" in output
