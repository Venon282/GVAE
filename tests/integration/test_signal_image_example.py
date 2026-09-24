"""Integration tests for `examples/03_signal_image_to_image.py` (spec §2.1 `EN-L1-DN`, §4,
§5, §6).

The example trains `(signal, image_in) -> image_out`: two encoders fused by PoE into one
latent space, and a decoder whose target (`image_out`) is not any encoder's input. Most of
what it demonstrates lives in small, separately testable pieces, so those are loaded
in-process through `importlib` (the file name starts with a digit, so it cannot be
imported normally) and tested directly: the synthetic data, the model wiring, and
`TranslationTrainer`. One test additionally runs the whole script as a real subprocess,
mirroring `test_config_driven_pipeline_example.py`.

The example imports nothing but `global_vae` and third-party packages (no sibling helper
module), which is what makes loading it in-process safe here.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from global_vae.training.trainer import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "03_signal_image_to_image.py"


def _loadExample() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_example_03", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _loadExample()


def _batch(example: ModuleType, num_samples: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    return example.buildSplit(num_samples, np.random.default_rng(seed))


class TestSyntheticData:
    def test_split_has_the_three_views_with_the_documented_shapes(
        self, example: ModuleType
    ) -> None:
        split = _batch(example, num_samples=5)
        assert set(split) == {"signal", "image_in", "image_out"}
        assert split["signal"].shape == (5, example.SIGNAL_LENGTH)
        assert split["image_in"].shape == (5, example.IMAGE_SIZE, example.IMAGE_SIZE)
        assert split["image_out"].shape == (5, example.IMAGE_SIZE, example.IMAGE_SIZE)

    def test_clean_images_stay_within_the_unit_range(self, example: ModuleType) -> None:
        clean = _batch(example, num_samples=16)["image_out"]
        assert float(clean.min()) >= 0.0
        assert float(clean.max()) <= 1.0

    def test_generation_is_reproducible_for_a_given_seed(self, example: ModuleType) -> None:
        first = _batch(example, seed=3)
        second = _batch(example, seed=3)
        for name in first:
            assert torch.equal(first[name], second[name])

    def test_the_input_image_is_not_the_target_image(self, example: ModuleType) -> None:
        """The point of the example: returning `image_in` must not solve the task."""
        split = _batch(example, num_samples=32)
        difference = (split["image_in"] - split["image_out"]).pow(2).mean()
        assert float(difference) > 0.15

    def test_degradation_erases_one_square_and_nothing_else_without_noise(
        self, example: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(example, "NOISE_STD", 0.0)
        params = example.generateBlobParameters(8, np.random.default_rng(1))
        clean = example.renderBlobImages(params)
        degraded = example.degradeImages(clean, np.random.default_rng(2))
        side = example.OCCLUSION_SIZE
        for index in range(clean.shape[0]):
            rows, columns = np.nonzero(degraded[index] != clean[index])
            assert degraded[index][rows, columns].max(initial=0.0) == 0.0
            if rows.size:
                assert rows.max() - rows.min() + 1 <= side
                assert columns.max() - columns.min() + 1 <= side

    def test_signal_is_blind_to_the_vertical_position(
        self, example: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same `cx` and `sigma`, different `cy`: identical signal. This blindness is what
        makes `signal` and the degraded image complementary."""
        monkeypatch.setattr(example, "SIGNAL_NOISE_STD", 0.0)
        params = np.array([[0.4, 0.3, 0.1], [0.4, 0.7, 0.1]])
        profiles = example.renderXProfiles(params, np.random.default_rng(0))
        assert np.allclose(profiles[0], profiles[1])

    def test_signal_does_depend_on_the_horizontal_position(
        self, example: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(example, "SIGNAL_NOISE_STD", 0.0)
        params = np.array([[0.3, 0.5, 0.1], [0.7, 0.5, 0.1]])
        profiles = example.renderXProfiles(params, np.random.default_rng(0))
        assert not np.allclose(profiles[0], profiles[1])

    def test_batches_keep_every_key_in_order(self, example: ModuleType) -> None:
        split = _batch(example, num_samples=10)
        batches = example.toBatches(split, batch_size=4)
        assert [batch["signal"].shape[0] for batch in batches] == [4, 4, 2]
        assert torch.equal(torch.cat([batch["image_out"] for batch in batches]), split["image_out"])
        assert all(set(batch) == set(split) for batch in batches)


class TestModel:
    def test_wiring_is_two_encoders_one_fusion_one_decoder(self, example: ModuleType) -> None:
        model = example.buildModel()
        assert set(model.encoders) == {"signal", "image_in"}
        assert set(model.decoders) == {"image_out"}
        assert set(model.fusions) == {"z_fused"}
        assert set(model.latent_spaces) == {"z_fused"}

    def test_the_decoder_name_is_not_an_encoder_name(self, example: ModuleType) -> None:
        model = example.buildModel()
        assert not set(model.decoders) & set(model.encoders)

    @pytest.mark.parametrize(
        "inputs",
        [{"signal"}, {"image_in"}, {"signal", "image_in"}],
        ids=["signal", "image", "both"],
    )
    def test_every_input_subset_produces_the_target_image(
        self, example: ModuleType, inputs: set[str]
    ) -> None:
        model = example.buildModel()
        model.eval()
        split = _batch(example, num_samples=3)
        output = model({name: split[name] for name in inputs})
        assert output["reconstructions"]["image_out"].shape == (
            3,
            example.IMAGE_SIZE,
            example.IMAGE_SIZE,
        )

    def test_uses_the_free_bits_regularizer(self, example: ModuleType) -> None:
        from global_vae.losses.regularizers.free_bits_kl import FreeBitsKlRegularizer

        assert isinstance(example.buildModel().regularizers["z_fused"], FreeBitsKlRegularizer)


class TestTranslationTrainer:
    def test_the_stock_trainer_cannot_train_on_a_target_only_key(self, example: ModuleType) -> None:
        """Documents why `TranslationTrainer` exists. If `Trainer` itself ever learns to
        ignore batch keys that have no encoder, this test fails: delete `TranslationTrainer`
        (keep its validation-without-dropout `evaluate` if still wanted) and this test."""
        model = example.buildModel()
        with pytest.raises(KeyError, match="image_out"):
            Trainer(model, device="cpu").computeLosses(_batch(example), step=0)

    def test_losses_are_computed_with_a_target_only_key(self, example: ModuleType) -> None:
        trainer = example.TranslationTrainer(example.buildModel(), device="cpu")
        losses = trainer.computeLosses(_batch(example), step=0)
        assert torch.isfinite(losses.total)

    def test_the_target_only_key_never_reaches_the_encoders(self, example: ModuleType) -> None:
        trainer = example.TranslationTrainer(example.buildModel(), device="cpu")
        kept = trainer._applyModalityDropout(_batch(example))
        assert set(kept) == {"signal", "image_in"}

    def test_dropout_hides_encoder_inputs_but_always_keeps_one(self, example: ModuleType) -> None:
        trainer = example.TranslationTrainer(
            example.buildModel(), device="cpu", modality_dropout_p=1.0
        )
        batch = _batch(example)
        for _ in range(20):
            kept = trainer._applyModalityDropout(batch)
            assert 1 <= len(kept) <= 2
            assert set(kept) <= {"signal", "image_in"}

    def test_reconstruction_target_is_the_clean_image_not_the_input(
        self, example: ModuleType
    ) -> None:
        """With the encoder inputs zeroed, the loss must still be measured against the
        real `image_out`: replacing `image_out` by zeros must change it."""
        model = example.buildModel()
        model.eval()
        trainer = example.TranslationTrainer(model, device="cpu")
        batch = _batch(example)
        with torch.no_grad():
            real = trainer.computeLosses(batch, step=0).reconstruction.item()
            zeroed = trainer.computeLosses(
                {**batch, "image_out": torch.zeros_like(batch["image_out"])}, step=0
            ).reconstruction.item()
        assert real != pytest.approx(zeroed)

    def test_validation_pass_uses_every_input_and_restores_the_dropout(
        self, example: ModuleType
    ) -> None:
        model = example.buildModel()
        trainer = example.TranslationTrainer(model, device="cpu", modality_dropout_p=1.0)
        seen: list[set[str]] = []
        model.register_forward_pre_hook(lambda module, args: seen.append(set(args[0])))

        trainer.evaluate([_batch(example), _batch(example, seed=1)])

        assert seen and all(inputs == {"signal", "image_in"} for inputs in seen)
        assert trainer.modality_dropout_p == 1.0

    def test_training_runs_and_the_loss_is_finite(self, example: ModuleType) -> None:
        torch.manual_seed(0)
        trainer = example.TranslationTrainer(
            example.buildModel(), device="cpu", modality_dropout_p=0.3
        )
        dataset = example.toBatches(_batch(example, num_samples=16), batch_size=8)
        history = trainer.fit(dataset, num_epochs=2, val_dataloader=dataset)
        assert len(history) == 2
        assert all(np.isfinite(entry["val/loss/total"]) for entry in history)


class TestEvaluationHelpers:
    def test_subset_label_is_sorted_and_joined(self, example: ModuleType) -> None:
        assert example.subsetLabel(frozenset({"signal", "image_in"})) == "image_in + signal"
        assert example.subsetLabel(frozenset({"signal"})) == "signal"

    def test_summary_has_a_row_per_subset_plus_the_copy_the_input_baseline(
        self, example: ModuleType
    ) -> None:
        targets = torch.rand(6, 8, 8)
        per_cell = {
            "mse": 0.1,
            "rmse": 0.3,
            "mae": 0.2,
            "r2": 0.5,
            "pearson_r": 0.7,
        }
        metrics = {
            frozenset({"signal"}): {"image_out": per_cell},
            frozenset({"signal", "image_in"}): {"image_out": per_cell},
        }
        rows = example.summarizeMetrics(metrics, baseline_inputs=targets + 1.0, targets=targets)
        assert set(rows) == {"signal", "image_in + signal", "baseline: return image_in"}
        assert rows["signal"] == {"mse": 0.1, "r2": 0.5, "pearson_r": 0.7}
        assert rows["baseline: return image_in"]["mse"] == pytest.approx(1.0)

    def test_table_has_a_header_and_one_line_per_row(self, example: ModuleType) -> None:
        rows = {"signal": {"mse": 0.1, "r2": 0.5, "pearson_r": 0.7}}
        table = example.formatMetricsTable(rows).splitlines()
        assert len(table) == 2
        assert "mse" in table[0]


class TestScriptEndToEnd:
    def test_runs_as_a_subprocess_and_writes_every_output(self, tmp_path: Path) -> None:
        """Tiny sizes: this checks the wiring of the whole pipeline, not its accuracy."""
        output_dir = tmp_path / "run"
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(_REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
            ),
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--num-epochs",
                "2",
                "--num-train",
                "64",
                "--num-val",
                "32",
                "--num-test",
                "32",
                "--output-dir",
                str(output_dir),
            ],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert completed.returncode == 0, completed.stderr

        for name in (
            "best.pt",
            "metrics.csv",
            "metrics_by_input_subset.json",
            "translation_grid.png",
            "loss_curves.png",
        ):
            assert (output_dir / name).is_file(), f"missing output '{name}'"

        rows = json.loads((output_dir / "metrics_by_input_subset.json").read_text())
        assert set(rows) == {
            "signal",
            "image_in",
            "image_in + signal",
            "baseline: return image_in",
        }
        for row in rows.values():
            assert set(row) == {"mse", "r2", "pearson_r"}
            assert all(np.isfinite(value) for value in row.values())
