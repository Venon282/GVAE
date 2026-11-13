"""Warm start of the pair model from two solo checkpoints (roadmap P6-2, ADR 0052).

What is tested, on two real solo checkpoints written by the real solo recipes
(`tests/integration/_warm_start_fixtures.py`) and on the pair model of
`configs/model/signal_image_pair.yaml`:

- the exactness the roadmap asks for: with both checkpoints loaded, in eval mode, the pair model's
  `use_mean=True` reconstruction of each modality from its own input equals the solo model's bit
  for bit, and so does every copied tensor, buffers included;
- the refusals, each by name and before any weight moves: a wrong modality, a wrong latent width,
  a changed module config (with shapes that still agree), a missing manifest, a manifest of a pair
  model, a missing or unexpected key, a wrong shape, a module the file does not hold, a collapsed
  solo encoder, a module loaded twice, a name that is not a module;
- that a failure leaves the model as it was (nothing is copied, and what an earlier entry copied
  is put back when a later one is refused by the collapse guard);
- that only weights move: a trainer built after the warm start has step 0, no history and an
  optimizer without state, and the probe of the collapse guard touches neither the random-number
  stream, the training mode nor a weight;
- the config layer: `applyWarmStartFromConfig` does nothing for an empty list and for a resumed
  run, and the pair stage's default beta is the constant `training.beta`.

The same wiring through `scripts/train.py` is in `test_warm_start_script.py`.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from global_vae.config.experiment import ExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    WarmStartEntryConfig,
    applyWarmStartFromConfig,
    buildTrainerFromConfig,
    computeModuleDigests,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import readCheckpointWeights, saveCheckpoint
from global_vae.training.stage_manifest import buildStageManifest
from global_vae.training.warm_start import PROBE_MAX_SAMPLES, WarmStartEntry, applyWarmStart
from tests.integration._warm_start_fixtures import (
    BATCH_SIZE,
    NUM_TRAIN_BATCHES,
    buildPairModel,
    loadExperiment,
    loadPairExperiment,
    makeImages,
    makePairedBatches,
    makeSignals,
    trainSoloCheckpoint,
)

MODALITIES = ("signal", "image")
SOLO_SPACE_WIDTH = 16

State = dict[str, torch.Tensor]


@pytest.fixture(scope="module")
def solo_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The `best.pt` of the signal and of the image solo stage, trained once for the module."""
    root = tmp_path_factory.mktemp("solo")
    return {modality: trainSoloCheckpoint(modality, root / modality) for modality in MODALITIES}


@pytest.fixture(scope="module")
def solo_models(
    solo_checkpoints: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, GlobalVae]:
    """The two solo models, rebuilt from their experiments and loaded from the checkpoints."""
    root = tmp_path_factory.mktemp("solo_models")
    models = {}
    for modality in MODALITIES:
        experiment = loadExperiment(f"experiment/pair_solo_{modality}", root)
        model = buildModelFromConfig(experiment.model)
        model.load_state_dict(readCheckpointWeights(solo_checkpoints[modality]).state_dict)
        models[modality] = model.eval()
    return models


def _entries(paths: dict[str, Path], **fields: Any) -> list[WarmStartEntry]:
    return [WarmStartEntry(paths[modality], (modality,), **fields) for modality in MODALITIES]


def _snapshot(model: GlobalVae) -> State:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assertSame(left: State, right: State) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _probeBatches() -> list[dict[str, torch.Tensor]]:
    return makePairedBatches(NUM_TRAIN_BATCHES, seed=5)


def _rewrite(source: Path, target: Path, change: Callable[[dict[str, Any]], None]) -> Path:
    """Copy a checkpoint to `target` after `change` edited its dict, in the safe mode."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    change(checkpoint)
    torch.save(checkpoint, target)
    return target


def _refusal(
    model: GlobalVae,
    entries: list[WarmStartEntry],
    experiment: ExperimentConfig,
    *,
    probe: Any = "default",
) -> str:
    """Run a warm start that must be refused and return the message.

    Also checks that the refusal left every tensor of the model as it was.
    """
    before = _snapshot(model)
    loader = _probeBatches() if probe == "default" else probe
    with pytest.raises(ValueError) as caught:
        applyWarmStart(
            model,
            entries,
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=loader,
        )
    _assertSame(before, _snapshot(model))
    return str(caught.value)


class TestTheCopyIsExact:
    def test_use_mean_reconstruction_equals_the_solo_models_bit_for_bit(
        self, solo_checkpoints: dict[str, Path], solo_models: dict[str, GlobalVae]
    ) -> None:
        pair, experiment = buildPairModel()
        applyWarmStart(
            pair,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        pair.eval()
        batch = makePairedBatches(1, seed=99)[0]
        with torch.no_grad():
            for modality in MODALITIES:
                solo_out = solo_models[modality]({modality: batch[modality]}, use_mean=True)
                pair_out = pair({modality: batch[modality]}, use_mean=True)
                assert torch.equal(
                    pair_out["reconstructions"][modality], solo_out["reconstructions"][modality]
                ), modality
            both = pair(batch, use_mean=True)
            for modality in MODALITIES:
                solo_out = solo_models[modality]({modality: batch[modality]}, use_mean=True)
                assert torch.equal(
                    both["reconstructions"][modality], solo_out["reconstructions"][modality]
                ), modality

    def test_the_posterior_of_each_space_is_the_solo_posterior(
        self, solo_checkpoints: dict[str, Path], solo_models: dict[str, GlobalVae]
    ) -> None:
        pair, experiment = buildPairModel()
        applyWarmStart(
            pair,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        pair.eval()
        batch = makePairedBatches(1, seed=98)[0]
        space = {"signal": "z_signal", "image": "z_image"}
        with torch.no_grad():
            params = pair.encode(batch, use_mean=True)["latent_params"]
            for modality in MODALITIES:
                solo_params = solo_models[modality].encode(
                    {modality: batch[modality]}, use_mean=True
                )["latent_params"]["z_fused"]
                assert torch.equal(params[space[modality]][0], solo_params[0])
                assert torch.equal(params[space[modality]][1], solo_params[1])

    def test_every_tensor_of_a_copied_module_is_the_solo_tensor_buffers_included(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        initial = _snapshot(pair)
        report = applyWarmStart(
            pair,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        after = _snapshot(pair)
        copied = 0
        for modality in MODALITIES:
            saved = readCheckpointWeights(solo_checkpoints[modality]).state_dict
            for key, value in saved.items():
                assert torch.equal(after[key], value), key
                copied += 1
            assert any("running_mean" in key for key in saved), "the encoder has batch norm"
        assert copied == sum(entry.num_tensors for entry in report.entries)
        assert copied == len(initial)
        assert any(not torch.equal(initial[key], after[key]) for key in initial)

    def test_only_what_the_entry_names_is_copied(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        initial = _snapshot(pair)
        applyWarmStart(
            pair,
            [WarmStartEntry(solo_checkpoints["signal"], ("encoders.signal",))],
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        after = _snapshot(pair)
        saved = readCheckpointWeights(solo_checkpoints["signal"]).state_dict
        for key in after:
            if key.startswith("encoders.signal."):
                assert torch.equal(after[key], saved[key]), key
            else:
                assert torch.equal(after[key], initial[key]), key

    def test_a_decoder_alone_is_copied_and_needs_no_probe(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        initial = _snapshot(pair)
        report = applyWarmStart(
            pair,
            [WarmStartEntry(solo_checkpoints["image"], ("decoders.image",))],
            module_digests=computeModuleDigests(experiment.model),
        )
        after = _snapshot(pair)
        saved = readCheckpointWeights(solo_checkpoints["image"]).state_dict
        assert report.entries[0].modules == ["decoders.image"]
        assert report.entries[0].active_units is None
        for key in after:
            if key.startswith("decoders.image."):
                assert torch.equal(after[key], saved[key]), key
            else:
                assert torch.equal(after[key], initial[key]), key

    def test_the_report_says_what_was_copied_and_how_active_the_encoder_is(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        report = applyWarmStart(
            pair,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        assert [entry.modules for entry in report.entries] == [
            ["encoders.signal", "decoders.signal"],
            ["encoders.image", "decoders.image"],
        ]
        assert [entry.stage for entry in report.entries] == ["solo_signal", "solo_image"]
        assert [entry.latent_space for entry in report.entries] == ["z_signal", "z_image"]
        for entry in report.entries:
            assert entry.active_units is not None and entry.active_units >= 1
            assert entry.num_dimensions == SOLO_SPACE_WIDTH
            assert entry.num_probe_samples == BATCH_SIZE * NUM_TRAIN_BATCHES

    def test_an_empty_list_reads_nothing_and_changes_nothing(self) -> None:
        pair, _ = buildPairModel()
        before = _snapshot(pair)
        report = applyWarmStart(pair, [])
        assert report.entries == []
        _assertSame(before, _snapshot(pair))


class TestOnlyTheWeightsMove:
    def test_a_trainer_built_afterwards_starts_a_new_run(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        experiment = loadPairExperiment(tmp_path)
        experiment.training.warm_start = [
            WarmStartEntryConfig(checkpoint=str(solo_checkpoints[modality]), modules=[modality])
            for modality in MODALITIES
        ]
        model = buildModelFromConfig(experiment.model)
        applyWarmStartFromConfig(
            model, experiment.training, experiment.model, probe_dataloader=_probeBatches()
        )
        trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
        assert trainer.global_step == 0
        assert trainer.start_epoch == 0
        assert trainer.history == []
        assert trainer.optimizer.state == {}

        manifest = trainer.stage_manifest
        assert manifest is not None
        assert manifest.stage == "pair"
        assert manifest.modalities == ("signal", "image")
        assert manifest.latent_spaces == {"z_signal": 16, "z_image": 16}
        assert manifest.module_digests == computeModuleDigests(experiment.model)

    def test_an_optimizer_built_before_the_warm_start_sees_the_new_weights(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        experiment = loadPairExperiment(tmp_path)
        model = buildModelFromConfig(experiment.model)
        trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
        applyWarmStart(
            model,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        saved = readCheckpointWeights(solo_checkpoints["signal"]).state_dict
        parameters = dict(model.named_parameters())
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert id(parameters["encoders.signal.to_mu.weight"]) in optimized
        assert torch.equal(
            parameters["encoders.signal.to_mu.weight"], saved["encoders.signal.to_mu.weight"]
        )

    def test_the_probe_leaves_the_random_stream_the_mode_and_the_weights_alone(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        digests = computeModuleDigests(experiment.model)
        pair.train()
        data = TensorDataset(
            makeSignals(64, torch.Generator().manual_seed(3)),
            makeImages(64, torch.Generator().manual_seed(4)),
        )
        loader = DataLoader(
            data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda rows: {
                "signal": torch.stack([row[0] for row in rows]),
                "image": torch.stack([row[1] for row in rows]),
            },
        )
        entries = _entries(solo_checkpoints)

        torch.manual_seed(11)
        applyWarmStart(pair, entries, module_digests=digests, probe_dataloader=None)
        expected_next = torch.rand(3)
        torch.manual_seed(11)
        applyWarmStart(pair, entries, module_digests=digests, probe_dataloader=loader)
        assert torch.equal(torch.rand(3), expected_next)
        assert pair.training is True

    def test_the_probe_stops_at_the_sample_cap(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()

        class Counting:
            def __init__(self) -> None:
                self.batches = makePairedBatches(PROBE_MAX_SAMPLES // BATCH_SIZE + 6, seed=6)
                self.yielded = 0

            def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
                for batch in self.batches:
                    self.yielded += 1
                    yield batch

        loader = Counting()
        report = applyWarmStart(
            pair,
            _entries(solo_checkpoints),
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=loader,
        )
        assert loader.yielded == PROBE_MAX_SAMPLES // BATCH_SIZE
        assert report.entries[0].num_probe_samples == PROBE_MAX_SAMPLES


class TestAWrongFileIsRefusedByName:
    def test_the_image_checkpoint_where_the_signal_was_meant(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["image"], ("signal",))]
        message = _refusal(pair, entries, experiment)
        assert "training.warm_start[0]" in message
        assert str(solo_checkpoints["image"]) in message
        assert "solo model of ['image']" in message
        assert "modality 'signal'" in message
        assert "wrong file" in message

    def test_a_latent_space_of_another_width(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel(["model.latent.spaces.z_signal.dim=8"])
        message = _refusal(pair, _entries(solo_checkpoints), experiment)
        assert "solo latent space 'z_fused'" in message
        assert "16 wide" in message
        assert "'z_signal' is 8 wide" in message

    def test_a_module_config_that_changed_while_the_shapes_did_not(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel(["+model.encoders.signal.kwargs.global_pool=max"])
        message = _refusal(pair, _entries(solo_checkpoints), experiment)
        assert "config of 'encoders.signal'" in message
        assert "not the one the checkpoint" in message
        assert str(solo_checkpoints["signal"]) in message

        # The shapes agree, so without the manifest the copy goes through: the digest is what
        # tells the two encoders apart.
        applyWarmStart(
            pair,
            _entries(solo_checkpoints, check_manifest=False),
            probe_dataloader=_probeBatches(),
        )

    def test_a_checkpoint_with_no_manifest(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        bare = _rewrite(
            solo_checkpoints["signal"],
            tmp_path / "bare.pt",
            lambda checkpoint: checkpoint.pop("stage_manifest"),
        )
        pair, experiment = buildPairModel()
        message = _refusal(pair, [WarmStartEntry(bare, ("signal",))], experiment)
        assert "no stage manifest" in message
        assert "check_manifest: false" in message

        report = applyWarmStart(
            pair,
            [WarmStartEntry(bare, ("signal",), check_manifest=False)],
            probe_dataloader=_probeBatches(),
        )
        assert report.entries[0].stage is None
        assert report.entries[0].active_units is not None

    def test_a_manifest_without_module_digests(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        old = _rewrite(
            solo_checkpoints["signal"],
            tmp_path / "old.pt",
            lambda checkpoint: checkpoint["stage_manifest"].pop("module_digests"),
        )
        pair, experiment = buildPairModel()
        message = _refusal(pair, [WarmStartEntry(old, ("signal",))], experiment)
        assert "no config digest for 'encoders.signal'" in message

    def test_the_checkpoint_of_a_pair_model_is_not_a_solo_checkpoint(self, tmp_path: Path) -> None:
        pair, experiment = buildPairModel()
        manifest = buildStageManifest(
            pair, "pair", experiment.model, computeModuleDigests(experiment.model)
        )
        path = tmp_path / "pair.pt"
        saveCheckpoint(path, pair, stage_manifest=manifest)
        target, _ = buildPairModel()
        message = _refusal(target, [WarmStartEntry(path, ("signal",))], experiment)
        assert "not a solo model" in message
        assert "z_signal" in message

    def test_digests_that_do_not_belong_to_the_model_are_refused(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        wrong = {
            path: digest
            for path, digest in computeModuleDigests(experiment.model).items()
            if path != "decoders.signal"
        }
        with pytest.raises(ValueError, match="no config digest was given for the pair model's"):
            applyWarmStart(pair, _entries(solo_checkpoints), module_digests=wrong)

    def test_a_file_that_does_not_exist(self, tmp_path: Path) -> None:
        pair, experiment = buildPairModel()
        with pytest.raises(FileNotFoundError, match="missing.pt"):
            applyWarmStart(
                pair,
                [WarmStartEntry(tmp_path / "missing.pt", ("signal",))],
                module_digests=computeModuleDigests(experiment.model),
            )


class TestKeysAndShapesAreExact:
    def test_a_wrong_shape(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel(["model.encoders.signal.kwargs.kernel_sizes=3"])
        entries = _entries(solo_checkpoints, check_manifest=False)
        message = _refusal(pair, entries, experiment)
        assert "shape mismatch for 'encoders.signal.conv.0.weight'" in message
        assert "(8, 1, 5)" in message
        assert "(8, 1, 3)" in message

    def test_a_missing_key(self, solo_checkpoints: dict[str, Path], tmp_path: Path) -> None:
        cut = _rewrite(
            solo_checkpoints["signal"],
            tmp_path / "cut.pt",
            lambda checkpoint: checkpoint["model_state_dict"].pop("encoders.signal.to_mu.bias"),
        )
        pair, experiment = buildPairModel()
        message = _refusal(pair, [WarmStartEntry(cut, ("signal",))], experiment)
        assert "no tensor for 1 key(s) of 'encoders.signal'" in message
        assert "'encoders.signal.to_mu.bias'" in message

    def test_an_unexpected_key(self, solo_checkpoints: dict[str, Path], tmp_path: Path) -> None:
        extra = _rewrite(
            solo_checkpoints["signal"],
            tmp_path / "extra.pt",
            lambda checkpoint: checkpoint["model_state_dict"].update(
                {"decoders.signal.extra.weight": torch.zeros(2)}
            ),
        )
        pair, experiment = buildPairModel()
        message = _refusal(pair, [WarmStartEntry(extra, ("signal",))], experiment)
        assert "1 key(s) that the pair model's 'decoders.signal' does not have" in message
        assert "'decoders.signal.extra.weight'" in message

    def test_a_module_the_file_does_not_hold(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["image"], ("signal",), check_manifest=False)]
        message = _refusal(pair, entries, experiment)
        assert "no tensor of module 'encoders.signal'" in message
        assert "encoders.image" in message


class TestACollapsedSoloModelIsRefused:
    @pytest.fixture
    def collapsed(self, solo_checkpoints: dict[str, Path], tmp_path: Path) -> Path:
        """The signal checkpoint with its encoder zeroed: every sample gets the same code."""

        def zero(checkpoint: dict[str, Any]) -> None:
            state = checkpoint["model_state_dict"]
            for key, value in state.items():
                if key.startswith("encoders.") and value.is_floating_point():
                    state[key] = torch.zeros_like(value)

        return _rewrite(solo_checkpoints["signal"], tmp_path / "collapsed.pt", zero)

    def test_it_is_refused_with_the_numbers_and_the_cause(self, collapsed: Path) -> None:
        pair, experiment = buildPairModel()
        message = _refusal(pair, [WarmStartEntry(collapsed, ("signal",))], experiment)
        assert "looks collapsed" in message
        assert str(collapsed) in message
        assert "0 of 16 dimensions of latent space 'z_signal'" in message
        assert "64 training samples" in message
        assert "choose-beta" in message
        assert "min_active_units: 0" in message

    def test_the_bar_is_per_entry(self, collapsed: Path, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        digests = computeModuleDigests(experiment.model)
        report = applyWarmStart(
            pair,
            [WarmStartEntry(collapsed, ("signal",), min_active_units=0)],
            module_digests=digests,
            probe_dataloader=_probeBatches(),
        )
        assert report.entries[0].active_units is None

        pair, experiment = buildPairModel()
        strict = [WarmStartEntry(solo_checkpoints["signal"], ("signal",), min_active_units=17)]
        message = _refusal(pair, strict, experiment)
        assert "requires at least 17" in message

    def test_a_refusal_after_a_copy_puts_the_copy_back(
        self, collapsed: Path, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        # The image entry is the good one and comes first, the collapsed signal one second: the
        # guard runs after the copy, so the image modules are restored from the snapshot.
        pair, experiment = buildPairModel()
        entries = [
            WarmStartEntry(solo_checkpoints["image"], ("image",)),
            WarmStartEntry(collapsed, ("signal",)),
        ]
        message = _refusal(pair, entries, experiment)
        assert "training.warm_start[1]" in message

    def test_without_data_the_guard_cannot_run_and_says_so(
        self, collapsed: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pair, experiment = buildPairModel()
        with caplog.at_level(logging.WARNING, logger="global_vae.training.warm_start"):
            report = applyWarmStart(
                pair,
                [WarmStartEntry(collapsed, ("signal",))],
                module_digests=computeModuleDigests(experiment.model),
            )
        assert report.entries[0].active_units is None
        assert "collapse guard did not run" in caplog.text

    def test_a_loader_without_the_modality_is_refused(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        batches = [{"image": batch["image"]} for batch in _probeBatches()]
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal",))]
        message = _refusal(pair, entries, experiment, probe=batches)
        assert "no input for the encoder of latent space 'z_signal'" in message

    def test_a_loader_of_a_single_sample_cannot_measure_a_variance(
        self, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        one = [{key: value[:1] for key, value in _probeBatches()[0].items()}]
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal",))]
        message = _refusal(pair, entries, experiment, probe=one)
        assert "at least 2 training samples" in message


class TestTheEntriesAreChecked:
    def test_a_module_loaded_twice(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        entries = [
            WarmStartEntry(solo_checkpoints["signal"], ("signal",)),
            WarmStartEntry(solo_checkpoints["signal"], ("encoders.signal",)),
        ]
        message = _refusal(pair, entries, experiment)
        assert "'encoders.signal' is also loaded by training.warm_start[0]" in message

    def test_two_modalities_in_one_entry(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal", "image"))]
        message = _refusal(pair, entries, experiment)
        assert "belong to several modalities" in message
        assert "one entry per checkpoint" in message

    def test_a_module_named_twice_in_one_entry(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal", "encoders.signal"))]
        assert "name a module twice" in _refusal(pair, entries, experiment)

    @pytest.mark.parametrize("name", ["latent.z_signal", "encoders.", "regularizers.signal"])
    def test_a_name_that_is_not_a_module_path(
        self, name: str, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["signal"], (name,))]
        message = _refusal(pair, entries, experiment)
        assert f"module '{name}' is neither a modality name nor a path" in message

    @pytest.mark.parametrize("name", ["spectrum", "encoders.spectrum"])
    def test_a_modality_the_pair_model_does_not_have(
        self, name: str, solo_checkpoints: dict[str, Path]
    ) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["signal"], (name,))]
        message = _refusal(pair, entries, experiment)
        assert "the pair model has no module 'encoders.spectrum'" in message
        assert "['image', 'signal']" in message

    def test_an_unknown_latent_space(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal",), latent_space="z_nope")]
        message = _refusal(pair, entries, experiment)
        assert "latent_space 'z_nope' is not a latent space of the pair model" in message
        assert "['z_image', 'z_signal']" in message

    def test_an_explicit_latent_space_is_used(self, solo_checkpoints: dict[str, Path]) -> None:
        pair, experiment = buildPairModel()
        report = applyWarmStart(
            pair,
            [WarmStartEntry(solo_checkpoints["signal"], ("signal",), latent_space="z_signal")],
            module_digests=computeModuleDigests(experiment.model),
            probe_dataloader=_probeBatches(),
        )
        assert report.entries[0].latent_space == "z_signal"

    def test_an_encoder_that_feeds_several_spaces_needs_the_space_named(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        experiment = loadExperiment(
            "experiment/pair_solo_signal", tmp_path, ["model=signal_image_shared_private"]
        )
        model = buildModelFromConfig(experiment.model)
        entries = [WarmStartEntry(solo_checkpoints["signal"], ("signal",))]
        message = _refusal(model, entries, experiment)
        assert "feeds 2 latent space(s)" in message
        assert "Name it with latent_space" in message


class TestTheConfigLayer:
    def test_an_empty_list_applies_nothing(self, tmp_path: Path) -> None:
        experiment = loadPairExperiment(tmp_path)
        assert experiment.training.warm_start == []
        model = buildModelFromConfig(experiment.model)
        before = _snapshot(model)
        report = applyWarmStartFromConfig(model, experiment.training, experiment.model)
        assert report.entries == []
        _assertSame(before, _snapshot(model))

    def test_a_resumed_run_does_not_read_the_solo_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        experiment = loadPairExperiment(tmp_path)
        experiment.training.warm_start = [
            WarmStartEntryConfig(checkpoint=str(tmp_path / "gone.pt"), modules=["signal"])
        ]
        experiment.training.resume_from = str(tmp_path / "checkpoint_epoch_0001.pt")
        model = buildModelFromConfig(experiment.model)
        before = _snapshot(model)
        with caplog.at_level(logging.INFO, logger="global_vae.config.training"):
            report = applyWarmStartFromConfig(model, experiment.training, experiment.model)
        assert report.entries == []
        assert "not applied" in caplog.text
        _assertSame(before, _snapshot(model))

    def test_an_entry_of_the_config_reaches_the_loader(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        experiment = loadPairExperiment(tmp_path)
        experiment.training.warm_start = [
            WarmStartEntryConfig(
                checkpoint=str(solo_checkpoints["signal"]),
                modules=["encoders.signal", "decoders.signal"],
                latent_space="z_signal",
            )
        ]
        model = buildModelFromConfig(experiment.model)
        report = applyWarmStartFromConfig(
            model, experiment.training, experiment.model, probe_dataloader=_probeBatches()
        )
        assert [entry.modules for entry in report.entries] == [
            ["encoders.signal", "decoders.signal"]
        ]

    def test_an_entry_with_nothing_to_copy_names_its_place(self, tmp_path: Path) -> None:
        experiment = loadPairExperiment(tmp_path)
        experiment.training.warm_start = [
            WarmStartEntryConfig(checkpoint="a.pt", modules=["signal"]),
            WarmStartEntryConfig(checkpoint="b.pt", modules=[]),
        ]
        model = buildModelFromConfig(experiment.model)
        with pytest.raises(ValueError, match=r"training\.warm_start\[1\]: modules must be"):
            applyWarmStartFromConfig(model, experiment.training, experiment.model)

    def test_without_a_model_config_the_module_digests_are_not_checked_and_it_says_so(
        self, solo_checkpoints: dict[str, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        experiment = loadPairExperiment(tmp_path, ["+model.encoders.signal.kwargs.global_pool=max"])
        experiment.training.warm_start = [
            WarmStartEntryConfig(checkpoint=str(solo_checkpoints["signal"]), modules=["signal"])
        ]
        model = buildModelFromConfig(experiment.model)
        with caplog.at_level(logging.WARNING, logger="global_vae.training.warm_start"):
            applyWarmStartFromConfig(
                model, experiment.training, model_config=None, probe_dataloader=_probeBatches()
            )
        assert "config of the copied modules was not checked" in caplog.text


class TestBetaStaysConstantInTheBeginningOfStageThree:
    def test_the_pair_stage_has_no_schedule_and_the_solo_end_value_as_beta(
        self, tmp_path: Path
    ) -> None:
        solo = loadExperiment("experiment/pair_solo_signal", tmp_path)
        pair = loadPairExperiment(tmp_path)
        end_value = solo.training.beta_schedules["z_fused"].kwargs["end_value"]
        assert pair.training.beta_schedules == {}
        assert pair.training.beta == end_value

        trainer = buildTrainerFromConfig(
            buildModelFromConfig(pair.model), pair.training, model_config=pair.model
        )
        assert trainer.beta_schedules == {}
        assert trainer.beta == end_value

    def test_the_solo_schedule_of_the_default_group_names_no_space_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pair = loadExperiment(
            "experiment/pair_solo_signal",
            tmp_path,
            ["model=signal_image_pair", "training.stage=pair"],
        )
        model = buildModelFromConfig(pair.model)
        with caplog.at_level(logging.WARNING, logger="global_vae.config.training"):
            buildTrainerFromConfig(model, pair.training, model_config=pair.model)
        assert "z_fused" in caplog.text
        assert "z_signal" in caplog.text
