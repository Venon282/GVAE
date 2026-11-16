"""Freezing modules and learning-rate groups on the real pair model (roadmap P6-5, ADR 0055).

The unit tests (`tests/unit/test_freeze_modules.py`, `tests/unit/test_param_groups.py`) pin the
mechanism on a tiny model. Here it runs on the model the roadmap is about: the signal/image pair
of `configs/model/signal_image_pair.yaml`, whose CNN blocks use `BatchNorm1d` (signal) and
`BatchNorm2d` (image), built and trained from the config, with the pair-stage loss terms of P6-4.

What the roadmap's "done when" asks, and the tests that pin it:

- the parameters **and the buffers** of a frozen module are bit-identical after training: either
  side of the pair, both kinds of `BatchNorm`, under the alignment and the cross reconstruction
  terms, with the same modules changing when the callback is left out;
- an unfreeze epoch works across a resume: a run stopped before, at, and after the release epoch
  and resumed gives the loss curve and the weights of the uninterrupted one;
- the groups reach the optimizer and the checkpoint: through Hydra overrides, in the optimizer
  built from the config, in the file a run writes, kept by a learning-rate schedule, and checked
  when a run is resumed with other groups than the one that wrote the file.

It also pins what the P6-4 review left open: the cross reconstruction runs the decoders in
training mode (ADR 0054), so their `BatchNorm` statistics see the other modality's codes, and with
the decoders frozen they stay where they were.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.experiment import ExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    BetaScheduleConfig,
    LossTermConfig,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed
from tests.integration._warm_start_fixtures import loadPairExperiment, makeLinkedBatches
from tests.unit._tiny_pair_model import changedNames, namesUnder, snapshot

SEED = 5
TRAIN_BATCHES = 4
REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-9
WEIGHT_ATOL = 1e-6

SIDES = {
    "signal": ["encoders.signal", "decoders.signal"],
    "image": ["encoders.image", "decoders.image"],
}
GROUP_OVERRIDE = (
    "training.param_groups=[{modules:[encoders.signal,decoders.signal],lr_factor:0.1,name:signal}]"
)


def _terms(
    which: tuple[str, ...] = ("latent_alignment", "cross_reconstruction"), end_value: float = 0.3
) -> list[LossTermConfig]:
    """The pair-stage terms of P6-4 on the one pair of spaces, ramped from zero in 8 steps."""
    available: dict[str, dict[str, Any]] = {
        "latent_alignment": {"pairs": [["z_signal", "z_image"]], "distance": "wasserstein2_diag"},
        "cross_reconstruction": {"pairs": [["z_signal", "z_image"], ["z_image", "z_signal"]]},
    }
    return [
        LossTermConfig(
            name=name,
            schedule=BetaScheduleConfig(
                strategy="linear_warmup", kwargs={"warmup_steps": 8, "end_value": end_value}
            ),
            kwargs=available[name],
        )
        for name in which
    ]


def _experiment(
    output_dir: Path,
    overrides: list[str] | None = None,
    callbacks: dict[str, dict[str, Any]] | None = None,
    terms: list[LossTermConfig] | None = None,
    epochs: int = 3,
) -> ExperimentConfig:
    """The pair model at stage 3 settings (no dropout, sum reduction), a checkpoint every epoch."""
    experiment = loadPairExperiment(
        output_dir,
        [
            "training.device=cpu",
            "training.modality_dropout_p=0.0",
            "training.reconstruction_reduction=sum",
            "training.beta=0.1",
            f"training.num_epochs={epochs}",
            *(overrides or []),
        ],
    )
    experiment.training.loggers = []
    experiment.training.callbacks = {
        **(callbacks or {}),
        "checkpoint": {"directory": str(output_dir / "checkpoints")},
    }
    experiment.training.loss_terms = terms if terms is not None else _terms()
    return experiment


def _freeze(modules: list[str], until_epoch: int | None = None) -> dict[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {"modules": modules}
    if until_epoch is not None:
        kwargs["until_epoch"] = until_epoch
    return {"freeze_modules": kwargs}


def _build(experiment: ExperimentConfig, seed: int = SEED) -> tuple[Trainer, GlobalVae]:
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model)
    trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
    return trainer, model


def _fit(trainer: Trainer, num_epochs: int) -> list[dict[str, float]]:
    return trainer.fit(makeLinkedBatches(TRAIN_BATCHES, 0), num_epochs=num_epochs)


def _kinds(model: GlobalVae, prefix: str) -> set[str]:
    """The last name part of every parameter and buffer under `prefix`, e.g. `running_mean`."""
    return {name.rsplit(".", 1)[-1] for name in namesUnder(model, prefix)}


class TestAFrozenSideOfThePairIsBitIdentical:
    @pytest.mark.parametrize("side", ["signal", "image"])
    def test_parameters_and_buffers_do_not_change_and_the_other_side_trains(
        self, tmp_path: Path, side: str
    ) -> None:
        other = "image" if side == "signal" else "signal"
        experiment = _experiment(tmp_path, callbacks=_freeze(SIDES[side]))
        trainer, model = _build(experiment)
        before = snapshot(model)
        history = _fit(trainer, 3)
        after = snapshot(model)

        for prefix in SIDES[side]:
            assert _kinds(model, prefix) >= {
                "weight",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            }
            assert changedNames(before, after, prefix) == [], prefix
        for prefix in SIDES[other]:
            changed = {name.rsplit(".", 1)[-1] for name in changedNames(before, after, prefix)}
            assert {"weight", "running_mean", "running_var", "num_batches_tracked"} <= changed
        assert history[-1]["train/loss/latent_alignment"] > 0.0

    def test_without_the_callback_the_same_modules_change_buffers_included(
        self, tmp_path: Path
    ) -> None:
        trainer, model = _build(_experiment(tmp_path))
        before = snapshot(model)
        _fit(trainer, 3)
        for prefix in SIDES["signal"] + SIDES["image"]:
            changed = {n.rsplit(".", 1)[-1] for n in changedNames(before, snapshot(model), prefix)}
            assert {"weight", "running_mean", "running_var"} <= changed, prefix

    def test_the_alignment_term_pulls_the_free_side_toward_the_frozen_anchor(
        self, tmp_path: Path
    ) -> None:
        """One-sided alignment (roadmap P6-5, P7-5): the signal posterior stays, the image comes.

        The gap between the two posterior means is compared with the one of the same run without
        the term, since at initialization both posteriors sit near zero and the gap says nothing.
        """
        probe = makeLinkedBatches(1, 123)[0]

        def run(terms: list[LossTermConfig], folder: str) -> tuple[torch.Tensor, torch.Tensor]:
            trainer, model = _build(
                _experiment(tmp_path / folder, callbacks=_freeze(SIDES["signal"]), terms=terms)
            )
            model.eval()
            with torch.no_grad():
                start = model.encode(probe, use_mean=True)["latent_params"]["z_signal"][0].clone()
            _fit(trainer, 6)
            model.eval()
            with torch.no_grad():
                mu = model.encode(probe, use_mean=True)["latent_params"]
            assert torch.equal(mu["z_signal"][0], start)  # the anchor did not move
            return mu["z_signal"][0], mu["z_image"][0]

        signal, image_aligned = run(_terms(("latent_alignment",), end_value=5.0), "aligned")
        signal_free, image_free = run([], "unaligned")
        assert torch.equal(signal, signal_free)
        gap_aligned = (image_aligned - signal).pow(2).mean()
        gap_free = (image_free - signal).pow(2).mean()
        assert gap_aligned < 0.5 * gap_free


class TestTheCallbackFromTheCommandLine:
    def test_hydra_overrides_switch_the_freeze_on_and_the_run_honours_them(
        self, tmp_path: Path
    ) -> None:
        experiment = loadPairExperiment(
            tmp_path,
            [
                "training.device=cpu",
                "training.modality_dropout_p=0.0",
                "training.loggers=[]",
                "+training.callbacks.freeze_modules.modules=[encoders.image]",
                "+training.callbacks.freeze_modules.until_epoch=2",
            ],
        )
        assert experiment.training.callbacks["freeze_modules"] == {
            "modules": ["encoders.image"],
            "until_epoch": 2,
        }
        experiment.training.callbacks.pop("best_checkpoint")  # it monitors a validation loss
        experiment.training.loss_terms = _terms()
        trainer, model = _build(experiment)
        before = snapshot(model)
        _fit(trainer, 2)
        assert changedNames(before, snapshot(model), "encoders.image") == []
        assert changedNames(before, snapshot(model), "encoders.signal") != []

    def test_a_misspelled_option_is_refused_when_the_trainer_is_built(self, tmp_path: Path) -> None:
        experiment = _experiment(
            tmp_path, callbacks={"freeze_modules": {"module": ["encoders.image"]}}
        )
        with pytest.raises(TypeError, match="module"):
            _build(experiment)


class TestFrozenDecodersKeepTheirBatchNormStatistics:
    def test_under_the_cross_reconstruction_the_decoders_do_not_drift(self, tmp_path: Path) -> None:
        terms = _terms(("cross_reconstruction",))
        frozen_trainer, frozen = _build(
            _experiment(tmp_path / "frozen", callbacks=_freeze(["decoders.*"]), terms=terms)
        )
        free_trainer, free = _build(_experiment(tmp_path / "free", terms=terms))
        before_frozen, before_free = snapshot(frozen), snapshot(free)
        _fit(frozen_trainer, 3)
        _fit(free_trainer, 3)

        assert changedNames(before_frozen, snapshot(frozen), "decoders") == []
        drift = {
            n.rsplit(".", 1)[-1] for n in changedNames(before_free, snapshot(free), "decoders")
        }
        assert {"running_mean", "running_var", "num_batches_tracked"} <= drift

    def test_the_encoders_still_learn_through_the_frozen_decoders(self, tmp_path: Path) -> None:
        trainer, model = _build(
            _experiment(
                tmp_path, callbacks=_freeze(["decoders.*"]), terms=_terms(("cross_reconstruction",))
            )
        )
        before = snapshot(model)
        _fit(trainer, 2)
        assert changedNames(before, snapshot(model), "encoders") != []


def _interruptAndResume(
    tmp_path: Path, interrupted_after: int, total: int, until_epoch: int
) -> tuple[Trainer, Trainer, list[dict[str, float]]]:
    """Return (uninterrupted trainer, resumed trainer, resumed history), each run to `total`."""
    freeze = _freeze(SIDES["signal"], until_epoch)

    def experiment(name: str, resume_from: Path | None = None) -> ExperimentConfig:
        config = _experiment(tmp_path / name, callbacks=freeze, epochs=total)
        config.training.resume_from = str(resume_from) if resume_from is not None else None
        return config

    whole, _ = _build(experiment("whole"))
    _fit(whole, total)

    first, _ = _build(experiment("first"))
    _fit(first, interrupted_after)
    checkpoint = (
        tmp_path / "first" / "checkpoints" / f"checkpoint_epoch_{interrupted_after - 1:04d}.pt"
    )
    assert checkpoint.is_file()

    resumed_experiment = experiment("resumed", checkpoint)
    resumed, _ = _build(resumed_experiment, seed=999)
    remaining = resumeTrainerFromConfig(resumed, resumed_experiment.training)
    assert remaining == total - interrupted_after
    history = _fit(resumed, remaining)
    return whole, resumed, history


class TestTheReleaseEpochOnThePairModel:
    def test_frozen_before_the_release_epoch_and_training_after_it(self, tmp_path: Path) -> None:
        trainer, model = _build(
            _experiment(tmp_path, callbacks=_freeze(SIDES["signal"], 2), epochs=4)
        )
        initial = snapshot(model)
        _fit(trainer, 2)  # epochs 0 and 1
        assert changedNames(initial, snapshot(model), "encoders.signal") == []
        assert changedNames(initial, snapshot(model), "decoders.signal") == []
        _fit(trainer, 1)  # epoch 2 is the first one in which the side trains
        released = {
            n.rsplit(".", 1)[-1] for n in changedNames(initial, snapshot(model), "encoders.signal")
        }
        assert {"weight", "running_mean", "num_batches_tracked"} <= released

    @pytest.mark.parametrize("interrupted_after", [1, 2, 3])
    def test_a_resumed_run_gives_the_uninterrupted_one_before_at_and_after_the_release_epoch(
        self, tmp_path: Path, interrupted_after: int
    ) -> None:
        """`until_epoch=2` of 4: the interruption is in the frozen epochs, at the edge, or after."""
        whole, resumed, _ = _interruptAndResume(tmp_path, interrupted_after, 4, 2)

        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name],
                tensor,
                rtol=REL_TOLERANCE,
                atol=WEIGHT_ATOL,
                msg=name,
            )
        for epoch, (seen, wanted) in enumerate(zip(resumed.history, whole.history, strict=True)):
            for key, value in wanted.items():
                assert seen[key] == pytest.approx(value, rel=REL_TOLERANCE, abs=ABS_TOLERANCE), (
                    f"epoch {epoch}, {key}"
                )

    def test_the_run_resumed_inside_the_frozen_epochs_keeps_the_frozen_side_to_the_bit(
        self, tmp_path: Path
    ) -> None:
        _, resumed, _ = _interruptAndResume(tmp_path, 1, 2, 3)  # frozen all through, resumed once
        raw = torch.load(
            tmp_path / "first" / "checkpoints" / "checkpoint_epoch_0000.pt",
            map_location="cpu",
            weights_only=True,
        )["model_state_dict"]
        state = resumed.model.state_dict()
        for prefix in SIDES["signal"]:
            for name in namesUnder(resumed.model, prefix):
                assert torch.equal(state[name], raw[name]), name


def _groupLayout(trainer: Trainer) -> list[tuple[Any, Any, int]]:
    return [(g.get("name"), g.get("lr"), len(g["params"])) for g in trainer.optimizer.param_groups]


class TestTheGroupsReachTheOptimizerAndTheCheckpoint:
    def test_without_param_groups_nothing_changes(self, tmp_path: Path) -> None:
        experiment = _experiment(tmp_path)
        assert experiment.training.param_groups == []
        trainer, _ = _build(experiment)
        (group,) = trainer.optimizer.param_groups
        assert "name" not in group

    def test_a_hydra_override_builds_the_groups_of_the_optimizer(self, tmp_path: Path) -> None:
        experiment = _experiment(tmp_path, [GROUP_OVERRIDE])
        (entry,) = experiment.training.param_groups
        assert (list(entry.modules), entry.lr_factor, entry.name) == (
            ["encoders.signal", "decoders.signal"],
            0.1,
            "signal",
        )
        trainer, model = _build(experiment)
        signal, default = trainer.optimizer.param_groups
        base = experiment.training.optimizer.kwargs["lr"]
        assert (signal["name"], default["name"]) == ("signal", "default")
        assert signal["lr"] == pytest.approx(0.1 * base) and default["lr"] == pytest.approx(base)
        owned = {id(p) for p in signal["params"]}
        expected = {
            id(p) for n, p in model.named_parameters() if n.startswith(tuple(SIDES["signal"]))
        }
        assert owned == expected

    def test_the_file_a_run_writes_holds_the_groups(self, tmp_path: Path) -> None:
        experiment = _experiment(tmp_path, [GROUP_OVERRIDE], epochs=2)
        trainer, _ = _build(experiment)
        _fit(trainer, 2)
        saved = torch.load(
            tmp_path / "checkpoints" / "checkpoint_epoch_0001.pt",
            map_location="cpu",
            weights_only=True,
        )["optimizer_state_dict"]["param_groups"]
        assert [(g["name"], g["lr_factor"]) for g in saved] == [("signal", 0.1), ("default", 1.0)]
        assert saved[0]["lr"] == pytest.approx(0.1 * saved[1]["lr"])

    def test_a_schedule_callback_keeps_the_factor_all_through_the_run(self, tmp_path: Path) -> None:
        schedule = {"cosine_lr": {"total_steps": 40, "warmup_steps": 4}}
        trainer, _ = _build(_experiment(tmp_path, [GROUP_OVERRIDE], callbacks=schedule))
        for _ in range(3):
            _fit(trainer, 1)
            slow, fast = (g["lr"] for g in trainer.optimizer.param_groups)
            assert 0.0 < fast and slow == pytest.approx(0.1 * fast)

    def test_a_run_resumed_with_the_same_groups_is_the_uninterrupted_run(
        self, tmp_path: Path
    ) -> None:
        def experiment(name: str, resume_from: Path | None = None) -> ExperimentConfig:
            config = _experiment(tmp_path / name, [GROUP_OVERRIDE], epochs=3)
            config.training.resume_from = str(resume_from) if resume_from is not None else None
            return config

        whole, _ = _build(experiment("whole"))
        _fit(whole, 3)
        first, _ = _build(experiment("first"))
        _fit(first, 1)
        resumed_config = experiment(
            "resumed", tmp_path / "first/checkpoints/checkpoint_epoch_0000.pt"
        )
        resumed, _ = _build(resumed_config, seed=999)
        _fit(resumed, resumeTrainerFromConfig(resumed, resumed_config.training))

        assert _groupLayout(resumed) == _groupLayout(whole)
        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name],
                tensor,
                rtol=REL_TOLERANCE,
                atol=WEIGHT_ATOL,
                msg=name,
            )

    @pytest.mark.parametrize(
        "other",
        [
            [],
            [
                "training.param_groups=[{modules:[encoders.signal,decoders.signal],"
                "lr_factor:0.5,name:signal}]"
            ],
            [
                "training.param_groups=[{modules:[encoders.signal,decoders.signal],"
                "lr_factor:0.1,name:signal},{modules:[encoders.image],lr_factor:0.3}]"
            ],
        ],
    )
    def test_a_run_resumed_with_other_groups_is_refused_by_the_trainer_that_resumes(
        self, tmp_path: Path, other: list[str]
    ) -> None:
        first, _ = _build(_experiment(tmp_path / "first", [GROUP_OVERRIDE], epochs=2))
        _fit(first, 1)
        resumed_config = _experiment(tmp_path / "resumed", other, epochs=2)
        resumed_config.training.resume_from = str(
            tmp_path / "first/checkpoints/checkpoint_epoch_0000.pt"
        )
        resumed, _ = _build(resumed_config, seed=999)
        with pytest.raises(ValueError, match=r"`training.param_groups`.*restore_optimizer=False"):
            resumeTrainerFromConfig(resumed, resumed_config.training)

    def test_a_pattern_that_matches_no_module_stops_the_run_when_the_trainer_is_built(
        self, tmp_path: Path
    ) -> None:
        override = "training.param_groups=[{modules:[encoders.sinal],lr_factor:0.1}]"
        experiment = _experiment(tmp_path, [override])
        with pytest.raises(
            ValueError, match=r"param_groups\[0\].*'encoders.sinal' matches no module"
        ):
            _build(experiment)

    def test_a_freeze_of_modules_the_model_does_not_have_stops_the_run_before_its_first_step(
        self, tmp_path: Path
    ) -> None:
        trainer, model = _build(_experiment(tmp_path, callbacks=_freeze(["encoders.sound"])))
        before = snapshot(model)
        with pytest.raises(ValueError, match="'encoders.sound' matches no module"):
            _fit(trainer, 1)
        assert changedNames(before, snapshot(model)) == []
