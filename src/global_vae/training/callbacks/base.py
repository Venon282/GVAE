"""Training-loop callback interface (spec §10: hooks for per-step/per-epoch metrics).

This is the mechanism `Trainer` uses to expose what is happening
during training (loss values, epoch boundaries) to anything that wants
to react to it: a metrics logger (TensorBoard, CSV, W&B, MLflow, spec
§10), a checkpointer, an early-stopping rule, a learning-rate scheduler.
None of those concerns belong inside `Trainer` itself, which only knows
how to run the forward/backward loop; `TrainerCallback` is the seam
between the two.

Concrete callbacks self-register by name via `@registerCallback(name)`
(see `registry.py`), like every other pluggable piece of this framework,
so a training run can pick any number of them from config
(`TrainingConfig.callbacks`, `docs/adr/0022-callback-registry.md`)
without `Trainer` or the config layer knowing about any concrete class.

Unlike the single-strategy registries (encoders, decoders, fusion,
assemblers, regularizers, beta schedules), where exactly one strategy is
picked per role, callbacks are meant to be composed: a training run
commonly wants several active at once (say, a CSV logger, a
checkpointer, and early stopping together). `Trainer` therefore takes a
plain `list[TrainerCallback]`, and the config selects several registry
names at once. This mirrors the callback pattern used by most training
frameworks (Keras, PyTorch Lightning, Hugging Face `Trainer`).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer


class TrainerCallback:
    """Base class for training-loop callbacks.

    Every hook below is a no-op by default, so a concrete callback
    only needs to override the events it actually cares about (e.g. a
    step-level metrics logger only overrides `onStepEnd`; a
    checkpointer only overrides `onEpochEnd`). This class is
    intentionally not an `ABC` with `@abstractmethod` hooks: unlike
    `AbstractEncoder`/`AbstractFusion`/etc., which describe a contract
    every concrete strategy must fully implement, a callback is
    defined by which events it *chooses* to react to, and leaving the
    rest as no-ops is the normal, expected case, not an incomplete
    implementation.

    A callback may also act on the `Trainer` it is handed, not only
    observe it. Two such levers exist, both used by built-in callbacks:
    `trainer.optimizer` (e.g. `ReduceLrOnPlateau` lowers its learning
    rate) and `trainer.should_stop` (e.g. `EarlyStopping` sets it to
    `True`; `Trainer.fit` then ends the run after every callback has
    finished handling the current epoch).
    """

    def onTrainBegin(self, trainer: "Trainer") -> None:
        """Called once, before the first epoch of `Trainer.fit` starts.

        Args:
            trainer: The `Trainer` instance running this training run.
        """

    def onEpochBegin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of every training epoch, before any step runs.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch about to start (0-based).
        """

    def onStepEnd(self, trainer: "Trainer", step: int, metrics: dict[str, float]) -> None:
        """Called after every optimizer step, with that step's metrics.

        Args:
            trainer: The `Trainer` instance running this training run.
            step: Global step index (monotonically increasing across
                every epoch of this `Trainer` instance's lifetime, not
                reset per epoch), matching the step argument every
                beta schedule (`training/beta_schedules/`) is resolved
                against for this same step.
            metrics: This step's scalar metrics (e.g.
                `"loss/total"`, `"loss/reconstruction"`,
                `"loss/regularization"`). Keys are stable across calls
                so a logger can treat this as a flat, appendable
                record; see `Trainer.computeLosses` for exactly which
                keys are produced.
        """

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Called after every training epoch (and its validation pass, if any).

        Callbacks are called in the order they were given to `Trainer`
        (for a config-built run: the order of `TrainingConfig.callbacks`).
        Setting `trainer.should_stop = True` here does not interrupt the
        callbacks that come after this one for the same epoch: they all
        run, then `Trainer.fit` stops before starting the next epoch.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: This epoch's metrics, averaged over every step in
                the epoch. Training metrics are prefixed `"train/"`;
                if a validation dataloader was given to `Trainer.fit`,
                validation metrics are also present, prefixed
                `"val/"`.
        """

    def onTrainEnd(self, trainer: "Trainer") -> None:
        """Called once, after `Trainer.fit` finishes.

        Always called, including when training exits early via an
        exception (e.g. `KeyboardInterrupt`) or via `trainer.should_stop`:
        `Trainer.fit` runs the training loop in a `try`/`finally` so every
        callback gets a chance to flush/close cleanly (e.g. a file-based
        logger closing its file handle) regardless of how training ended.

        Args:
            trainer: The `Trainer` instance running this training run.
        """
