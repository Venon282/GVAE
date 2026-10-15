"""How a learning rate appears in `Trainer`'s metrics (ADR 0028).

A callback that changes the optimizer's learning rate (`ReduceLrOnPlateau`, a warmup or a cosine
schedule of your own) declares it with `TrainerCallback.acts_on_learning_rate`. When any callback
of a run does, `Trainer.fitEpoch` adds the learning rate each step used to that step's metrics,
under the key `"lr"`. The epoch metrics then carry its mean over the epoch's steps as `"train/lr"`,
so it shows in the epoch log line, in `Trainer.history`, in every logger and in the checkpoint
history that `scripts/visualize_latent.py` plots.

With several parameter groups, each gets its own key, `"lr/group0"`, `"lr/group1"`, and so on.
This module holds the three pieces that spell those keys out, so the trainer, the log formatting
and the plots agree on them.
"""

from torch.optim import Optimizer

LEARNING_RATE_KEY = "lr"


def learningRateMetrics(optimizer: Optimizer) -> dict[str, float]:
    """Read the learning rate of every parameter group of `optimizer`.

    Args:
        optimizer: The optimizer whose `param_groups` are read at call time, so the value is
            the one the next `optimizer.step()` uses.

    Returns:
        `{"lr": value}` for an optimizer with one parameter group, otherwise
        `{"lr/group0": value, "lr/group1": value, ...}`, one entry per group in order.
    """
    groups = optimizer.param_groups
    if len(groups) == 1:
        return {LEARNING_RATE_KEY: float(groups[0]["lr"])}
    return {
        f"{LEARNING_RATE_KEY}/group{index}": float(group["lr"])
        for index, group in enumerate(groups)
    }


def isLearningRateKey(key: str) -> bool:
    """Whether a metric key is a learning rate.

    Args:
        key: A step or epoch metric key.

    Returns:
        `True` for `"lr"`, `"lr/group1"` and their epoch forms `"train/lr"` and
        `"train/lr/group1"`: any key with an `lr` path component. `False` for anything else,
        `"train/loss/total"` included.
    """
    return LEARNING_RATE_KEY in key.split("/")
