"""Shared plateau-detection algorithm for `ReduceLrOnPlateau` and `EarlyStopping`.

Both callbacks watch one metric for improvement and take an action once it
has stopped improving for `patience` epochs. "Has this metric stopped
improving" is exactly the algorithm `torch.optim.lr_scheduler.ReduceLROnPlateau`
itself uses (`mode`, `threshold`, `threshold_mode`, `cooldown`, `patience`):
this project's own `ReduceLrOnPlateau` (`reduce_lr_on_plateau.py`) mirrors that
class directly, and `EarlyStopping` (`early_stopping.py`) reuses the identical
plateau signal to stop training instead of lowering the learning rate, since
plain `torch` ships no early-stopping class of its own. `PlateauTracker` is the
one place this comparison logic (including its floating-point-sensitive
`threshold`/`threshold_mode` handling) is written down, so both callbacks stay
exactly in sync with each other and with torch's own semantics instead of each
re-deriving it.

Leading underscore: an internal helper shared by two sibling modules in this
subpackage, not itself a registered callback or part of the public API (the
same convention already used for private test helpers in this codebase, e.g.
`tests/integration/_script_fixtures.py`).
"""

import math
from dataclasses import dataclass, field


@dataclass
class PlateauTracker:
    """Tracks whether a monitored value has stopped improving, torch-`ReduceLROnPlateau`-style.

    Attributes:
        mode: `"min"` (lower is better, e.g. a loss) or `"max"` (higher is
            better, e.g. an accuracy or a custom metric).
        threshold: How much better a new value must be than the current
            best to count as an improvement (interpreted per
            `threshold_mode`). Matches
            `torch.optim.lr_scheduler.ReduceLROnPlateau`'s own default of
            `1e-4`.
        threshold_mode: `"rel"` (the improvement must be at least
            `threshold` *relative* to the current best: `best * (1 -
            threshold)` for `mode="min"`, `best * (1 + threshold)` for
            `mode="max"`) or `"abs"` (an absolute margin: `best -
            threshold` / `best + threshold`).
        patience: Number of consecutive non-improving epochs allowed
            before `step` reports a plateau. `patience=0` means "plateau
            on the very first non-improving epoch".
        cooldown: Number of epochs to wait, after a plateau was reported,
            before bad epochs are counted again (mirrors
            `ReduceLROnPlateau`'s own `cooldown`: gives whatever action the
            caller took in response to the plateau time to actually affect
            the monitored metric before patience starts accumulating
            again).
        best: The best value observed so far (`+inf` for `mode="min"`,
            `-inf` for `mode="max"`, until the first `step` call).
        num_bad_epochs: Consecutive non-improving epochs observed since
            the last improvement (reset by cooldown too, exactly like
            `ReduceLROnPlateau`).
        cooldown_counter: Epochs left in the current cooldown window.

    Raises:
        ValueError: If `mode` is not `"min"`/`"max"`, if `threshold_mode`
            is not `"rel"`/`"abs"`, if `patience` is negative, or if
            `cooldown` is negative.
    """

    mode: str = "min"
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    patience: int = 10
    cooldown: int = 0
    best: float = field(init=False)
    num_bad_epochs: int = field(default=0, init=False)
    cooldown_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{self.mode}'.")
        if self.threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got '{self.threshold_mode}'.")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}.")
        if self.cooldown < 0:
            raise ValueError(f"cooldown must be non-negative, got {self.cooldown}.")
        self.best = math.inf if self.mode == "min" else -math.inf

    @property
    def in_cooldown(self) -> bool:
        """Whether a cooldown window is currently active.

        Returns:
            `True` if `cooldown_counter > 0`.
        """
        return self.cooldown_counter > 0

    def _isBetter(self, value: float) -> bool:
        """Whether `value` counts as an improvement over `self.best`.

        Args:
            value: The candidate value.

        Returns:
            `True` if `value` improves on `self.best` by at least
            `self.threshold`, interpreted per `self.threshold_mode`.
        """
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return value < self.best * (1.0 - self.threshold)
            return value < self.best - self.threshold
        if self.threshold_mode == "rel":
            return value > self.best * (1.0 + self.threshold)
        return value > self.best + self.threshold

    def step(self, value: float) -> bool:
        """Update the tracker with this epoch's value.

        Args:
            value: This epoch's monitored value.

        Returns:
            `True` exactly on the epoch `num_bad_epochs` first exceeds
            `patience` (i.e. once per plateau, not on every subsequent
            non-improving epoch while already past patience): the caller
            reacts once (lower the learning rate, stop training) and
            `resetAfterAction` (called automatically here) starts a fresh
            cooldown/patience window for the next plateau, mirroring
            `ReduceLROnPlateau`'s own `_reduce_lr`-then-reset behavior.
        """
        if self._isBetter(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # bad epochs during cooldown do not count

        plateaued = self.num_bad_epochs > self.patience
        if plateaued:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        return plateaued
