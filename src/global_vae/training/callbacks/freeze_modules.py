"""Keep chosen modules of the model fixed for a while, or for the whole run (roadmap P6-5).

The pair stage can undo what the solo stages learned, especially with few pairs. This callback is
the gentle way out: the modules it names stay exactly as they are, and the rest of the model trains
around them. It is also the mechanism of a one-sided alignment (one modality is an anchor and the
other moves toward it) and of a frozen reference (`docs/roadmap.md` P7-5).

A frozen module is three things at once, and the callback does all three:

- its parameters do not train: `requires_grad` is switched off, and any gradient they hold is
  dropped, so that no optimizer (weight decay included) moves them;
- it is in evaluation mode, which is the part that is easy to forget. The CNN blocks of this
  framework use `BatchNorm` by default (`nn.BatchNorm1d` and `nn.BatchNorm2d` in
  `utils/conv_blocks.py`), and in training mode a `BatchNorm` layer moves its running mean and
  variance at every forward pass, parameters frozen or not: with the small batches that few pairs
  give, a "frozen" module would drift anyway. The same goes for dropout layers. The model is put
  in training mode at the start of every epoch (`Trainer.fitEpoch`), so the callback puts the
  frozen modules back to evaluation mode right after, in `onEpochBegin`;
- it stays in the optimizer. Nothing is removed from or added to the optimizer's groups, so the
  checkpoint has the same layout whether a module is frozen or not, and a run can be resumed across
  the epoch at which a module is released.

`modules` are name patterns (`training/module_patterns.py`): `encoders.signal`, `decoders.*`, `*`.
`until_epoch` is the first epoch in which the modules train: they are frozen in epochs
`0 .. until_epoch - 1` and released at the start of epoch `until_epoch`. `None` (default) freezes
them for the whole run. Whether an epoch is a frozen one is a function of the epoch number alone,
and `Trainer` hands the callback the real number after a resume, so there is no state to save: a
resumed run is frozen, or released, exactly as the uninterrupted one would be.

A parameter that was not trainable before the run (the person's own `requires_grad_(False)`) is
left alone: the callback does not freeze it a second time, and does not make it trainable when it
releases the others. When the run ends the callback puts every `requires_grad` back as it found it.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from torch import nn

from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback
from global_vae.training.module_patterns import checkPatterns, matchModules, parametersOf

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("freeze_modules")
class FreezeModules(TrainerCallback):
    """Freeze the modules whose names match `modules`, until `until_epoch` or for the whole run.

    Attributes:
        modules: The name patterns, as a tuple.
        until_epoch: The first epoch in which the modules train, or `None` for never.
    """

    def __init__(self, modules: Sequence[str], until_epoch: int | None = None) -> None:
        """Initialize the callback.

        Args:
            modules: Module name patterns (`encoders.signal`, `decoders.*`; `*` stands for any
                characters, dots included). A module stands for its parameters and those of every
                module inside it. Checked against the model when the run starts, and a pattern
                that matches no module is an error.
            until_epoch: The epoch (0-based) at which the modules are released: frozen while the
                epoch is below it, training from it on. At least 1 (an `until_epoch` of 0 would
                freeze nothing; leave the callback out for that). `None` (default) never
                releases them.

        Raises:
            ValueError: If `modules` is a string, is empty or holds anything but non-empty
                strings, or if `until_epoch` is not `None` or an integer of at least 1.
        """
        self.modules = checkPatterns(modules, "freeze_modules: `modules`")
        if until_epoch is not None and (
            isinstance(until_epoch, bool) or not isinstance(until_epoch, int) or until_epoch < 1
        ):
            raise ValueError(
                f"freeze_modules: `until_epoch` must be None (never release) or an integer of "
                f"at least 1, got {until_epoch!r}. To freeze nothing, leave the callback out."
            )
        self.until_epoch = until_epoch
        self._modules: list[nn.Module] = []
        self._parameters: list[nn.Parameter] = []
        self._was_trainable: dict[int, bool] = {}
        self._frozen: bool | None = None

    def isFrozenAt(self, epoch: int) -> bool:
        """Return whether the modules are frozen during epoch `epoch`.

        Args:
            epoch: A 0-based epoch number.

        Returns:
            `True` if `until_epoch` is `None` or `epoch` is below it.
        """
        return self.until_epoch is None or epoch < self.until_epoch

    def onTrainBegin(self, trainer: "Trainer") -> None:
        """Find the modules and note which of their parameters train now.

        Args:
            trainer: The `Trainer` instance running this training run.

        Raises:
            ValueError: If a pattern matches no module of `trainer.model`, or if the modules
                matched have no parameter (there would be nothing to freeze).
        """
        matched = matchModules(trainer.model, self.modules, "freeze_modules")
        self._modules = list(matched.values())
        self._parameters = parametersOf(self._modules)
        if not self._parameters:
            raise ValueError(
                f"freeze_modules: the modules matched by {list(self.modules)} "
                f"({sorted(matched)[:5]}) have no parameter, so there is nothing to freeze."
            )
        self._was_trainable = {id(p): p.requires_grad for p in self._parameters}
        self._frozen = None

    def onEpochBegin(self, trainer: "Trainer", epoch: int) -> None:
        """Freeze or release the modules for `epoch`, after the model was put in training mode.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch about to start (0-based); after a resume, the real one.

        Raises:
            ValueError: If freezing leaves no parameter for the optimizer to train in this epoch
                (the backward pass would fail with a message that does not say why).
        """
        frozen = self.isFrozenAt(epoch)
        for parameter in self._parameters:
            if self._was_trainable[id(parameter)]:
                parameter.requires_grad_(not frozen)
                if frozen:
                    parameter.grad = None
        if frozen:
            for module in self._modules:
                module.eval()
            self._requireSomethingToTrain(trainer, epoch)
        if frozen != self._frozen:
            logger.info(
                "freeze_modules: epoch %d, %s %d parameter(s) of %d module(s) matching %s%s.",
                epoch,
                "frozen:" if frozen else "released:",
                len(self._parameters),
                len(self._modules),
                list(self.modules),
                "" if self.until_epoch is None else f" (until epoch {self.until_epoch})",
            )
            self._frozen = frozen

    def onTrainEnd(self, trainer: "Trainer") -> None:
        """Put every `requires_grad` back as it was when the run began.

        Args:
            trainer: The `Trainer` instance running this training run (unused).
        """
        for parameter in self._parameters:
            parameter.requires_grad_(self._was_trainable[id(parameter)])
        self._frozen = None

    def _requireSomethingToTrain(self, trainer: "Trainer", epoch: int) -> None:
        """Raise if no parameter the optimizer holds can train in this epoch.

        Args:
            trainer: The trainer whose optimizer is read.
            epoch: The running epoch, for the message.
        """
        trainable = any(
            parameter.requires_grad
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        )
        if not trainable:
            raise ValueError(
                f"freeze_modules: in epoch {epoch}, every parameter that the optimizer holds is "
                f"frozen (modules {list(self.modules)}), so there is nothing to train. Freeze "
                f"fewer modules, or set `until_epoch`. To keep both VAEs frozen and train only an "
                f"alignment head, give `latent_alignment` `adapters`: the adapter is a parameter "
                f"of the term, not of the model, so it keeps training (ADR 0060)."
            )
