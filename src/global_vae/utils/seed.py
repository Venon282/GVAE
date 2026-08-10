"""Global random-seed management (spec §10: "global seed management, deterministic-mode
flag documented").

This must run before model construction (an encoder/decoder's weights
are initialized randomly the moment `nn.Linear(...)` etc. is called),
so it is deliberately a standalone function a training script calls
first, not something `GlobalVae` or `Trainer` manage themselves.

Seeds every RNG this codebase's randomness can come from: Python's
`random` (used nowhere in this package directly today, but a caller's
own data pipeline may use it), NumPy (same reasoning; optional, only
seeded if installed), and PyTorch's CPU and CUDA generators (weight
initialization, `torch.randn_like` in `LatentSpace.reparameterize`,
`MmdRegularizer`, `Trainer`'s modality dropout, dropout layers if a
future encoder/decoder uses them, ...).
"""

import logging
import os
import random

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def setGlobalSeed(seed: int, deterministic: bool = False, warn_only: bool = False) -> None:
    """Seed every RNG this codebase's randomness can come from, and set the determinism mode.

    Args:
        seed: The seed value, applied identically to `random`, NumPy
            (if installed), and PyTorch's CPU and CUDA generators (all
            devices).
        deterministic: If `False` (default), only the seed above is
            set; PyTorch's cuDNN backend is left free to auto-tune
            (`torch.backends.cudnn.benchmark = True`), which is faster
            but means the exact same seed can still produce slightly
            different results between two runs on GPU, since some
            cuDNN algorithms are not deterministic. Set to `True` for
            bit-for-bit reproducibility across runs (useful when
            debugging or comparing two configurations that should only
            differ in the one thing being changed): this calls
            `torch.use_deterministic_algorithms(True)`, disables cuDNN
            auto-tuning, and best-effort sets the `CUBLAS_WORKSPACE_CONFIG`
            environment variable PyTorch's own documentation requires
            for deterministic CUDA behavior on some operations. This
            trades speed for reproducibility: deterministic algorithms
            are often slower than their auto-tuned/non-deterministic
            counterparts, and are not available for every operation
            (see `warn_only`). Toggling back to `False` after a prior
            `True` call explicitly resets every flag this touches, so
            the setting never silently leaks across calls within the
            same process (e.g. in a notebook).
        warn_only: Only meaningful when `deterministic=True`. If an
            operation with no deterministic implementation is used
            while `torch.use_deterministic_algorithms(True)` is active,
            the default (`False`) raises `RuntimeError` at that
            operation, which is the strict, unambiguous choice: it
            surfaces the problem instead of silently producing a
            non-reproducible run. Set to `True` to instead only warn
            and continue, if hitting such an operation should not stop
            training outright.

    Note:
        Setting `CUBLAS_WORKSPACE_CONFIG` only reliably takes effect if
        no CUDA context has been created yet in this process, so call
        `setGlobalSeed` as early as possible in a script, before any
        CUDA tensor is created. It is set via `os.environ.setdefault`,
        not overwritten, so a value the caller already configured
        (e.g. a different valid workspace size) is respected rather
        than clobbered.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.info("Global seed set to %d (deterministic=%s).", seed, deterministic)
