"""reconstruction subpackage of global_vae.losses (spec §2.2, §11, §12, roadmap P5-3).

`aggregate` holds `computeTotalReconstructionLoss`, the per-modality aggregation of ADR 0029 and
ADR 0030, and the reduction helpers. `base` and `registry` hold the pluggable loss
(`AbstractReconstructionLoss`, `registerReconstructionLoss`, `getReconstructionLossClass`,
`listRegisteredReconstructionLosses`, ADR 0048). Importing this package registers every built-in
loss through its `@registerReconstructionLoss` decorator, and re-exports what
`global_vae.losses.reconstruction` was a module of, so no import of the old module changes.
"""

import global_vae.losses.reconstruction.bce  # noqa: F401
import global_vae.losses.reconstruction.bce_with_logits  # noqa: F401
import global_vae.losses.reconstruction.chamfer  # noqa: F401
import global_vae.losses.reconstruction.l1  # noqa: F401
import global_vae.losses.reconstruction.masked_mse  # noqa: F401
import global_vae.losses.reconstruction.mse  # noqa: F401
import global_vae.losses.reconstruction.smooth_l1  # noqa: F401
import global_vae.losses.reconstruction.tabular_mixed  # noqa: F401
import global_vae.losses.reconstruction.token_cross_entropy  # noqa: F401
from global_vae.losses.reconstruction.aggregate import (
    DEFAULT_REDUCTION,
    SUPPORTED_REDUCTIONS,
    LossFn,
    collectReconstructionLossModules,
    computeTotalReconstructionLoss,
    listSupportedReductions,
    resolveReduction,
    validateReduction,
)
from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import (
    getReconstructionLossClass,
    listRegisteredReconstructionLosses,
    registerReconstructionLoss,
)

__all__ = [
    "DEFAULT_REDUCTION",
    "SUPPORTED_REDUCTIONS",
    "AbstractReconstructionLoss",
    "LossFn",
    "collectReconstructionLossModules",
    "computeTotalReconstructionLoss",
    "getReconstructionLossClass",
    "listRegisteredReconstructionLosses",
    "listSupportedReductions",
    "registerReconstructionLoss",
    "resolveReduction",
    "validateReduction",
]
