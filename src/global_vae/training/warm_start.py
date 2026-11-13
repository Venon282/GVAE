"""Warm start: copy trained modules from solo checkpoints into a pair model (P6-2, ADR 0052).

Stage 3 of a staged recipe starts where stages 1 and 2 ended. The pair model has the encoder and the
decoder of each modality under the names the solo models give them, and two latent spaces that carry
no weights, so what moves is the weights of those modules. `applyWarmStart` copies them, and refuses
a file that is not the one that was meant before a weight is touched:

1. the manifest of the checkpoint (ADR 0051) must describe a solo model of the modality asked for,
   with a latent space as wide as the pair model's, and the config of each copied module must be
   the one the checkpoint was trained with (`StageManifest.module_digests`);
2. the keys the checkpoint holds under each module must be exactly the keys the pair model's module
   has, with the same shapes (parameters and buffers, running statistics included);
3. after the copy, the encoder must not have collapsed: probed on the pair's training data with the
   posterior mean, it must have active latent units (`min_active_units`, default 1, a total
   collapse).

Only the model weights of a checkpoint are used (`readCheckpointWeights`, safe mode). The optimizer
state, the counters, the history and the random-number state are not: stage 3 is a new run with its
own optimizer, its own seed and a step counter at 0. A call that fails leaves the model as it found
it.

This module holds the mechanism and knows nothing of Hydra: `global_vae.config.training` owns the
`training.warm_start` field and builds the arguments from it.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from global_vae.evaluation.latent_diagnostics import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    computeLatentDiagnostics,
)
from global_vae.training.checkpoint import CheckpointWeights, readCheckpointWeights
from global_vae.training.stage_manifest import StageManifest

if TYPE_CHECKING:
    from global_vae.models.global_vae import GlobalVae

logger = logging.getLogger(__name__)

# The collapse guard encodes at most this many samples of the training loader, once.
PROBE_MAX_SAMPLES = 256

_MODULE_KINDS = ("encoders", "decoders")


@dataclass(frozen=True)
class WarmStartEntry:
    """One solo checkpoint to copy modules from.

    Attributes:
        checkpoint: The solo checkpoint file.
        modules: What to copy. A modality name (`"signal"`) stands for `encoders.signal` and
            `decoders.signal`; `"encoders.signal"` or `"decoders.signal"` names one module. A
            module has the same name in the checkpoint's model and in the pair model, and all
            the modules of one entry belong to one modality (a solo checkpoint covers one). Any
            sequence of names; it is stored as a tuple.
        latent_space: The pair model's latent space that the solo latent space becomes. `None`
            (default) takes the one space the copied encoder feeds (the one the copied decoder
            reads, when only a decoder is copied), and is an error when there are several.
        check_manifest: `True` (default) requires the checkpoint's stage manifest and checks it
            (solo model, modality, latent width, module config). `False` does not read it, and
            leaves the shape checks and the collapse guard: for a file written before the
            manifest existed, or by a run that set no `training.stage`.
        min_active_units: The fewest active latent units the copied encoder may have over the
            probe data. `1` (default) refuses a total collapse; `0` switches the guard off.

    Raises:
        ValueError: If `checkpoint` is empty, `modules` is empty or holds an empty name,
            `latent_space` is an empty string, or `min_active_units` is negative.
    """

    checkpoint: str | Path
    modules: Sequence[str]
    latent_space: str | None = None
    check_manifest: bool = True
    min_active_units: int = 1

    def __post_init__(self) -> None:
        """Refuse an entry that cannot name its checkpoint, its modules or a valid bar."""
        if not str(self.checkpoint).strip():
            raise ValueError("checkpoint must be a non-empty path.")
        object.__setattr__(self, "modules", tuple(self.modules))
        if not self.modules or not all(
            isinstance(name, str) and name.strip() for name in self.modules
        ):
            raise ValueError(
                f"modules must be a non-empty list of non-empty names, got {list(self.modules)!r}."
            )
        if self.latent_space is not None and not self.latent_space.strip():
            raise ValueError("latent_space must be a non-empty name or None.")
        if (
            isinstance(self.min_active_units, bool)
            or not isinstance(self.min_active_units, int)
            or self.min_active_units < 0
        ):
            raise ValueError(
                f"min_active_units must be an integer of at least 0, got {self.min_active_units!r}."
            )


@dataclass
class WarmStartEntryReport:
    """What one entry did.

    Attributes:
        checkpoint: The file that was read.
        stage: The stage label of its manifest, or `None` (no manifest, or no label).
        modules: The module paths that were copied (`"encoders.signal"`, ...).
        num_tensors: How many tensors (parameters and buffers) were copied.
        latent_space: The pair model's latent space the entry maps to, or `None` when it was
            not needed (the guard is off and the manifest was not checked).
        active_units: Active units of `latent_space` over the probe data, or `None` when the
            guard did not run (switched off, no encoder copied, or no data passed).
        num_dimensions: Width of `latent_space`, set when `active_units` is.
        num_probe_samples: How many samples were encoded, set when `active_units` is.
    """

    checkpoint: str
    stage: str | None
    modules: list[str]
    num_tensors: int
    latent_space: str | None = None
    active_units: int | None = None
    num_dimensions: int | None = None
    num_probe_samples: int | None = None


@dataclass
class WarmStartReport:
    """What `applyWarmStart` did, one `WarmStartEntryReport` per entry, in order."""

    entries: list[WarmStartEntryReport] = field(default_factory=list)


@dataclass
class _Planned:
    """An entry that passed every check that needs no copy, ready to be applied."""

    label: str
    entry: WarmStartEntry
    paths: list[str]
    modality: str
    latent_space: str | None
    weights: CheckpointWeights
    states: dict[str, dict[str, torch.Tensor]]


def _label(index: int, entry: WarmStartEntry) -> str:
    return f"training.warm_start[{index}] ('{entry.checkpoint}')"


def _expandModules(label: str, entry: WarmStartEntry, model: "GlobalVae") -> tuple[str, list[str]]:
    """Turn the names of an entry into module paths and find its modality.

    Args:
        label: The entry, for error messages.
        entry: The entry.
        model: The pair model, which must have every module.

    Returns:
        `(modality, paths)`: the one modality all the modules belong to, and the module paths in
        the order they are written, `encoders` before `decoders` for a bare modality name.

    Raises:
        ValueError: If a name is not a modality or an `encoders.` / `decoders.` path, the model
            lacks a module, the modules belong to several modalities, or a module is named twice.
    """
    paths: list[str] = []
    for name in entry.modules:
        if "." in name:
            kind, _, module_name = name.partition(".")
            if kind not in _MODULE_KINDS or not module_name:
                raise ValueError(
                    f"{label}: module '{name}' is neither a modality name nor a path of the form "
                    f"'encoders.<name>' or 'decoders.<name>'. Only encoders and decoders carry "
                    f"weights to copy; the latent spaces have none."
                )
            paths.append(name)
        else:
            paths.extend(f"{kind}.{name}" for kind in _MODULE_KINDS)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label}: modules {list(entry.modules)!r} name a module twice.")

    modalities = {path.partition(".")[2] for path in paths}
    if len(modalities) != 1:
        raise ValueError(
            f"{label}: modules {list(entry.modules)!r} belong to several modalities "
            f"({sorted(modalities)}). A solo checkpoint covers one modality: use one entry per "
            f"checkpoint."
        )

    for path in paths:
        kind, _, module_name = path.partition(".")
        container: nn.ModuleDict = getattr(model, kind)
        if module_name not in container:
            raise ValueError(
                f"{label}: the pair model has no module '{path}'. Its {kind}: {sorted(container)}. "
                f"The modules of the pair model must carry the names the solo model gives them."
            )
    return modalities.pop(), paths


def _resolveLatentSpace(
    label: str, entry: WarmStartEntry, model: "GlobalVae", modality: str, paths: list[str]
) -> str:
    """Find the pair model's latent space that the solo latent space becomes.

    Args:
        label: The entry, for error messages.
        entry: The entry; its `latent_space` wins when given.
        model: The pair model.
        modality: The modality of the entry.
        paths: The module paths of the entry.

    Returns:
        The latent space name.

    Raises:
        ValueError: If `entry.latent_space` is not a space of the model, or none is given and the
            copied encoder (or decoder, when no encoder is copied) does not touch exactly one.
    """
    spaces = model.latent_spaces
    if entry.latent_space is not None:
        if entry.latent_space not in spaces:
            raise ValueError(
                f"{label}: latent_space '{entry.latent_space}' is not a latent space of the pair "
                f"model. Its spaces: {sorted(spaces)}."
            )
        return entry.latent_space

    graph = model.routing_graph
    if f"encoders.{modality}" in paths:
        candidates = list(graph.encoder_to_latents.get(modality, []))
        role = f"encoder '{modality}' feeds"
    else:
        candidates = [
            name for name, readers in graph.latent_to_decoders.items() if modality in readers
        ]
        role = f"decoder '{modality}' reads"
    if len(candidates) != 1:
        raise ValueError(
            f"{label}: the {role} {len(candidates)} latent space(s) ({sorted(candidates)}), so "
            f"the space the solo latent becomes is not determined. Name it with latent_space."
        )
    return candidates[0]


def _checkManifest(
    label: str,
    entry: WarmStartEntry,
    weights: CheckpointWeights,
    model: "GlobalVae",
    modality: str,
    paths: list[str],
    latent_space: str,
    module_digests: Mapping[str, str] | None,
) -> StageManifest:
    """Check the checkpoint's stage manifest against the entry and the pair model.

    Args:
        label: The entry, for error messages.
        entry: The entry.
        weights: What the checkpoint holds.
        model: The pair model.
        modality: The modality of the entry.
        paths: The module paths of the entry.
        latent_space: The pair model's space the solo latent becomes.
        module_digests: Module path to the digest of the pair model's own resolved config, or
            `None` to leave the module configs unchecked (no model config at hand).

    Returns:
        The manifest, once it has passed.

    Raises:
        ValueError: If there is no manifest, it does not describe a solo model, its modality is
            not the entry's, its latent width is not the pair model's, or a module's config digest
            is missing or different.
    """
    manifest = weights.manifest
    stage_text = "" if manifest is None else f" (stage {manifest.stage!r})"
    if manifest is None:
        raise ValueError(
            f"{label}: the checkpoint has no stage manifest, so it cannot be checked against the "
            f"pair model. Train the solo stage again with training.stage set "
            f"(docs/how-to/train-in-stages.md), or set check_manifest: false on this entry to "
            f"rely on the shape checks alone."
        )
    if len(manifest.modalities) != 1 or len(manifest.latent_spaces) != 1:
        raise ValueError(
            f"{label}: the checkpoint{stage_text} is not a solo model: it has modalities "
            f"{list(manifest.modalities)} and latent spaces {sorted(manifest.latent_spaces)}. A "
            f"warm start copies from checkpoints with one modality and one latent space."
        )
    if modality not in manifest.modalities:
        raise ValueError(
            f"{label}: the checkpoint{stage_text} is the solo model of "
            f"{list(manifest.modalities)}, but the entry asks for {list(entry.modules)!r} "
            f"(modality '{modality}'). Is this the wrong file?"
        )
    ((solo_space, solo_width),) = manifest.latent_spaces.items()
    pair_width = model.latent_spaces[latent_space].dim
    if solo_width != pair_width:
        raise ValueError(
            f"{label}: the solo latent space '{solo_space}' of the checkpoint{stage_text} is "
            f"{solo_width} wide, but the pair model's '{latent_space}' is {pair_width} wide. The "
            f"two spaces must have equal widths (roadmap D-16)."
        )
    if module_digests is None:
        logger.warning(
            "%s: the config of the copied modules was not checked, no model config was given.",
            label,
        )
        return manifest
    for path in paths:
        recorded = manifest.module_digests.get(path)
        if recorded is None:
            raise ValueError(
                f"{label}: the checkpoint{stage_text} records no config digest for '{path}' (it "
                f"was written before module digests existed, or by a run that was given no model "
                f"config). Train the solo stage again, or set check_manifest: false."
            )
        expected = module_digests.get(path)
        if expected is None:
            raise ValueError(
                f"{label}: no config digest was given for the pair model's '{path}', so the "
                f"checkpoint cannot be checked against it. The digests must be computed from the "
                f"config that built the model."
            )
        if recorded != expected:
            raise ValueError(
                f"{label}: the config of '{path}' in the pair model (digest {expected[:12]}) is "
                f"not the one the checkpoint{stage_text} was trained with (digest "
                f"{recorded[:12]}). The pair model file must give this module the registry name "
                f"and the kwargs of the solo model file, and the same latent width."
            )
    return manifest


def _planCopy(
    label: str, model: "GlobalVae", paths: list[str], state_dict: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, torch.Tensor]]:
    """Check that each module's keys and shapes agree, and cut out its part of the state dict.

    Args:
        label: The entry, for error messages.
        model: The pair model.
        paths: The module paths to copy.
        state_dict: The checkpoint's model state dict.

    Returns:
        Module path to the module's own state dict (keys relative to the module), ready for
        `load_state_dict`.

    Raises:
        ValueError: If the checkpoint holds nothing under a module, misses a key of it, holds a
            key the module does not have, or a shape differs. The message names the key.
    """
    states: dict[str, dict[str, torch.Tensor]] = {}
    for path in paths:
        module = model.get_submodule(path)
        prefix = f"{path}."
        saved = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if not saved:
            held = sorted({".".join(key.split(".")[:2]) for key in state_dict})
            raise ValueError(
                f"{label}: the checkpoint holds no tensor of module '{path}'. It holds: {held}. "
                f"Is this the wrong file, or a model that names its modules differently?"
            )
        wanted = module.state_dict()
        missing = sorted(set(wanted) - set(saved))
        unexpected = sorted(set(saved) - set(wanted))
        if missing:
            raise ValueError(
                f"{label}: the checkpoint has no tensor for {len(missing)} key(s) of '{path}', "
                f"the first being '{prefix}{missing[0]}'. The solo model and the pair model "
                f"must build this module the same way."
            )
        if unexpected:
            raise ValueError(
                f"{label}: the checkpoint holds {len(unexpected)} key(s) that the pair model's "
                f"'{path}' does not have, the first being '{prefix}{unexpected[0]}'. The solo "
                f"model and the pair model must build this module the same way."
            )
        for key, value in saved.items():
            if tuple(value.shape) != tuple(wanted[key].shape):
                raise ValueError(
                    f"{label}: shape mismatch for '{prefix}{key}': the checkpoint has "
                    f"{tuple(value.shape)} and the pair model has {tuple(wanted[key].shape)}."
                )
        states[path] = saved
    return states


def _probeMeans(
    model: "GlobalVae",
    spaces: Iterable[str],
    dataloader: Iterable[Any],
    max_samples: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode training samples with the posterior mean and collect `(mu, logvar)` per space.

    The model is put in eval mode and back, nothing is differentiated, and the random-number
    state of the CPU (and of the model's CUDA device) is restored, so the probe changes no weight
    and does not shift the stream training will draw from.

    Args:
        model: The pair model, already holding the copied weights.
        spaces: The latent spaces to collect.
        dataloader: An iterable of dict batches, the pair's training loader.
        max_samples: Stop after the batch that brings the count to at least this many.

    Returns:
        Space name -> `(mu, logvar)`, each `(N, dim)` on the CPU, `N <= max_samples`.

    Raises:
        ValueError: If a space receives no encoding (the batches lack the modality that feeds
            it), or fewer than 2 samples were seen (a variance needs two).
    """
    wanted = list(spaces)
    collected: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {name: [] for name in wanted}
    device = next(model.parameters()).device
    devices = [device] if device.type == "cuda" else []
    was_training = model.training
    seen = 0
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            for batch in dataloader:
                inputs = {
                    name: tensor.to(device)
                    for name, tensor in model.selectEncoderInputs(batch).items()
                }
                params = model.encode(inputs, use_mean=True)["latent_params"]
                batch_size = 0
                for name in wanted:
                    if name not in params:
                        raise ValueError(
                            f"Warm start: the probe batches hold no input for the encoder of "
                            f"latent space '{name}': the batch keys are {sorted(batch)}."
                        )
                    mu, logvar = params[name]
                    collected[name].append((mu.float().cpu(), logvar.float().cpu()))
                    batch_size = mu.shape[0]
                seen += batch_size
                if seen >= max_samples:
                    break
    finally:
        model.train(was_training)
    if seen < 2:
        raise ValueError(
            f"Warm start: the collapse guard needs at least 2 training samples to measure a "
            f"variance, and the loader gave {seen}. Set min_active_units: 0 to skip the guard."
        )
    return {
        name: (
            torch.cat([mu for mu, _ in parts])[:max_samples],
            torch.cat([logvar for _, logvar in parts])[:max_samples],
        )
        for name, parts in collected.items()
    }


def applyWarmStart(
    model: "GlobalVae",
    entries: Sequence[WarmStartEntry],
    module_digests: Mapping[str, str] | None = None,
    probe_dataloader: Iterable[Any] | None = None,
    max_probe_samples: int = PROBE_MAX_SAMPLES,
) -> WarmStartReport:
    """Copy the modules named by `entries` from solo checkpoints into `model`.

    Every entry is checked first (file, manifest, keys and shapes), nothing is copied until all
    pass, and a failure after the copy (the collapse guard) puts the copied modules back, so a
    failed call leaves `model` as it was.

    Args:
        model: The pair model, built from its own config. Its modules are overwritten in place,
            so an optimizer built over its parameters afterwards, or before, sees the new values.
        entries: What to copy. Empty returns an empty report and reads nothing.
        module_digests: Module path to the digest of the pair model's own resolved config
            (`global_vae.config.training.computeModuleDigests`), which a checkpoint's manifest is
            compared against. `None` (default) leaves the module configs unchecked, with a
            warning, for a caller that has no model config.
        probe_dataloader: The pair's training loader, an iterable of dict batches, for the
            collapse guard. `None` (default) runs no guard, and the report says so.
        max_probe_samples: At most this many samples are encoded for the guard, once.

    Returns:
        A `WarmStartReport`.

    Raises:
        FileNotFoundError: If a checkpoint file does not exist.
        pickle.UnpicklingError: If the safe mode refuses a file (see `readCheckpointWeights`).
        ValueError: For every refusal described in the module docstring, each naming the entry
            (`training.warm_start[i]`) and its file: a wrong modality, a manifest that is missing
            or of another kind of model, a latent width or module config that differs, a missing,
            unexpected or mis-shaped key, a module loaded twice, and a collapsed encoder.
    """
    report = WarmStartReport()
    if not entries:
        return report

    planned: list[_Planned] = []
    taken: dict[str, str] = {}
    for index, entry in enumerate(entries):
        label = _label(index, entry)
        modality, paths = _expandModules(label, entry, model)
        for path in paths:
            if path in taken:
                raise ValueError(f"{label}: module '{path}' is also loaded by {taken[path]}.")
            taken[path] = label
        needs_space = entry.check_manifest or (
            entry.min_active_units > 0 and f"encoders.{modality}" in paths
        )
        latent_space = (
            _resolveLatentSpace(label, entry, model, modality, paths) if needs_space else None
        )
        weights = readCheckpointWeights(entry.checkpoint)
        if entry.check_manifest:
            assert latent_space is not None
            _checkManifest(
                label, entry, weights, model, modality, paths, latent_space, module_digests
            )
        states = _planCopy(label, model, paths, weights.state_dict)
        planned.append(_Planned(label, entry, paths, modality, latent_space, weights, states))

    snapshot = {
        path: {
            key: value.detach().clone()
            for key, value in model.get_submodule(path).state_dict().items()
        }
        for item in planned
        for path in item.paths
    }
    try:
        for item in planned:
            for path, state in item.states.items():
                model.get_submodule(path).load_state_dict(state, strict=True)
            report.entries.append(
                WarmStartEntryReport(
                    checkpoint=str(item.entry.checkpoint),
                    stage=None if item.weights.manifest is None else item.weights.manifest.stage,
                    modules=list(item.paths),
                    num_tensors=sum(len(state) for state in item.states.values()),
                    latent_space=item.latent_space,
                )
            )
        _runCollapseGuard(model, planned, report, probe_dataloader, max_probe_samples)
    except BaseException:
        for path, state in snapshot.items():
            model.get_submodule(path).load_state_dict(state, strict=True)
        raise

    for entry_report in report.entries:
        logger.info(
            "Warm start: copied %s (%d tensors) from '%s' (stage %r)%s.",
            ", ".join(entry_report.modules),
            entry_report.num_tensors,
            entry_report.checkpoint,
            entry_report.stage,
            ""
            if entry_report.active_units is None
            else f"; '{entry_report.latent_space}': {entry_report.active_units} of "
            f"{entry_report.num_dimensions} latent units active over "
            f"{entry_report.num_probe_samples} samples",
        )
    return report


def _runCollapseGuard(
    model: "GlobalVae",
    planned: list[_Planned],
    report: WarmStartReport,
    probe_dataloader: Iterable[Any] | None,
    max_probe_samples: int,
) -> None:
    """Refuse a copied encoder that has fewer active latent units than its entry requires.

    Args:
        model: The pair model, holding the copied weights.
        planned: The entries that were applied, in the order of `report.entries`.
        report: Filled in place with the active units of each probed entry.
        probe_dataloader: The training loader, or `None` to run nothing.
        max_probe_samples: The most samples to encode.

    Raises:
        ValueError: If an entry's encoder has fewer active units than `min_active_units`.
    """
    guarded = [
        (item, entry_report)
        for item, entry_report in zip(planned, report.entries, strict=True)
        if item.entry.min_active_units > 0 and f"encoders.{item.modality}" in item.paths
    ]
    if not guarded:
        return
    if probe_dataloader is None:
        logger.warning(
            "Warm start: the collapse guard did not run, no data was given to probe the encoders."
        )
        return
    spaces = {item.latent_space for item, _ in guarded if item.latent_space is not None}
    means = _probeMeans(model, sorted(spaces), probe_dataloader, max_probe_samples)
    for item, entry_report in guarded:
        assert item.latent_space is not None
        mu, logvar = means[item.latent_space]
        diagnostics = computeLatentDiagnostics(mu, logvar)
        entry_report.active_units = diagnostics.active_units
        entry_report.num_dimensions = diagnostics.num_dimensions
        entry_report.num_probe_samples = int(mu.shape[0])
        if diagnostics.active_units < item.entry.min_active_units:
            raise ValueError(
                f"{item.label}: the solo encoder of '{item.modality}' looks collapsed. Over "
                f"{mu.shape[0]} training samples, {diagnostics.active_units} of "
                f"{diagnostics.num_dimensions} dimensions of latent space '{item.latent_space}' "
                f"have a posterior-mean variance above {DEFAULT_ACTIVE_UNIT_THRESHOLD} (the entry "
                f"requires at least {item.entry.min_active_units}). The model encodes every "
                f"sample to nearly the same code, so aligning it with another modality would "
                f"align nothing. The usual cause is a beta too large for the data "
                f"(docs/how-to/choose-beta.md): train the solo stage again. To accept this file "
                f"anyway, set min_active_units: 0."
            )
