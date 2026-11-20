"""Adapter heads for the alignment term: spaces that keep their own widths (P6-8, ADR 0060).

`latent_alignment` compares the posteriors of two latent spaces coordinate by coordinate, so the
two must have one width (roadmap D-16). Two spaces that have each been trained at the width that
suits their modality do not. An **adapter** is a latent head (`heads/`, a `linear` one by default)
that the alignment term, and only the alignment term, puts between a space and the distance: it
reads the space's posterior `(mu, logvar)` and returns one of another width. The spaces, their
encoders and their decoders keep the widths they have, and nothing else reads an adapter.

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]        # widths 16 and 8
        distance: wasserstein2_diag
        stop_gradient: target               # z_signal stays; the adapter and z_image move
        adapters:
          z_image: 16                       # linear, 8 -> 16; or {name: linear, out_dim: 16}
```

The adapter of a space is a parameter of the term: the trainer puts it in its optimizer, in its
checkpoint (`loss_terms_state_dict`, under `<term>.adapters.<space>`), and restores it when a run
resumes (ADR 0047). It is built when the term is checked against the model, because its input
width is the width of the space it reads, which is the model's to say.

This module is the part that is not a term: parsing what the YAML gives, building the heads,
applying one, and reading them back from a checkpoint for the evaluation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from global_vae.heads.base import AbstractLatentHead
from global_vae.heads.registry import getHeadClass
from global_vae.losses.distances.base import LatentParams
from global_vae.models.global_vae import GlobalVae

DEFAULT_ADAPTER_HEAD = "linear"
"""The head an adapter is built from when its entry gives only a width."""

_ENTRY_KEYS = ("name", "out_dim", "kwargs")


@dataclass(frozen=True)
class AdapterSpec:
    """What it takes to build the adapter of one latent space (but for the width it reads).

    Attributes:
        head: The registered latent head (`heads/registry.py`), `linear` by default.
        out_dim: The width of the adapted posterior, the one the distance compares at.
        kwargs: The head's constructor arguments other than `in_dim` and `out_dim`.
    """

    head: str
    out_dim: int
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def toText(self) -> str:
        """The form that `parseAdapterText` reads back: `out_dim`, or `head:out_dim`."""
        return (
            str(self.out_dim)
            if self.head == DEFAULT_ADAPTER_HEAD
            else f"{self.head}:{self.out_dim}"
        )


def _parseEntry(space: str, entry: Any, where: str) -> AdapterSpec:
    """Turn one value of `adapters` (a width, or `{name, out_dim, kwargs}`) into a spec."""
    if isinstance(entry, bool):
        raise ValueError(
            f"{where}: the adapter of '{space}' must be a width or a mapping, got {entry!r}."
        )
    if isinstance(entry, int):
        width, head, kwargs = entry, DEFAULT_ADAPTER_HEAD, {}
    elif isinstance(entry, Mapping):
        unknown = sorted(set(entry) - set(_ENTRY_KEYS))
        if unknown or "out_dim" not in entry:
            raise ValueError(
                f"{where}: the adapter of '{space}' takes the keys {list(_ENTRY_KEYS)} with "
                f"`out_dim` required, got {sorted(entry)}."
            )
        width, head = entry["out_dim"], entry.get("name", DEFAULT_ADAPTER_HEAD)
        kwargs = dict(entry.get("kwargs") or {})
        if not isinstance(head, str) or not head:
            raise ValueError(
                f"{where}: the adapter of '{space}' names its head with a non-empty string, "
                f"got {head!r}."
            )
    else:
        raise ValueError(
            f"{where}: the adapter of '{space}' must be a width or a mapping "
            f"{{name, out_dim, kwargs}}, got {entry!r}."
        )
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(
            f"{where}: the adapter of '{space}' needs a positive `out_dim`, got {width!r}."
        )
    return AdapterSpec(head=head, out_dim=width, kwargs=kwargs)


def parseAdapterSpecs(adapters: Any, where: str) -> dict[str, AdapterSpec]:
    """Check the `adapters` argument of a term and turn it into specs.

    Args:
        adapters: `None` (no adapter), or a mapping latent space name -> a width (a `linear`
            head) or `{name, out_dim, kwargs}`.
        where: Who is asking, for the error messages (the term's name).

    Returns:
        Space name -> `AdapterSpec`, in the order given. Empty for `None` or `{}`.

    Raises:
        ValueError: If `adapters` is not a mapping, a name is not a non-empty string without a
            dot (it becomes a module name), or an entry is malformed (the message names the space).
    """
    if adapters is None:
        return {}
    if not isinstance(adapters, Mapping):
        raise ValueError(
            f"{where}: `adapters` must be a mapping latent space name -> a width or "
            f"{{name, out_dim, kwargs}}, got {adapters!r}."
        )
    specs: dict[str, AdapterSpec] = {}
    for space, entry in adapters.items():
        if not isinstance(space, str) or not space or "." in space:
            raise ValueError(
                f"{where}: an adapter is named by a latent space, a non-empty string without a "
                f"dot, got {space!r}."
            )
        specs[space] = _parseEntry(space, entry, where)
    return specs


def parseAdapterText(text: str) -> tuple[str, AdapterSpec]:
    """Read one `--alignment-adapter` argument: `SPACE=WIDTH` or `SPACE=HEAD:WIDTH`.

    Args:
        text: For example `z_image=16` or `z_image=linear:16`.

    Returns:
        The space and its spec.

    Raises:
        ValueError: If the text is not of that form or the width is not a positive integer.
    """
    space, separator, rest = text.partition("=")
    head, _, width = rest.rpartition(":")
    if not separator or not space or not width.lstrip("-").isdigit():
        raise ValueError(
            f"An adapter is written SPACE=WIDTH or SPACE=HEAD:WIDTH (z_image=16, "
            f"z_image=linear:16), got '{text}'."
        )
    spec = _parseEntry(
        space, {"name": head or DEFAULT_ADAPTER_HEAD, "out_dim": int(width)}, "--alignment-adapter"
    )
    return space, spec


def buildAdapter(model: GlobalVae, space: str, spec: AdapterSpec, where: str) -> AbstractLatentHead:
    """Build the adapter of one latent space of `model`, on the device of the model.

    Args:
        model: The model whose space is adapted: its width is the head's `in_dim`.
        space: The latent space the adapter reads.
        spec: The head, the output width and the head's other arguments.
        where: Who is asking, for the error messages.

    Returns:
        The head, a module.

    Raises:
        ValueError: If `space` is not a latent space of `model` (the message lists them) or the
            head refuses its widths (the identity head does, when they differ).
        KeyError: If the head name is not registered (the message lists them).
    """
    if space not in model.latent_spaces:
        raise ValueError(
            f"{where}: the adapter names '{space}', which is not a latent space of this model "
            f"(latent spaces: {sorted(model.latent_spaces)})."
        )
    in_dim = model.latent_spaces[space].dim
    try:
        head = getHeadClass(spec.head)(in_dim=in_dim, out_dim=spec.out_dim, **dict(spec.kwargs))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{where}: the adapter of '{space}' ({spec.head}, {in_dim} -> {spec.out_dim}) could "
            f"not be built: {error}"
        ) from error
    return head.to(next(model.parameters()).device)


def buildAdapters(model: GlobalVae, specs: Mapping[str, AdapterSpec], where: str) -> nn.ModuleDict:
    """Build every adapter of a term for `model`.

    Args:
        model: The model the term runs on.
        specs: Space name -> spec.
        where: Who is asking, for the error messages.

    Returns:
        A `nn.ModuleDict` space name -> head.

    Raises:
        ValueError: As `buildAdapter`.
        KeyError: As `buildAdapter`.
    """
    return nn.ModuleDict(
        {space: buildAdapter(model, space, spec, where) for space, spec in specs.items()}
    )


def applyAdapter(adapter: nn.Module | None, posterior: LatentParams) -> LatentParams:
    """Adapt a posterior `(mu, logvar)`, or return it as it is when the space has no adapter.

    Args:
        adapter: A latent head, or `None`.
        posterior: `(mu, logvar)`, each `(batch, width)`.

    Returns:
        The adapted `(mu, logvar)`, each `(batch, out_dim)`.
    """
    if adapter is None:
        return posterior
    mu, logvar = adapter(*posterior)
    return mu, logvar


def loadAdapters(
    model: GlobalVae,
    specs: Mapping[str, AdapterSpec],
    state: Mapping[str, torch.Tensor],
    where: str,
) -> dict[str, AbstractLatentHead]:
    """Build the adapters of a term and fill them from the state a checkpoint holds.

    Args:
        model: The model the adapters read the spaces of.
        specs: Space name -> spec, as the training config gave them.
        state: The `adapters.<space>.<parameter>` tensors of the term (what `readLossTermState`
            returns for the term's name).
        where: Who is asking, for the error messages.

    Returns:
        Space name -> head, in evaluation mode.

    Raises:
        ValueError: If a space is not in the model, a head cannot be built, or the state does not
            fit the head (the message gives the parameters it lacks or does not expect, which is
            what a mismatched `--alignment-adapter` width looks like).
        KeyError: If a head name is not registered.
    """
    built: dict[str, AbstractLatentHead] = {}
    for space, spec in specs.items():
        head = buildAdapter(model, space, spec, where)
        prefix = f"adapters.{space}."
        own = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        try:
            head.load_state_dict(own, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"{where}: the checkpoint's adapter of '{space}' does not fit a '{spec.head}' head "
                f"of width {spec.out_dim} on this model ({error}). Give the head and the width "
                f"that the training config gave it."
            ) from error
        built[space] = head.eval()
    return built
