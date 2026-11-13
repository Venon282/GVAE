"""Stage manifest: what a checkpoint of a staged recipe says about itself (roadmap P6-1, ADR 0051).

A staged recipe trains a model in several runs that hand checkpoints to each other: a solo run per
modality, then a pair run that starts from both (Phase 6). A checkpoint of such a run has to say, in
a form a program can read, which modality it covers, which latent space it has and how wide, and
which model config built it, so that the run that loads it can refuse a wrong file before the first
weight is copied and not at the first shape mismatch.

`StageManifest` is that record. It is made of plain values only (strings, numbers, `None`, lists and
dicts), the rule of `training/checkpoint.py` (ADR 0034), so a checkpoint that carries it is still
read by `torch.load(weights_only=True)`. It is opt-in: a run sets `training.stage` and gets a
manifest, a run that does not writes the file it always wrote.

This module owns the record, its builder and the digest. `training/checkpoint.py` owns where it sits
in the file (one key, `stage_manifest`, present only when there is a manifest), and `Trainer` holds
the manifest of its run so that every writer of a checkpoint, the callbacks included, writes it.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from global_vae.models.global_vae import GlobalVae


@dataclass(frozen=True)
class StageManifest:
    """The identity of one stage of a staged recipe, as stored in its checkpoints.

    Attributes:
        stage: A free label for the stage (`"solo_signal"`, `"pair"`, ...), or `None`. The
            framework never interprets it; two stages may share one.
        modalities: The batch keys the model's encoders ingest, in encoder order
            (`GlobalVae.input_modalities`).
        latent_spaces: Latent space name to its width (`LatentSpace.dim`), in the order of the
            model.
        config_digest: `computeConfigDigest` of the model config that built the model, or `None`
            when the manifest was built without one.

    Raises:
        ValueError: If `stage` is given and is not a non-empty string.
    """

    stage: str | None = None
    modalities: tuple[str, ...] = ()
    latent_spaces: dict[str, int] = field(default_factory=dict)
    config_digest: str | None = None

    def __post_init__(self) -> None:
        """Refuse a stage label that is empty or not a string."""
        if self.stage is not None and (not isinstance(self.stage, str) or not self.stage.strip()):
            raise ValueError(
                f"The stage label must be a non-empty string or None, got {self.stage!r}."
            )

    def toDict(self) -> dict[str, Any]:
        """Return the manifest as a dict of plain values, the form stored in a checkpoint.

        Returns:
            `{"stage", "modalities", "latent_spaces", "config_digest"}` with `modalities` as a
            list, so that the dict is the same before and after a round trip through JSON.
        """
        return {
            "stage": self.stage,
            "modalities": list(self.modalities),
            "latent_spaces": dict(self.latent_spaces),
            "config_digest": self.config_digest,
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> "StageManifest":
        """Rebuild a manifest from `toDict`'s output.

        A key that is absent keeps its default, and a key this version does not know is ignored,
        so a manifest written by a later version, with more fields, still reads.

        Args:
            data: The dict stored in a checkpoint under `stage_manifest`.

        Returns:
            The manifest.

        Raises:
            ValueError: If a field it knows has the wrong type (`modalities` not a list of
                strings, `latent_spaces` not a mapping from a string to an integer, `stage` or
                `config_digest` neither `None` nor a string).
        """
        stage = data.get("stage")
        config_digest = data.get("config_digest")
        modalities = data.get("modalities", ())
        latent_spaces = data.get("latent_spaces", {})
        if config_digest is not None and not isinstance(config_digest, str):
            raise ValueError(
                f"The manifest's config_digest must be a string or None, got {config_digest!r}."
            )
        if not isinstance(modalities, list | tuple) or not all(
            isinstance(name, str) for name in modalities
        ):
            raise ValueError(
                f"The manifest's modalities must be a list of strings, got {modalities!r}."
            )
        if not isinstance(latent_spaces, Mapping) or not all(
            isinstance(name, str) and isinstance(width, int) and not isinstance(width, bool)
            for name, width in latent_spaces.items()
        ):
            raise ValueError(
                f"The manifest's latent_spaces must map a name to an integer width, "
                f"got {latent_spaces!r}."
            )
        return cls(
            stage=stage,
            modalities=tuple(modalities),
            latent_spaces=dict(latent_spaces),
            config_digest=config_digest,
        )


def computeConfigDigest(config: Any) -> str:
    """Return a stable fingerprint of a model config.

    The digest is the SHA-256 of the config as canonical JSON: a dataclass instance
    (`ModelConfig`) is turned into a dict with `dataclasses.asdict`, keys are sorted, and there
    is no insignificant whitespace. Two configs with the same content give the same digest
    whatever the order their keys were written in, in any process and on any machine, and a
    changed value gives another one. It is the fingerprint of the config as written: an argument
    left to its default in one config and spelled out in another gives two digests.

    Args:
        config: A dataclass instance, or a mapping, made of plain values (`dict`, `list`,
            `tuple`, `str`, number, `bool`, `None`). An `OmegaConf` node must first go through
            `OmegaConf.to_container`.

    Returns:
        The lowercase hexadecimal digest, 64 characters.

    Raises:
        TypeError: If the config holds a value JSON cannot represent. It is an error and not a
            silent `str()`, because the text of an object may hold a memory address and the
            digest would differ on every run.
    """
    payload = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config

    def refuse(value: object) -> object:
        raise TypeError(
            f"Cannot digest a config that holds {type(value).__name__} {value!r}: only plain "
            f"values (dict, list, tuple, str, number, bool, None) have a stable fingerprint."
        )

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=refuse)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def buildStageManifest(
    model: "GlobalVae", stage: str | None = None, model_config: Any = None
) -> StageManifest:
    """Describe a built model as the manifest of one stage.

    Args:
        model: The model the stage trains.
        stage: The free label of the stage, or `None`.
        model_config: The `ModelConfig` (or a mapping of plain values) that built `model`, for
            the digest. `None` (default) leaves `config_digest` at `None`.

    Returns:
        The manifest: the model's input modalities, the width of each of its latent spaces and,
        when `model_config` is given, its digest.

    Raises:
        ValueError: If `stage` is given and is not a non-empty string.
        TypeError: If `model_config` holds a value `computeConfigDigest` refuses.
    """
    return StageManifest(
        stage=stage,
        modalities=tuple(model.input_modalities),
        latent_spaces={name: space.dim for name, space in model.latent_spaces.items()},
        config_digest=None if model_config is None else computeConfigDigest(model_config),
    )
