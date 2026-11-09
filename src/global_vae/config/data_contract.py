"""Agreement between what the loader owns and what the model declares (roadmap P5-7, ADR 0049).

Three numbers are fixed by the loader and repeated by the model: the sample rate of an audio
modality, the vocabulary size of a text modality and the categorical block sizes of a table. The
framework never fills them in (they are data-dependent, spec §10 and §12), so two files each held a
copy and nothing compared them. `DataConfig.loader_facts` is where the loader states them, and
`checkDataContract` is the comparison with the model's own arguments.

This is the one module that reads both config domains. `config/model*.py` knows nothing about the
data and `config/data.py` nothing about the model, on purpose (their docstrings say so), so the
check cannot live in either. `loadExperimentConfig` and `scripts/train.py` call it right after the
config is materialized, before any model is built, so a disagreement stops the run before the first
step and not at the first batch.

A model value is compared only when there is one: an argument the model leaves out, and whose class
has no default for it, is not compared. A fact the data block leaves out is not compared either, so
a config that sets none of them is checked against nothing and behaves as it always did.

Every error starts with the dotted path of the offending field (ADR 0025), the field of the model
when the two disagree, since the loader owns the value and the model is the one that has to change.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from global_vae.config.data import DataConfig, LoaderFactsConfig
from global_vae.config.model_resolution import collectDecoders, collectEncoders
from global_vae.config.model_schema import DecoderConfig, EncoderConfig, ModelConfig
from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass

FACT_NAMES = ("sample_rate", "vocab_size", "categorical_sizes")
"""The fields of `LoaderFactsConfig`, which are also the names of the model arguments they match."""


def _isInteger(value: Any, minimum: int) -> bool:
    """Whether `value` is an integer (a `bool` is not one here) of at least `minimum`."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validateLoaderFacts(data: DataConfig, prefix: str = "data") -> None:
    """Check the values of `data.loader_facts` on their own, without looking at any model.

    Args:
        data: The data config.
        prefix: Dotted path of `data`, the start of every error path.

    Raises:
        ValueError: If a modality's entry is not a `LoaderFactsConfig`, a `sample_rate` or
            `vocab_size` is not a positive integer, or `categorical_sizes` is not a non-empty
            list of integers that are each at least 2. The message starts with the path of the
            field, for example `data.loader_facts.audio.sample_rate`.
    """
    for modality, facts in (data.loader_facts or {}).items():
        path = f"{prefix}.loader_facts.{modality}"
        if not isinstance(facts, LoaderFactsConfig):
            raise ValueError(
                f"{path}: expected a block with the fields {list(FACT_NAMES)}, got {facts!r}."
            )
        for fact in ("sample_rate", "vocab_size"):
            value = getattr(facts, fact)
            if value is not None and not _isInteger(value, 1):
                raise ValueError(f"{path}.{fact}: expected a positive integer, got {value!r}.")
        sizes = facts.categorical_sizes
        if sizes is not None and (
            isinstance(sizes, str)
            or not isinstance(sizes, Sequence)
            or len(sizes) == 0
            or not all(_isInteger(size, 2) for size in sizes)
        ):
            raise ValueError(
                f"{path}.categorical_sizes: expected a non-empty list of integers, each at least "
                f"2 (a column with one category carries no information), got {sizes!r}."
            )


def _normalized(value: Any) -> Any:
    """A list for any sequence that is not a string, so `(3, 5)` and `[3, 5]` compare equal."""
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return value


def _modelValue(
    component: EncoderConfig | DecoderConfig, fact: str, getClass: Callable[[str], Any]
) -> tuple[Any, str] | None:
    """The value a model component has for `fact`, and how it came to have it.

    Args:
        component: The config of one encoder or decoder.
        fact: One of `FACT_NAMES`.
        getClass: The registry lookup of that kind of component.

    Returns:
        `(value, how)` where `how` is `"declares"` for a value written in the config and
        `"leaves unset, so the class uses"` for the default of the class's own argument. `None`
        when there is nothing to compare: no such argument in the config or in the class, a
        required argument left out (the class raises on its own), or a `null` or a `None` default,
        which both mean "not declared".
    """
    if fact in component.kwargs:
        value = component.kwargs[fact]
        return None if value is None else (value, "declares")
    try:
        parameter = inspect.signature(getClass(component.name)).parameters.get(fact)
    except (KeyError, ValueError, TypeError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    if parameter.default is None:
        return None
    return parameter.default, "leaves unset, so the class uses"


def checkDataContract(
    model: ModelConfig, data: DataConfig, model_prefix: str = "model", data_prefix: str = "data"
) -> None:
    """Fail when a model declares a value that the loader's facts contradict.

    For every modality of `data.loader_facts` and every fact given there, each encoder and decoder
    of that modality (the instances named like it) is compared with the same-named argument:
    `sample_rate` on the audio classes, `vocab_size` on the text classes, `categorical_sizes` on
    the tabular decoder. A model that declares nothing is not compared, and neither is a fact the
    data block does not give, so `data.loader_facts = None` returns at once without reading the
    model. A model that leaves out an argument whose class has a default is compared on that
    default: a table that has two categorical blocks and a decoder with no `categorical_sizes` is a
    disagreement, since the decoder then treats every column as numeric.

    Call it once the config is materialized and before any model is built.
    `global_vae.config.experiment.loadExperimentConfig` and `scripts/train.py` do. A config built
    by hand in Python is checked by calling this function.

    Args:
        model: The model config.
        data: The data config.
        model_prefix: Dotted path of `model`, the start of the error path of a model field.
        data_prefix: Dotted path of `data`, the start of the error path of a data field.

    Raises:
        ValueError: If a value of `data.loader_facts` is invalid (`validateLoaderFacts`), a
            modality of `loader_facts` is not an encoder or decoder name of the model (the message
            starts with `data.loader_facts.<modality>` and lists the names), or a model value
            differs from the loader's. The last message starts with the path of the model field,
            for example `model.modalities.audio.encoder.kwargs.sample_rate`, and names the data
            field it disagrees with and both values.
    """
    validateLoaderFacts(data, data_prefix)
    if not data.loader_facts:
        return
    encoders = collectEncoders(model, model_prefix)
    decoders = collectDecoders(model, model_prefix)
    names = sorted(set(encoders) | set(decoders))
    for modality, facts in data.loader_facts.items():
        data_path = f"{data_prefix}.loader_facts.{modality}"
        if modality not in names:
            raise ValueError(
                f"{data_path}: the model has no encoder or decoder named '{modality}', so nothing "
                f"would be compared. The names of the model are {names}. Rename the entry, or "
                f"remove it."
            )
        components: list[tuple[EncoderConfig | DecoderConfig, str, Callable[[str], Any]]] = []
        if modality in encoders:
            components.append((*encoders[modality], getEncoderClass))
        if modality in decoders:
            components.append((*decoders[modality], getDecoderClass))
        for fact in FACT_NAMES:
            expected = getattr(facts, fact)
            if expected is None:
                continue
            for component, component_path, getClass in components:
                found = _modelValue(component, fact, getClass)
                if found is None:
                    continue
                value, how = found
                if _normalized(value) != _normalized(expected):
                    raise ValueError(
                        f"{component_path}.kwargs.{fact}: the model {how} {value!r}, but "
                        f"{data_path}.{fact} says the loader gives {expected!r}. The loader owns "
                        f"this value: set the model to {expected!r}, or change the loader and "
                        f"this block together."
                    )
