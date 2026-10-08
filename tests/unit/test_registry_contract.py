"""Contract test shared by every self-registration registry (spec §10, roadmap P0-4(c)).

The codebase has nine registries (encoders, decoders, fusion strategies, assemblers, latent
regularizers, data transforms, beta schedules, callbacks and experiment loggers), each a small
module with the same three functions. This file states the behaviour they all promise and checks
it once per registry, so that a registry that drifts, or a new one that forgets part of the
pattern, fails here:

* a class registered under a name is returned by the lookup, unchanged;
* registering a name twice raises `ValueError` and keeps the first class;
* looking up an unknown name raises `KeyError`, and the message lists the available names;
* the `listRegistered*` function returns names in sorted order;
* importing the registry's subpackage registers at least one built-in.

A registry added later (the latent head registry of roadmap P1-1, for example) joins by adding
one `_RegistrySpec` to `_REGISTRIES`; `test_nine_registries_are_covered` makes forgetting that
visible. Behaviour that belongs to one registry only (which built-ins exist, what each does) stays
in that registry's own tests, such as `test_callback_registry.py` and `test_assemblers.py`.

Every test runs against a private copy of the registry's dictionary, swapped in with
`monkeypatch`, so the throwaway classes registered here never leak into other tests.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import (
    getAssemblerClass,
    listRegisteredAssemblers,
    registerAssembler,
)
from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import (
    getTransformClass,
    listRegisteredTransforms,
    registerTransform,
)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders, registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders, registerEncoder
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import getFusionClass, listRegisteredFusions, registerFusion
from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.registry import (
    getRegularizerClass,
    listRegisteredRegularizers,
    registerRegularizer,
)
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.registry import (
    getBetaScheduleClass,
    listRegisteredBetaSchedules,
    registerBetaSchedule,
)
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import (
    getCallbackClass,
    listRegisteredCallbacks,
    registerCallback,
)
from global_vae.training.loggers.base import AbstractExperimentLogger
from global_vae.training.loggers.registry import (
    getLoggerClass,
    listRegisteredLoggers,
    registerLogger,
)


@dataclass(frozen=True)
class _RegistrySpec:
    """Everything the contract needs to know about one registry.

    Attributes:
        label: Short name used as the pytest id.
        subpackage: Dotted path whose import registers the built-ins.
        registry_module: Dotted path of the module holding the registry dictionary.
        registry_attr: Name of that dictionary in `registry_module`.
        base: Abstract base class every entry subclasses.
        register: The `registerX(name)` class decorator.
        get: The `getXClass(name)` lookup.
        list_names: The `listRegisteredX()` function.
    """

    label: str
    subpackage: str
    registry_module: str
    registry_attr: str
    base: type
    register: Callable[[str], Callable[[Any], Any]]
    get: Callable[[str], type]
    list_names: Callable[[], list[str]]


_REGISTRIES: tuple[_RegistrySpec, ...] = (
    _RegistrySpec(
        "encoders",
        "global_vae.encoders",
        "global_vae.encoders.registry",
        "_ENCODER_REGISTRY",
        AbstractEncoder,
        registerEncoder,
        getEncoderClass,
        listRegisteredEncoders,
    ),
    _RegistrySpec(
        "decoders",
        "global_vae.decoders",
        "global_vae.decoders.registry",
        "_DECODER_REGISTRY",
        AbstractDecoder,
        registerDecoder,
        getDecoderClass,
        listRegisteredDecoders,
    ),
    _RegistrySpec(
        "fusion",
        "global_vae.fusion",
        "global_vae.fusion.registry",
        "_FUSION_REGISTRY",
        AbstractFusion,
        registerFusion,
        getFusionClass,
        listRegisteredFusions,
    ),
    _RegistrySpec(
        "assemblers",
        "global_vae.assemblers",
        "global_vae.assemblers.registry",
        "_ASSEMBLER_REGISTRY",
        AbstractAssembler,
        registerAssembler,
        getAssemblerClass,
        listRegisteredAssemblers,
    ),
    _RegistrySpec(
        "regularizers",
        "global_vae.losses.regularizers",
        "global_vae.losses.regularizers.registry",
        "_REGULARIZER_REGISTRY",
        AbstractLatentRegularizer,
        registerRegularizer,
        getRegularizerClass,
        listRegisteredRegularizers,
    ),
    _RegistrySpec(
        "transforms",
        "global_vae.data.transforms",
        "global_vae.data.transforms.registry",
        "_TRANSFORM_REGISTRY",
        AbstractTransform,
        registerTransform,
        getTransformClass,
        listRegisteredTransforms,
    ),
    _RegistrySpec(
        "beta_schedules",
        "global_vae.training.beta_schedules",
        "global_vae.training.beta_schedules.registry",
        "_BETA_SCHEDULE_REGISTRY",
        AbstractBetaSchedule,
        registerBetaSchedule,
        getBetaScheduleClass,
        listRegisteredBetaSchedules,
    ),
    _RegistrySpec(
        "callbacks",
        "global_vae.training.callbacks",
        "global_vae.training.callbacks.registry",
        "_CALLBACK_REGISTRY",
        TrainerCallback,
        registerCallback,
        getCallbackClass,
        listRegisteredCallbacks,
    ),
    _RegistrySpec(
        "loggers",
        "global_vae.training.loggers",
        "global_vae.training.loggers.registry",
        "_LOGGER_REGISTRY",
        AbstractExperimentLogger,
        registerLogger,
        getLoggerClass,
        listRegisteredLoggers,
    ),
)

_IDS = [spec.label for spec in _REGISTRIES]


def _newEntry(spec: _RegistrySpec, label: str) -> type:
    """Make a throwaway subclass of the registry's base class.

    The class is never instantiated, so it does not need to implement the base's abstract
    methods: registries store classes and do not check them.

    Args:
        spec: The registry the class is for.
        label: Distinguishes several classes made for the same registry.

    Returns:
        A new subclass of `spec.base`.
    """
    return type(f"_Contract{label}", (spec.base,), {})


@pytest.fixture(params=_REGISTRIES, ids=_IDS)
def spec(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _RegistrySpec:
    """One registry per test run, with its dictionary replaced by a private copy.

    Importing the subpackage first makes sure the built-ins are in the copy, whatever order the
    tests run in. `monkeypatch` puts the original dictionary back after the test.

    Args:
        request: Pytest request carrying the parametrized `_RegistrySpec`.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The `_RegistrySpec` under test.
    """
    registry_spec: _RegistrySpec = request.param
    importlib.import_module(registry_spec.subpackage)
    module = importlib.import_module(registry_spec.registry_module)
    original = getattr(module, registry_spec.registry_attr)
    monkeypatch.setattr(module, registry_spec.registry_attr, dict(original))
    return registry_spec


def test_nine_registries_are_covered() -> None:
    """The contract runs over all nine registries; adding a tenth means adding a spec."""
    assert len(_REGISTRIES) == 9
    assert len({s.registry_attr for s in _REGISTRIES}) == 9


class TestRegisterAndLookUp:
    """A registered class is found again under its name."""

    def test_lookup_returns_the_registered_class(self, spec: _RegistrySpec) -> None:
        """`getX(name)` returns the very class that was registered."""
        entry = _newEntry(spec, "Lookup")
        spec.register("contract_lookup")(entry)
        assert spec.get("contract_lookup") is entry

    def test_decorator_returns_the_class_unchanged(self, spec: _RegistrySpec) -> None:
        """Used as `@registerX(name)`, the decorator leaves the class as it was."""
        entry = _newEntry(spec, "Identity")
        assert spec.register("contract_identity")(entry) is entry

    def test_registered_name_is_listed(self, spec: _RegistrySpec) -> None:
        """The new name shows up in `listRegisteredX()`."""
        spec.register("contract_listed")(_newEntry(spec, "Listed"))
        assert "contract_listed" in spec.list_names()

    def test_two_names_resolve_independently(self, spec: _RegistrySpec) -> None:
        """Two registrations do not overwrite each other."""
        first, second = _newEntry(spec, "First"), _newEntry(spec, "Second")
        spec.register("contract_first")(first)
        spec.register("contract_second")(second)
        assert spec.get("contract_first") is first
        assert spec.get("contract_second") is second


class TestDuplicateName:
    """A name can be registered once."""

    def test_duplicate_raises_value_error(self, spec: _RegistrySpec) -> None:
        """The second registration of a name raises `ValueError`."""
        spec.register("contract_duplicate")(_newEntry(spec, "Original"))
        with pytest.raises(ValueError, match="already registered"):
            spec.register("contract_duplicate")(_newEntry(spec, "Impostor"))

    def test_duplicate_message_names_the_key(self, spec: _RegistrySpec) -> None:
        """The message says which name clashed."""
        spec.register("contract_clash")(_newEntry(spec, "Original"))
        with pytest.raises(ValueError, match="contract_clash"):
            spec.register("contract_clash")(_newEntry(spec, "Impostor"))

    def test_duplicate_keeps_the_first_class(self, spec: _RegistrySpec) -> None:
        """A refused registration does not replace what was already there."""
        original = _newEntry(spec, "Original")
        spec.register("contract_keep")(original)
        with pytest.raises(ValueError, match="already registered"):
            spec.register("contract_keep")(_newEntry(spec, "Impostor"))
        assert spec.get("contract_keep") is original

    def test_built_in_names_are_protected_too(self, spec: _RegistrySpec) -> None:
        """Re-using the name of a built-in raises as well, and the built-in stays."""
        built_in = spec.list_names()[0]
        before = spec.get(built_in)
        with pytest.raises(ValueError, match="already registered"):
            spec.register(built_in)(_newEntry(spec, "Impostor"))
        assert spec.get(built_in) is before


class TestUnknownName:
    """Looking up a name that was never registered fails with a useful message."""

    def test_unknown_raises_key_error(self, spec: _RegistrySpec) -> None:
        """An unknown name raises `KeyError`."""
        with pytest.raises(KeyError):
            spec.get("contract_does_not_exist")

    def test_message_names_the_unknown_key(self, spec: _RegistrySpec) -> None:
        """The message repeats the name that was asked for."""
        with pytest.raises(KeyError) as excinfo:
            spec.get("contract_does_not_exist")
        assert "contract_does_not_exist" in str(excinfo.value)

    def test_message_lists_every_available_name(self, spec: _RegistrySpec) -> None:
        """The message lists all registered names, including one added by this test."""
        spec.register("contract_available")(_newEntry(spec, "Available"))
        with pytest.raises(KeyError) as excinfo:
            spec.get("contract_does_not_exist")
        message = str(excinfo.value)
        names = spec.list_names()
        assert "contract_available" in names
        for name in names:
            assert name in message

    def test_message_lists_the_names_sorted_after_available(self, spec: _RegistrySpec) -> None:
        """The text after `Available: ` is the sorted listing, name for name."""
        spec.register("contract_zzz_last")(_newEntry(spec, "Last"))
        spec.register("contract_aaa_first")(_newEntry(spec, "First"))
        with pytest.raises(KeyError) as excinfo:
            spec.get("contract_does_not_exist")
        message = excinfo.value.args[0]
        listed = message.split("Available: ", 1)[1].split(", ")
        assert listed == spec.list_names()
        assert listed == sorted(listed)


class TestListing:
    """`listRegisteredX()` is a sorted list of names."""

    def test_returns_a_list_of_strings(self, spec: _RegistrySpec) -> None:
        """The return type is a plain list of `str`."""
        names = spec.list_names()
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)

    def test_is_sorted_after_out_of_order_registrations(self, spec: _RegistrySpec) -> None:
        """Names registered in reverse alphabetical order still come back sorted."""
        spec.register("contract_zzz")(_newEntry(spec, "Z"))
        spec.register("contract_mmm")(_newEntry(spec, "M"))
        spec.register("contract_aaa")(_newEntry(spec, "A"))
        names = spec.list_names()
        assert names == sorted(names)
        assert (
            names.index("contract_aaa") < names.index("contract_mmm") < names.index("contract_zzz")
        )

    def test_returns_a_fresh_list(self, spec: _RegistrySpec) -> None:
        """Editing the returned list does not edit the registry."""
        names = spec.list_names()
        names.clear()
        assert spec.list_names() != []


class TestBuiltIns:
    """Importing a registry's subpackage registers its built-in strategies (spec §10)."""

    def test_subpackage_registers_at_least_one_built_in(self, spec: _RegistrySpec) -> None:
        """A registry that is empty after its import would make every lookup a `KeyError`."""
        assert spec.list_names() != []

    def test_every_built_in_resolves_to_a_subclass_of_the_base(self, spec: _RegistrySpec) -> None:
        """Each registered built-in implements the registry's abstract interface."""
        for name in spec.list_names():
            assert issubclass(spec.get(name), spec.base), name
