"""`utils.seed.setGlobalSeed`, both modes (spec section 10, roadmap P5-9).

The `deterministic=False` branch ran in every training test. The `deterministic=True` branch
ran in none, so it was 78% covered: nothing checked that it sets the flags it documents, that it
leaves a `CUBLAS_WORKSPACE_CONFIG` the caller already set alone, or that a later `False` call
resets what a `True` call set. The function changes process-wide state (PyTorch's determinism
mode, two cuDNN flags, an environment variable and four generators), so every test runs inside a
fixture that puts all of it back, and no other test sees what these do.
"""

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from global_vae.utils.seed import setGlobalSeed

_CUBLAS = "CUBLAS_WORKSPACE_CONFIG"


@pytest.fixture(autouse=True)
def restoreGlobalState() -> Iterator[None]:
    """Save the process-wide state that `setGlobalSeed` touches, and restore it afterwards."""
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cublas = os.environ.get(_CUBLAS)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cublas is None:
            os.environ.pop(_CUBLAS, None)
        else:
            os.environ[_CUBLAS] = cublas
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _draws() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(()))


class TestTheSeed:
    def test_the_same_seed_gives_the_same_draws_from_all_three_generators(self) -> None:
        setGlobalSeed(123)
        first = _draws()
        setGlobalSeed(123)
        assert _draws() == first

    def test_another_seed_gives_other_draws(self) -> None:
        setGlobalSeed(1)
        first = _draws()
        setGlobalSeed(2)
        assert _draws() != first

    def test_the_seed_does_not_depend_on_the_mode(self) -> None:
        setGlobalSeed(7, deterministic=False)
        free = _draws()
        setGlobalSeed(7, deterministic=True)
        assert _draws() == free


class TestTheDefaultMode:
    def test_it_lets_cudnn_auto_tune_and_does_not_ask_for_deterministic_algorithms(self) -> None:
        setGlobalSeed(0)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark

    def test_it_does_not_touch_the_cublas_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_CUBLAS, raising=False)
        setGlobalSeed(0)
        assert _CUBLAS not in os.environ


class TestTheDeterministicMode:
    def test_it_sets_every_flag_it_documents(self) -> None:
        setGlobalSeed(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark

    def test_it_sets_the_cublas_variable_when_the_caller_has_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_CUBLAS, raising=False)
        setGlobalSeed(0, deterministic=True)
        assert os.environ[_CUBLAS] == ":4096:8"

    def test_it_keeps_a_cublas_variable_the_caller_has_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_CUBLAS, ":16:8")
        setGlobalSeed(0, deterministic=True)
        assert os.environ[_CUBLAS] == ":16:8"

    def test_warn_only_turns_an_unavailable_algorithm_into_a_warning(self) -> None:
        setGlobalSeed(0, deterministic=True, warn_only=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

    def test_a_later_default_call_resets_every_flag_it_set(self) -> None:
        setGlobalSeed(0, deterministic=True, warn_only=True)
        setGlobalSeed(0, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark

    def test_a_model_built_twice_with_one_seed_has_the_same_weights(self) -> None:
        setGlobalSeed(5, deterministic=True)
        first = torch.nn.Linear(4, 3).weight.detach().clone()
        setGlobalSeed(5, deterministic=True)
        assert torch.equal(torch.nn.Linear(4, 3).weight, first)
