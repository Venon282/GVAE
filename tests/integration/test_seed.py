"""Unit tests for `utils.seed.setGlobalSeed` (spec §10 reproducibility requirement)."""

import os
import random

import numpy as np
import torch

from global_vae.utils.seed import setGlobalSeed


def test_same_seed_reproduces_python_random() -> None:
    setGlobalSeed(42)
    first = [random.random() for _ in range(5)]
    setGlobalSeed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_same_seed_reproduces_numpy() -> None:
    setGlobalSeed(42)
    first = np.random.rand(5)
    setGlobalSeed(42)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_same_seed_reproduces_torch() -> None:
    setGlobalSeed(42)
    first = torch.randn(5)
    setGlobalSeed(42)
    second = torch.randn(5)
    assert torch.equal(first, second)


def test_same_seed_reproduces_model_initialization() -> None:
    """The actual motivating case: two freshly-constructed models should get identical weights."""
    setGlobalSeed(7)
    first_layer = torch.nn.Linear(16, 4)
    setGlobalSeed(7)
    second_layer = torch.nn.Linear(16, 4)
    assert torch.equal(first_layer.weight, second_layer.weight)
    assert torch.equal(first_layer.bias, second_layer.bias)


def test_different_seeds_give_different_sequences() -> None:
    setGlobalSeed(1)
    first = torch.randn(5)
    setGlobalSeed(2)
    second = torch.randn(5)
    assert not torch.equal(first, second)


class TestDeterministicFlag:
    def test_deterministic_true_sets_backend_flags(self) -> None:
        setGlobalSeed(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_deterministic_false_resets_backend_flags(self) -> None:
        setGlobalSeed(0, deterministic=True)
        setGlobalSeed(0, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_deterministic_true_sets_cublas_workspace_env_var(self) -> None:
        setGlobalSeed(0, deterministic=True)
        assert "CUBLAS_WORKSPACE_CONFIG" in os.environ

    def test_cublas_workspace_env_var_does_not_override_an_existing_value(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            setGlobalSeed(0, deterministic=True)
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        finally:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

    def test_warn_only_does_not_raise(self) -> None:
        setGlobalSeed(0, deterministic=True, warn_only=True)
        setGlobalSeed(0, deterministic=False)  # reset for any tests that run after this one
