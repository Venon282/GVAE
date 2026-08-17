"""Shared helper for dynamically importing a callable from a `"module.path:function_name"`
string (spec §10, §11: model/data construction stays the caller's own responsibility;
this is the one mechanism every entry point in this framework uses to reach into the
caller's own code without importing it directly).

Extracted from `scripts/evaluate.py`'s original private `_importCallable` so
`scripts/train.py` and `config/data.py` (spec's C9 Hydra config layer) can share the
exact same behavior instead of duplicating it. `scripts/evaluate.py` keeps a
`_importCallable = importCallable` alias for backward compatibility with its own
existing tests, which call the private name directly on the loaded script module.
"""

import importlib


def importCallable(spec: str) -> object:
    """Import and return the callable named by `"module.path:function_name"`.

    Args:
        spec: A string of the form `"module.path:function_name"`.

    Returns:
        The imported callable.

    Raises:
        ValueError: If `spec` does not contain exactly one `:` separator.
        ModuleNotFoundError: If `module.path` cannot be imported.
        AttributeError: If `function_name` does not exist on that module.
    """
    if spec.count(":") != 1:
        raise ValueError(f"Expected 'module.path:function_name', got '{spec}'.")
    module_path, function_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)
