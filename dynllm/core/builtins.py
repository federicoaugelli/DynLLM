"""Built-in model presets.

A built-in model is a complete DynLLM model declaration with sensible defaults.
Users opt in by naming the model after the preset (``- name: laya``) or by
setting ``builtin: <name>`` explicitly.  Every field can still be overridden.

The defaults are deliberately free of imports from :mod:`dynllm.core.config`
so that the configuration module can use them before its own classes exist.
"""

from __future__ import annotations

from typing import Any

BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "laya": {
        # In-process System One decision model (ModernBERT + decision head).
        "backend": "laya",
        "model_type": "decision",
        "path": "convaiinnovations/laya",
        "vram_mb": 0,
        "device": "cpu",
        "laya_preload": False,
        "laya_max_loaded": 2,
        "laya_default_model": "multilingual",
        "unload_time": -1,
    },
}

BUILTIN_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    # The PyTorch package (not ``laya[serve]``: DynLLM runs the Router
    # in-process and needs no HTTP server).  ``torch`` is listed so the
    # installer can check it, but auto-install deliberately refuses to add
    # or replace it: a default install would pull the CUDA wheel and could
    # clobber a custom XPU/CUDA build.
    "laya": (
        "laya",
        "torch",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "numpy",
    ),
}

# Version range used by the automatic installer.  The API used here
# (Router, lazy loading, predict controls) is stable across 0.4.x.
BUILTIN_PACKAGE_SPECS: dict[str, str] = {
    "laya": "laya>=0.4,<0.5",
}

__all__ = [
    "BUILTIN_DEFAULTS",
    "BUILTIN_PACKAGE_SPECS",
    "BUILTIN_REQUIREMENTS",
]
