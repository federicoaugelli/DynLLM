from __future__ import annotations

import math

import pytest

from dynllm.core.config import BackendType, ModelType, Settings


def test_laya_name_alone_applies_builtin() -> None:
    settings = Settings.model_validate({"models": [{"name": "laya"}]})
    model = settings.models[0]

    assert model.builtin == "laya"
    assert model.backend == BackendType.laya
    assert model.model_type == ModelType.decision
    assert model.vram_mb == 0
    assert model.device == "cpu"
    assert model.laya_preload is False
    assert model.laya_max_loaded == 2
    assert model.laya_default_model == "multilingual"
    assert model.unload_time == math.inf
    assert BackendType.laya in settings.enabled_backends


def test_builtin_can_use_a_different_model_name() -> None:
    settings = Settings.model_validate(
        {"models": [{"name": "router", "builtin": "laya", "device": "cpu"}]}
    )
    model = settings.models[0]

    assert model.builtin == "laya"
    assert model.backend == BackendType.laya
    assert model.name == "router"


def test_builtin_defaults_can_be_overridden() -> None:
    settings = Settings.model_validate(
        {
            "models": [
                {
                    "name": "laya",
                    "device": "cuda",
                    "vram_mb": 2048,
                    "laya_preload": True,
                    "laya_preload_models": ["english"],
                    "laya_max_loaded": 1,
                    "laya_model": "english",
                }
            ]
        }
    )
    model = settings.models[0]

    assert model.device == "cuda"
    assert model.vram_mb == 2048
    assert model.laya_preload is True
    assert model.laya_preload_models == ["english"]
    assert model.laya_max_loaded == 1
    assert model.laya_model == "english"


def test_explicit_backend_and_path_disable_name_inference() -> None:
    settings = Settings.model_validate(
        {
            "models": [
                {
                    "name": "laya",
                    "path": "/mnt/models/custom",
                    "backend": "openvino",
                    "vram_mb": 100,
                }
            ]
        }
    )
    model = settings.models[0]

    assert model.builtin is None
    assert model.backend == BackendType.openvino


def test_repeating_the_builtin_backend_keeps_defaults() -> None:
    settings = Settings.model_validate(
        {"models": [{"name": "laya", "backend": "laya"}]}
    )
    model = settings.models[0]

    assert model.builtin == "laya"
    assert model.model_type == ModelType.decision


def test_unknown_builtin_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown builtin 'nope'"):
        Settings.model_validate({"models": [{"name": "x", "builtin": "nope"}]})


def test_laya_fields_are_rejected_on_other_backends() -> None:
    with pytest.raises(ValueError, match="laya_preload is only supported"):
        Settings.model_validate(
            {
                "models": [
                    {
                        "name": "phi",
                        "path": "/tmp/phi",
                        "backend": "openvino",
                        "vram_mb": 0,
                        "laya_preload": True,
                    }
                ]
            }
        )


def test_laya_gpu_requires_vram_budget() -> None:
    with pytest.raises(ValueError, match="requires vram_mb > 0"):
        Settings.model_validate({"models": [{"name": "laya", "device": "auto"}]})


def test_laya_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="laya device must be one of"):
        Settings.model_validate({"models": [{"name": "laya", "device": "vulkan"}]})
