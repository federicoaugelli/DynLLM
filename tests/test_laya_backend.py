from __future__ import annotations

import sys
import types

import pytest

from dynllm.backends.laya import LayaBackend
from dynllm.core.config import ModelConfig, Settings


class FakeRouter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.preloaded: object = None
        self.unloaded = False
        self.predict_calls: list[tuple[object, object, dict]] = []

    def preload(self, names) -> None:
        self.preloaded = names

    def unload(self) -> None:
        self.unloaded = True

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append((state, questions, kwargs))
        return {"answers": {"q": {"noul": 0.5}}, "routing": {"model": "english"}}


def _laya_model(**overrides) -> ModelConfig:
    raw = {"name": "laya"}
    raw.update(overrides)
    return Settings.model_validate({"models": [raw]}).models[0]


def _install_fake_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("laya")
    module.Router = FakeRouter
    monkeypatch.setitem(sys.modules, "laya", module)


@pytest.mark.asyncio
async def test_load_is_lazy_and_predict_forwards_controls(monkeypatch) -> None:
    _install_fake_laya(monkeypatch)
    backend = LayaBackend()
    model = _laya_model(device="cpu", laya_preload=False)

    await backend._load(model)

    assert backend.loaded is True
    router = backend._router
    assert isinstance(router, FakeRouter)
    assert router.kwargs["max_loaded"] == 2
    assert router.kwargs["device"] == "cpu"
    assert "preload" not in router.kwargs
    assert router.preloaded is None

    result = await backend.predict(
        "text",
        {"q": {"type": "noul"}},
        model="english",
        lang="en",
        max_len=512,
        ignored="value",
    )

    assert result["routing"]["model"] == "english"
    assert router.predict_calls == [
        (
            "text",
            {"q": {"type": "noul"}},
            {"lang": "en", "max_len": 512, "model": "english"},
        )
    ]


@pytest.mark.asyncio
async def test_preload_uses_configured_checkpoints(monkeypatch) -> None:
    _install_fake_laya(monkeypatch)
    backend = LayaBackend()
    model = _laya_model(
        device="cpu",
        laya_preload=True,
        laya_preload_models=["english"],
        laya_max_loaded=1,
    )

    await backend._load(model)

    router = backend._router
    assert router.preloaded == ["english"]
    assert router.kwargs["max_loaded"] == 1


@pytest.mark.asyncio
async def test_unload_releases_router(monkeypatch) -> None:
    _install_fake_laya(monkeypatch)
    backend = LayaBackend()
    await backend._load(_laya_model(device="cpu"))
    router = backend._router

    await backend._unload()

    assert backend.loaded is False
    assert backend._router is None
    assert router.unloaded is True


@pytest.mark.asyncio
async def test_missing_laya_package_has_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "laya", None)
    backend = LayaBackend()

    with pytest.raises(RuntimeError, match="uv pip install 'laya"):
        await backend._load(_laya_model(device="cpu"))
