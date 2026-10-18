from __future__ import annotations

import pytest

from dynllm.core.config import Settings
from dynllm.core.vram_manager import RequestTracker, VRAMManager
from dynllm.db.models import ModelState, ModelStatus


class FakeState:
    def __init__(self, models: list[ModelState]) -> None:
        self.models = models
        self.unloaded: list[str] = []

    async def total_loaded_vram(self) -> int:
        return sum(m.vram_mb for m in self.models if m.status == ModelStatus.loaded)

    async def get_loaded(self) -> list[ModelState]:
        return [m for m in self.models if m.status == ModelStatus.loaded]

    async def get(self, name: str):
        return next((m for m in self.models if m.name == name), None)

    async def set_unloading(self, name: str) -> None:
        model = await self.get(name)
        if model is not None:
            model.status = ModelStatus.unloading

    async def set_unloaded(self, name: str) -> None:
        model = await self.get(name)
        if model is not None:
            model.status = ModelStatus.unloaded
            model.vram_mb = 0
            self.unloaded.append(name)


class _TestManager(VRAMManager):
    def __init__(self, settings: Settings, state: FakeState) -> None:
        self._settings = settings
        self._state = state
        self._requests = RequestTracker()

    async def _unload_by_name(self, model_name: str) -> None:
        await self._state.set_unloading(model_name)
        await self._state.set_unloaded(model_name)


def _state(name: str, vram_mb: int, order: int) -> ModelState:
    return ModelState(
        name=name,
        status=ModelStatus.loaded,
        backend="test",
        vram_mb=vram_mb,
        load_order=order,
    )


@pytest.mark.asyncio
async def test_zero_vram_model_is_not_evicted_for_gpu_space() -> None:
    laya = _state("laya", 0, order=3)
    gpu_a = _state("gpu-a", 60, order=1)
    gpu_b = _state("gpu-b", 40, order=2)
    state = FakeState([laya, gpu_a, gpu_b])
    manager = _TestManager(Settings(total_vram_mb=100), state)

    await manager._evict_for(30)

    assert state.unloaded == ["gpu-b"]
    assert laya.status == ModelStatus.loaded


@pytest.mark.asyncio
async def test_zero_vram_only_reports_no_evictable_model() -> None:
    laya = _state("laya", 0, order=1)
    state = FakeState([laya])
    manager = _TestManager(Settings(total_vram_mb=5), state)

    with pytest.raises(RuntimeError, match="No loaded model can free GPU memory"):
        await manager._evict_for(10)

    assert state.unloaded == []


@pytest.mark.asyncio
async def test_busy_and_zero_vram_models_are_named_in_error() -> None:
    laya = _state("laya", 0, order=1)
    gpu = _state("gpu", 60, order=2)
    state = FakeState([laya, gpu])
    manager = _TestManager(Settings(total_vram_mb=100), state)
    await manager.increment_active("gpu")

    with pytest.raises(RuntimeError, match="Busy models: \\['gpu'\\]"):
        await manager._evict_for(50)
