from __future__ import annotations

from pathlib import Path

import pytest

from dynllm.backends import BackendRegistry
from dynllm.backends.neutronstar import NeutronStarBackend
from dynllm.core.config import BackendType, ModelConfig, Settings


def _settings(*backends: BackendType) -> Settings:
    return Settings(enabled_backends=list(backends), models=[])


def test_registry_creates_neutronstar_when_enabled() -> None:
    registry = BackendRegistry(_settings(BackendType.neutronstar))
    backend = registry.create(BackendType.neutronstar)
    assert isinstance(backend, NeutronStarBackend)
    assert backend.backend_type == BackendType.neutronstar


def test_registry_rejects_disabled_backend() -> None:
    registry = BackendRegistry(_settings(BackendType.llamacpp))
    with pytest.raises(RuntimeError, match="not enabled"):
        registry.create(BackendType.neutronstar)


@pytest.mark.anyio
async def test_neutronstar_start_is_a_scaffold() -> None:
    backend = NeutronStarBackend()
    model = ModelConfig(
        name="ornith",
        path=Path("/tmp/ornith.gguf"),
        backend=BackendType.neutronstar,
        vram_mb=1,
    )
    with pytest.raises(NotImplementedError, match="scaffold"):
        await backend.start(model, 9100)
