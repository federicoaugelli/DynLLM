from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dynllm.api import routes
from dynllm.backends.laya import LayaBackend
from dynllm.main import create_app


class DummyVRAM:
    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self._ports: dict[str, int] = {}
        self.instances: dict[str, object] = {}

    async def ensure_loaded(self, model):
        self.loaded_models.append(model.name)
        return self._ports.setdefault(model.name, 9123)

    async def get_port(self, model_name: str):
        return self._ports.get(model_name)

    def get_instance(self, model_name: str):
        return self.instances.get(model_name)

    async def increment_active(self, model_name: str) -> None:
        pass

    async def decrement_active(self, model_name: str) -> None:
        pass


class DummyLayaBackend(LayaBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, dict]] = []

    async def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {
            "answers": {"billing": {"choice": "billing", "confidence": 0.9}},
            "routing": {"model": "english"},
        }


class DummyState:
    def __init__(self) -> None:
        self.touched: list[str] = []

    async def touch(self, model_name: str) -> None:
        self.touched.append(model_name)


@pytest.fixture
def app(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
auto_install: false
enabled_backends:
  - llamacpp
models:
  - name: clef
    path: /tmp/Clef.gguf
    backend: llamacpp
    model_type: decision
    vram_mb: 16000
  - name: llama3
    path: /tmp/llama3.gguf
    backend: llamacpp
    model_type: llm
    vram_mb: 5500
  - name: laya
    device: cpu
""".strip()
    )
    return create_app(str(config_path))


@pytest.fixture
def client(app, monkeypatch: pytest.MonkeyPatch):
    dummy_vram = DummyVRAM()
    dummy_state = DummyState()
    app.state.vram_manager = dummy_vram
    app.state.state_manager = dummy_state

    forwarded: list[tuple[int, str, bytes]] = []

    async def fake_forward_request(request, port, path, body):
        forwarded.append((port, path, body))
        return routes.Response(content=b"{}", media_type="application/json")

    monkeypatch.setattr(routes, "forward_request", fake_forward_request)
    dummy_vram.instances["laya"] = DummyLayaBackend()
    app.state._dummy_vram = dummy_vram
    app.state._dummy_state = dummy_state
    app.state._forwarded = forwarded
    return TestClient(app)


def test_systemone_proxies_to_backend(client):
    response = client.post(
        "/v1/systemone",
        json={
            "model": "clef",
            "state": "I was charged twice.",
            "questions": {
                "route": {
                    "type": "choice",
                    "criteria": {"billing": "refunds", "shipping": "parcels"},
                }
            },
        },
    )

    assert response.status_code == 200
    assert client.app.state._dummy_vram.loaded_models == ["clef"]
    assert client.app.state._dummy_state.touched == ["clef"]
    port, path, body = client.app.state._forwarded[0]
    assert path == "v1/systemone"
    assert b'"model":"clef"' in body


def test_systemone_rejects_llm_model(client):
    response = client.post(
        "/v1/systemone",
        json={
            "model": "llama3",
            "state": "text",
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        },
    )

    assert response.status_code == 400
    assert "configured as 'llm'" in response.json()["detail"]


def test_systemone_unknown_model(client):
    response = client.post(
        "/v1/systemone",
        json={"model": "ghost", "state": "text", "questions": {}},
    )

    assert response.status_code == 404


def test_systemone_dispatches_to_in_process_laya(client):
    backend = client.app.state._dummy_vram.instances["laya"]

    response = client.post(
        "/v1/systemone",
        json={
            "model": "laya",
            "state": "I was charged twice.",
            "questions": {
                "billing": {
                    "type": "choice",
                    "criteria": {"billing": "refunds", "other": "everything else"},
                }
            },
            "lang": "en",
            "max_len": 512,
            "unknown_control": "ignored",
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"]["billing"]["choice"] == "billing"
    assert client.app.state._dummy_vram.loaded_models == ["laya"]
    assert client.app.state._dummy_state.touched == ["laya"]
    # The request must not be sent to a subprocess proxy.
    assert client.app.state._forwarded == []

    state, questions, kwargs = backend.calls[0]
    assert state == "I was charged twice."
    assert "billing" in questions
    assert kwargs["lang"] == "en"
    assert kwargs["max_len"] == 512
    assert kwargs.get("model") is None
