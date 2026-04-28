from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dynllm.api import routes
from dynllm.main import create_app


class DummyVRAM:
    def __init__(self) -> None:
        self.loaded_models: list[str] = []

    async def ensure_loaded(self, model):  # noqa: ANN001
        self.loaded_models.append(model.name)
        return 9123


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
enabled_backends:
  - openvino
models:
  - name: whisper-ov
    path: /tmp/whisper-ov
    backend: openvino
    model_type: transcription
    target_device: CPU
    vram_mb: 0
  - name: speech-ov
    path: /tmp/speech-ov
    backend: openvino
    model_type: speech
    target_device: CPU
    vram_mb: 0
  - name: phi3-ov
    path: /tmp/phi3-ov
    backend: openvino
    model_type: llm
    target_device: GPU
    vram_mb: 4096
""".strip()
    )
    return create_app(str(config_path))


@pytest.fixture
def client(app, monkeypatch: pytest.MonkeyPatch):
    dummy_vram = DummyVRAM()
    dummy_state = DummyState()
    routes.set_managers(dummy_vram, dummy_state)

    forwarded: list[tuple[int, str, bytes]] = []

    async def fake_forward_request(request, port, path, body):  # noqa: ANN001
        forwarded.append((port, path, body))
        return routes.Response(content=b"ok", media_type="application/json")

    monkeypatch.setattr(routes, "forward_request", fake_forward_request)
    app.state._dummy_vram = dummy_vram
    app.state._dummy_state = dummy_state
    app.state._forwarded = forwarded
    return TestClient(app)


def test_audio_transcriptions_proxy_to_ovms(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", b"wav-data", "audio/wav")},
        data={"model": "whisper-ov"},
    )

    assert response.status_code == 200
    assert client.app.state._dummy_vram.loaded_models == ["whisper-ov"]
    assert client.app.state._dummy_state.touched == ["whisper-ov"]
    assert client.app.state._forwarded[0][1] == "v3/audio/transcriptions"


def test_audio_translations_proxy_to_ovms(client):
    response = client.post(
        "/v1/audio/translations",
        files={"file": ("speech.wav", b"wav-data", "audio/wav")},
        data={"model": "whisper-ov"},
    )

    assert response.status_code == 200
    assert client.app.state._forwarded[0][1] == "v3/audio/translations"


def test_audio_speech_proxy_to_ovms(client):
    response = client.post(
        "/v1/audio/speech",
        json={"model": "speech-ov", "input": "Hello world"},
    )

    assert response.status_code == 200
    assert client.app.state._dummy_vram.loaded_models == ["speech-ov"]
    assert client.app.state._dummy_state.touched == ["speech-ov"]
    assert client.app.state._forwarded[0][1] == "v3/audio/speech"


def test_chat_rejects_transcription_model(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "whisper-ov",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert "configured as 'transcription'" in response.json()["detail"]


def test_transcription_rejects_llm_model(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", b"wav-data", "audio/wav")},
        data={"model": "phi3-ov"},
    )

    assert response.status_code == 400
    assert "configured as 'llm'" in response.json()["detail"]


def test_transformers_transcription_rewrites_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enabled_backends:
  - transformers
models:
  - name: whisper-hf
    path: /tmp/whisper-hf
    backend: transformers
    model_type: transcription
    device: xpu
    dtype: float16
    vram_mb: 2048
""".strip()
    )
    app = create_app(str(config_path))

    dummy_vram = DummyVRAM()
    dummy_state = DummyState()
    routes.set_managers(dummy_vram, dummy_state)

    forwarded: list[tuple[int, str, bytes]] = []

    async def fake_forward_request(request, port, path, body):  # noqa: ANN001
        forwarded.append((port, path, body))
        return routes.Response(content=b"ok", media_type="application/json")

    monkeypatch.setattr(routes, "forward_request", fake_forward_request)

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", b"wav-data", "audio/wav")},
        data={"model": "whisper-hf"},
    )

    assert response.status_code == 200
    assert b"/tmp/whisper-hf" in forwarded[0][2]
    assert b'name="model"\r\n\r\n/tmp/whisper-hf' in forwarded[0][2]
