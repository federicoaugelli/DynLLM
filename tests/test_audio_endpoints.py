from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dynllm.api import routes
from dynllm.backends.tts.base import TTSEngine
from dynllm.main import create_app


class DummyTTSEngine(TTSEngine):
    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def synthesize(self, text, *, voice=None, response_format="wav", speed=1.0):
        return b"fake-audio-data"


class DummyVRAM:
    def __init__(self) -> None:
        self.loaded_models: list[str] = []

    async def ensure_loaded(self, model):
        self.loaded_models.append(model.name)
        return 9123

    def get_backend(self, backend_type):
        from dynllm.backends.tts import TTSBackend

        tts = TTSBackend()
        tts._engine = DummyTTSEngine(model_path="", device="cpu")
        tts._engine._loaded = True
        tts._pid = id(tts._engine)
        return tts


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
  - tts
models:
  - name: whisper-ov
    path: /tmp/whisper-ov
    backend: openvino
    model_type: transcription
    target_device: CPU
    vram_mb: 0
  - name: speech-tts
    path: Qwen/Qwen3-TTS
    backend: tts
    model_type: speech
    tts_engine: qwen
    target_device: cpu
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

    async def fake_forward_request(request, port, path, body):
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


def test_audio_speech_via_tts_backend(client):
    response = client.post(
        "/v1/audio/speech",
        json={"model": "speech-tts", "input": "Hello world"},
    )

    assert response.status_code == 200
    assert client.app.state._dummy_vram.loaded_models == ["speech-tts"]
    assert client.app.state._dummy_state.touched == ["speech-tts"]
    assert response.content == b"fake-audio-data"
    assert response.headers["content-type"] == "audio/wav"


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
