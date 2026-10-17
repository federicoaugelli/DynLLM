from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dynllm.backends.neutronstar as neutronstar_module
from dynllm.api import routes
from dynllm.backends import BackendRegistry
from dynllm.backends.neutronstar import NeutronStarBackend
from dynllm.core.config import BackendType, ModelConfig, Settings
from dynllm.main import create_app


def _settings(*backends: BackendType) -> Settings:
    return Settings(enabled_backends=list(backends), models=[])


def _ns_model(tmp_path: Path, name: str = "minicpm-v4.7", **extra) -> ModelConfig:
    workspace = tmp_path / name
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "model.json").write_text(
        json.dumps({"name": name, "gguf": f"{name}.gguf"})
    )
    return ModelConfig(
        name=name,
        path=workspace,
        backend=BackendType.neutronstar,
        vram_mb=12000,
        **extra,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_creates_neutronstar_when_enabled() -> None:
    registry = BackendRegistry(_settings(BackendType.neutronstar))
    backend = registry.create(BackendType.neutronstar)
    assert isinstance(backend, NeutronStarBackend)
    assert backend.backend_type == BackendType.neutronstar


def test_registry_rejects_disabled_backend() -> None:
    registry = BackendRegistry(_settings(BackendType.llamacpp))
    with pytest.raises(RuntimeError, match="not enabled"):
        registry.create(BackendType.neutronstar)


# ---------------------------------------------------------------------------
# Command construction & validation
# ---------------------------------------------------------------------------


def test_command_flags(tmp_path: Path) -> None:
    backend = NeutronStarBackend(binary="ns-server")
    model = _ns_model(tmp_path)

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[0] == "ns-server"
    assert cmd[cmd.index("--model") + 1] == str(model.path)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "9101"
    assert cmd[cmd.index("--alias") + 1] == "minicpm-v4.7"
    assert "--max-seq" not in cmd  # only when ns_max_seq is configured


def test_max_seq_and_extra_args_appended(tmp_path: Path) -> None:
    backend = NeutronStarBackend(binary="ns-server")
    model = _ns_model(tmp_path, ns_max_seq=8192, extra_args=["--foo"])

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--max-seq") + 1] == "8192"
    assert cmd[-1] == "--foo"


def test_validate_rejects_missing_path(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    model = _ns_model(tmp_path)
    model.path = tmp_path / "does-not-exist"

    with pytest.raises(RuntimeError, match="Model path not found"):
        backend._validate(model)


def test_validate_rejects_dir_without_model_json(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    model = _ns_model(tmp_path)
    (model.path / "model.json").unlink()

    with pytest.raises(RuntimeError, match="model.json"):
        backend._validate(model)


def test_validate_accepts_bare_gguf(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    gguf = tmp_path / "merged.gguf"
    gguf.write_bytes(b"GGUF")
    model = ModelConfig(
        name="ornith",
        path=gguf,
        backend=BackendType.neutronstar,
        vram_mb=12000,
    )
    backend._validate(model)  # must not raise


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_build_env_none_without_ns_env(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    assert backend._build_env(_ns_model(tmp_path)) is None


def test_build_env_merges_ns_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEP_ME", "1")
    backend = NeutronStarBackend()
    model = _ns_model(tmp_path, ns_env={"NS3_MAX_SEQ": "8192"})

    env = backend._build_env(model)

    assert env is not None
    assert env["NS3_MAX_SEQ"] == "8192"
    assert env["KEEP_ME"] == "1"


def test_build_env_rejects_ns3_spec(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    model = _ns_model(tmp_path, ns_env={"NS3_SPEC": "1"})

    with pytest.raises(RuntimeError, match="NS3_SPEC"):
        backend._build_env(model)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_ready_timeout_override(tmp_path: Path) -> None:
    backend = NeutronStarBackend()
    assert backend.ready_timeout_seconds == 300.0


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = iter(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, *args, **kwargs):
        return next(self._responses)


@pytest.mark.anyio
async def test_is_ready_succeeds_on_health_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    responses = [
        _FakeResponse(500, {"error": "nope"}),
        _FakeResponse(200, {"status": "ok"}),
    ]

    def factory(*args, **kwargs):
        return _FakeClient(responses)

    monkeypatch.setattr(neutronstar_module.httpx, "AsyncClient", factory)
    backend = NeutronStarBackend()
    model = _ns_model(tmp_path)
    assert await backend.is_ready(9101, model, timeout=5.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_ns_fields_rejected_for_other_backends() -> None:
    with pytest.raises(ValueError, match="ns_max_seq"):
        ModelConfig(
            name="bad",
            path=Path("/models/x.gguf"),
            backend=BackendType.llamacpp,
            vram_mb=1024,
            ns_max_seq=8192,
        )

    with pytest.raises(ValueError, match="ns_env"):
        ModelConfig(
            name="bad",
            path=Path("/models/x.gguf"),
            backend=BackendType.openvino,
            vram_mb=1024,
            ns_env={"NS3_MAX_SEQ": "8192"},
        )


def test_extra_args_allowed_for_neutronstar() -> None:
    model = ModelConfig(
        name="ok",
        path=Path("/models/ws"),
        backend=BackendType.neutronstar,
        vram_mb=1024,
        extra_args=["--max-seq", "16384"],
    )
    assert model.extra_args == ["--max-seq", "16384"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class DummyVRAM:
    def __init__(self) -> None:
        self.loaded_models: list[str] = []

    async def ensure_loaded(self, model):
        self.loaded_models.append(model.name)
        return 9123

    async def increment_active(self, model_name: str) -> None:
        pass

    async def decrement_active(self, model_name: str) -> None:
        pass


class DummyState:
    async def touch(self, model_name: str) -> None:
        pass


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enabled_backends:
  - neutronstar
models:
  - name: minicpm-v4.7
    path: /models/MiniCPM-V-4.7-35B-A3B
    backend: neutronstar
    vram_mb: 12000
""".strip()
    )
    app = create_app(str(config_path))
    app.state.vram_manager = DummyVRAM()
    app.state.state_manager = DummyState()

    forwarded: list[tuple[str, bytes]] = []

    async def fake_forward_request(request, port, path, body):
        forwarded.append((path, body))
        return routes.Response(content=b"{}", media_type="application/json")

    async def fake_forward_streaming_request(request, port, path, body):
        forwarded.append((path, body))
        return routes.Response(content=b"sse", media_type="text/event-stream")

    monkeypatch.setattr(routes, "forward_request", fake_forward_request)
    monkeypatch.setattr(routes, "forward_streaming_request", fake_forward_streaming_request)
    app.state._forwarded = forwarded
    return TestClient(app)


def test_chat_completions_proxied_on_v1(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "minicpm-v4.7", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert client.app.state._forwarded[0][0] == "v1/chat/completions"


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/v1/completions", {"model": "minicpm-v4.7", "prompt": "hi"}),
        ("/v1/embeddings", {"model": "minicpm-v4.7", "input": "hi"}),
        ("/v1/rerank", {"model": "minicpm-v4.7", "query": "q", "documents": ["a"]}),
    ],
)
def test_unsupported_endpoints_rejected(
    client: TestClient, endpoint: str, payload: dict
) -> None:
    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 400
    assert "only exposes /v1/chat/completions" in resp.json()["detail"]
    assert client.app.state._forwarded == []
