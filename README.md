# DynLLM

Agnostic OpenAI-compatible proxy for dynamic model loading and unloading.

DynLLM sits between your OpenAI-compatible client (OpenWebUI, LangChain, curl, …) and
your local inference backends (llama.cpp, OpenVINO Model Server, and/or Hugging Face transformers). It automatically
loads models on demand, tracks VRAM usage, evicts models when memory is tight, and
unloads idle models after a configurable timeout.

---

## Features

- **OpenAI-compatible API** – `/v1/chat/completions`, `/v1/completions`, `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech`, `/v1/models`
- **Dynamic loading** – models are started on first request and stopped when idle
- **VRAM budgeting** – LIFO eviction keeps total GPU memory within a configured limit
- **Multi-backend** – supports llama.cpp (GGUF), OpenVINO Model Server (IR), and `transformers serve`
- **Per-model idle timeout** – override the global timeout per model, or set `inf`/`-1` to never auto-unload
- **Startup preloading** – specify models to load when DynLLM starts
- **Safe mid-generation** – active inference requests are never interrupted by eviction
- **Persistent state** – SQLite database survives restarts; stale states are healed on startup
- **systemd integration** – ships with a ready-made unit file and installer script

---

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- At least one backend installed:
  - **llama.cpp** – build `llama-server` from [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  - **OpenVINO Model Server** – see the [OVMS installation guide](https://docs.openvino.ai/2024/ovms_docs_deploying_server.html)
  - **Hugging Face transformers** – install `transformers[serving]`; for Intel GPUs install torch XPU wheels first

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/youruser/DynLLM
cd DynLLM
uv sync

# 2. Create a config
cp config.example.yaml config.yaml
# Edit config.yaml – set total_vram_mb, add your models

# 3. Run
uv run dynllm
# or with an explicit config path:
uv run dynllm --config /path/to/config.yaml
```

The proxy starts on `http://0.0.0.0:8000` by default.

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust as needed.

### Top-level settings

| Key | Default | Description |
|---|---|---|
| `server.host` | `0.0.0.0` | Bind address |
| `server.port` | `8000` | Listen port |
| `total_vram_mb` | `8192` | VRAM budget in MB; eviction fires when exceeded |
| `idle_timeout_seconds` | `300` | Global idle auto-unload timeout (seconds) |
| `enabled_backends` | `[llamacpp, openvino]` | Active backends |
| `models_dir` | — | Optional base dir for relative model paths |
| `db_path` | `dynllm_state.db` | SQLite state database path |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |
| `preload_models` | `[]` | Model names to load on startup |

### Backend settings (`backend:`)

| Key | Default | Description |
|---|---|---|
| `llamacpp_binary` | `llama-server` | Path or name of the llama-server binary |
| `ovms_binary` | `ovms` | Path or name of the OVMS binary |
| `transformers_binary` | `transformers` | Path or name of the Hugging Face transformers CLI |
| `port_range_start` | `9100` | Start of port range for backend subprocesses |
| `port_range_end` | `9200` | End of port range for backend subprocesses |

### Model declaration fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique model ID; used as the `model` field in API requests |
| `path` | yes | Path to the `.gguf` file, OpenVINO IR directory, or local Hugging Face model directory |
| `backend` | yes | `llamacpp`, `openvino`, or `transformers` |
| `model_type` | no | `llm`, `transcription`, or `speech`. Default: `llm` |
| `vram_mb` | yes | Estimated VRAM in MB when loaded (used for eviction math) |
| `target_device` | no | OpenVINO target device (`CPU`, `GPU`, `NPU`). Default: `CPU` |
| `n_gpu_layers` | no | llama.cpp only – GPU layers (`-1` = all). Default: `-1` |
| `context_size` | no | llama.cpp only – context window size. Default: `4096` |
| `ovms_shape` | no | OpenVINO only – shape hint (e.g. `"auto"`) |
| `device` | no | transformers only – execution device (`auto`, `cpu`, `cuda`, `xpu`) |
| `dtype` | no | transformers only – load dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `unload_time` | no | Per-model idle timeout (seconds). Overrides `idle_timeout_seconds`. Use `-1` or `inf` to never auto-unload |

### Example config

```yaml
server:
  host: "0.0.0.0"
  port: 8000

total_vram_mb: 7500
idle_timeout_seconds: 300

enabled_backends:
  - llamacpp
  - openvino
  - transformers

backend:
  llamacpp_binary: "llama-server"
  ovms_binary: "ovms"
  transformers_binary: "transformers"
  port_range_start: 9100
  port_range_end: 9200

# Load this model immediately at startup
preload_models:
  - "llama3-8b-q4"

models:
  - name: "llama3-8b-q4"
    path: "/mnt/models/gguf/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    backend: llamacpp
    model_type: llm
    vram_mb: 5500
    n_gpu_layers: -1
    context_size: 4096

  - name: "phi3-mini-ov"
    path: "/mnt/models/openvino/phi-3-mini-4k-instruct-ov"
    backend: openvino
    model_type: llm
    target_device: GPU
    vram_mb: 4096
    ovms_shape: "auto"
    unload_time: -1   # keep this model loaded permanently

  - name: "whisper-large-v3-ov"
    path: "/mnt/models/openvino/whisper-large-v3-ov"
    backend: openvino
    model_type: transcription
    target_device: CPU
    vram_mb: 0

  - name: "speecht5-ov"
    path: "/mnt/models/openvino/speecht5-ov"
    backend: openvino
    model_type: speech
    target_device: CPU
    vram_mb: 0

  - name: "qwen25-3b-hf"
    path: "/mnt/models/huggingface/Qwen2.5-3B-Instruct"
    backend: transformers
    model_type: llm
    device: xpu
    dtype: bfloat16
    vram_mb: 6500
```

---

## API Reference

DynLLM exposes a standard OpenAI-compatible REST API. Point any OpenAI client at
`http://<host>:<port>` and use the model `name` values from your config.

### `GET /v1/models`

Returns the list of configured models in OpenAI format.

```json
{
  "object": "list",
  "data": [
    { "id": "llama3-8b-q4", "object": "model", "created": 1700000000, "owned_by": "dynllm" }
  ]
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Supports streaming (`"stream": true`).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### `POST /v1/completions`

OpenAI-compatible text completions. Supports streaming.

### `POST /v1/audio/transcriptions`

OpenAI-compatible speech-to-text endpoint. DynLLM accepts the standard multipart request and proxies it to OVMS or `transformers serve`, depending on the configured backend.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "model=whisper-large-v3-ov" \
  -F "file=@speech.wav"
```

### `POST /v1/audio/translations`

OpenAI-compatible speech translation endpoint. This uses the same OpenVINO transcription models and maps to OVMS `/v3/audio/translations`.

```bash
curl http://localhost:8000/v1/audio/translations \
  -F "model=whisper-large-v3-ov" \
  -F "file=@speech-es.wav"
```

### `POST /v1/audio/speech`

OpenAI-compatible text-to-speech endpoint. OVMS currently returns WAV audio.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"speecht5-ov","input":"Hello from DynLLM"}' \
  -o speech.wav
```

### `GET /admin/models`

Returns detailed internal state for every known model (status, port, PID, VRAM, timestamps).

### `POST /admin/models/unload`

Manually unload a model from VRAM.

```bash
curl -X POST http://localhost:8000/admin/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b-q4"}'
```

---

## How It Works

### Load on demand

When a `/v1/chat/completions`, `/v1/completions`, or `/v1/audio/*` request arrives:
1. DynLLM looks up the model in the config by name.
2. If not loaded: checks whether enough VRAM is free.
3. If not enough VRAM: evicts models in **LIFO order** (most recently loaded first),
   skipping any model currently serving a request.
4. Starts the backend subprocess and waits for it to be ready.
5. Proxies the request to the backend and streams the response back.

### Idle auto-unload

A background scheduler checks loaded models every 30 seconds. Any model idle longer
than its effective timeout (per-model `unload_time` > global `idle_timeout_seconds`)
is stopped and its VRAM is freed. Models with active requests are never evicted.

### Per-model `unload_time`

Set `unload_time: -1` (or `inf`) on a model to keep it permanently in VRAM unless
VRAM pressure forces eviction or you manually unload it via `/admin/models/unload`.

### Startup preloading

List model names under `preload_models:` to have them loaded before the proxy
starts accepting traffic. This reduces first-request latency.

### Startup backend logging

At startup DynLLM logs the available torch execution backends detected in the runtime, for example `['cpu', 'xpu']` or `['cpu', 'cuda']`. This helps verify that the installed torch build matches the hardware backend you expect to use.

### VRAM eviction (LIFO)

When a new model needs to be loaded and there is insufficient free VRAM, DynLLM
evicts the **most recently loaded** model first (Last-In, First-Out). This heuristic
favours keeping the models you have been using longest resident in memory.

Models with in-flight requests are **never** evicted; the load will fail with HTTP 503
if eviction is impossible due to all candidates being busy.

### State persistence

All model state (status, PID, port, VRAM, timestamps) is stored in a SQLite database.
On startup, any models stuck in a `loading` or `unloading` state (e.g. due to a crash)
are automatically reset to `unloaded`.

---

## systemd Deployment

```bash
# Run as root
sudo bash systemd/install.sh
```

The installer:
1. Creates a `dynllm` system user
2. Copies the project to `/opt/dynllm`
3. Runs `uv sync --frozen`
4. Creates a default config at `/opt/dynllm/config.yaml`
5. Installs and enables the systemd unit `dynllm.service`

```bash
sudo systemctl status dynllm
sudo journalctl -u dynllm -f
```

GPU access is granted via `/dev/dri` (Intel/AMD). For NVIDIA, uncomment the relevant
lines in `systemd/dynllm.service`.

---

## Backend Notes

### llama.cpp

- Serves **GGUF** models only.
- One `llama-server` process per loaded model.
- Readiness is detected via `GET /health`.
- Relevant config fields: `n_gpu_layers`, `context_size`.

### OpenVINO Model Server (OVMS)

- Serves **OpenVINO IR** model directories only (not GGUF).
- One `ovms` process per loaded model.
- `model_type: llm` uses the standard single-model OVMS config flow.
- `model_type: transcription` and `model_type: speech` use OVMS audio task mode.
- Audio endpoints require an OVMS release with OpenAI audio support, such as `2025.4+`.
- Current OVMS speech-to-text support is limited to `wav` and `mp3` uploads.
- gRPC is disabled (`--port 0`); only the REST API is used.
- Uses OVMS's OpenAI-compatible `/v3/` endpoints.
- Readiness is detected in two phases:
  1. `/v1/config` reports all model versions as `AVAILABLE`.
  2. A probe request to `/v3/chat/completions` confirms the inference path is live
      (prevents the 404 that OVMS can return immediately after reporting AVAILABLE).

### Hugging Face transformers

- Serves local Hugging Face model directories through `transformers serve`.
- One `transformers serve` process per loaded model.
- DynLLM keeps the public model alias from `config.yaml` and rewrites backend requests to the local model path expected by `transformers serve`.
- Supports `model_type: llm` and `model_type: transcription`.
- Relevant config fields: `device`, `dtype`.
- For Intel GPUs, install torch from the XPU wheel index before installing `transformers[serving]`:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
uv pip install "transformers[serving]"
```

- For CUDA systems, install the CUDA-specific torch wheels first, then `transformers[serving]`.
- DynLLM still applies the same VRAM accounting, LIFO eviction, and idle unload rules used for llama.cpp and OVMS.

---

## License

See [LICENSE](LICENSE).
