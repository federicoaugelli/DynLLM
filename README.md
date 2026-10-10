# DynLLM

Agnostic OpenAI-compatible proxy for dynamic model loading and unloading.

DynLLM sits between your OpenAI-compatible client (OpenWebUI, LangChain, curl, …) and
your local inference backends (llama.cpp, OpenVINO Model Server, Hugging Face transformers,
plus in-process TTS engines and a PII privacy filter). It automatically loads models on
demand, tracks VRAM usage, evicts models when memory is tight, and unloads idle models
after a configurable timeout.

---

## Features

- **OpenAI-compatible API** – `/v1/chat/completions`, `/v1/completions`, `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech`, `/v1/images/generations`, `/v1/embeddings`, `/v1/rerank`, `/v1/systemone`, `/v1/models`
- **litellm guardrail API** – `/beta/litellm_basic_guardrail_api` for PII masking via the in-process privacy filter
- **Dynamic loading** – models are started on first request and stopped when idle
- **VRAM budgeting** – LIFO eviction keeps total GPU memory within a configured limit
- **Multi-backend** – supports llama.cpp (GGUF), OpenVINO Model Server (IR), `transformers serve`, in-process TTS engines (Qwen3-TTS, Supertonic), the `privacy_filter` PII-masking backend, and a scaffolded NeutronStar (`ns-server`) backend
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
  - **TTS engines** (optional, `backend: tts`) – install the engine package plus `soundfile`
  - **Privacy filter** (optional, `backend: privacy_filter`) – needs `transformers` and `torch`
  - **NeutronStar** (optional, `backend: neutronstar`) – build `ns-server` from the NeutronStar
    project. The integration is currently a **scaffold**: the backend type and config are wired
    up, but launching `ns-server` is not implemented yet.

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
| `enabled_backends` | `[llamacpp, openvino]` | Active backends (`llamacpp`, `openvino`, `transformers`, `tts`, `privacy_filter`, `neutronstar`) |
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
| `neutronstar_binary` | `ns-server` | Path or name of the NeutronStar `ns-server` binary (scaffold) |
| `port_range_start` | `9100` | Start of port range for backend subprocesses |
| `port_range_end` | `9200` | End of port range for backend subprocesses |

### Model declaration fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique model ID; used as the `model` field in API requests |
| `path` | yes | Path to the `.gguf` file, OpenVINO IR directory, or local Hugging Face model directory |
| `backend` | yes | `llamacpp`, `openvino`, `transformers`, `tts`, `privacy_filter`, or `neutronstar` |
| `model_type` | no | `llm`, `decision`, `transcription`, `speech`, `image_generation`, `embedding`, `rerank`, `classification`, `detection`, `segmentation`, `ocr`. Default: `llm`. Note: `decision` (System One / JEV-compatible) is served by `backend: llamacpp`; `speech` is served by `backend: tts` or `transformers` (not OpenVINO); `classification` is reserved for `backend: privacy_filter` |
| `vram_mb` | yes | Estimated VRAM in MB when loaded (used for eviction math) |
| `target_device` | no | OpenVINO target device (`CPU`, `GPU`, `NPU`). Default: `CPU` |
| `n_gpu_layers` | no | llama.cpp only – GPU layers (`-1` = all). Default: `-1` |
| `context_size` | no | llama.cpp only – context window size. Default: `4096` |
| `mmproj` | no | llama.cpp only – path to a multimodal projector GGUF (`--mmproj`); needed for vision/audio decision models such as Clef |
| `extra_args` | no | llama.cpp only – extra raw `llama-server` arguments as a list (e.g. `["--batch-size", "2048"]`) |
| `ovms_shape` | no | OpenVINO only – shape hint (e.g. `"auto"`) |
| `device` | no | transformers only – execution device (`auto`, `cpu`, `cuda`, `xpu`) |
| `dtype` | no | transformers only – load dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `quantization` | no | transformers only – `none`, `bnb-4bit`, or `bnb-8bit` for bitsandbytes quantization |
| `trust_remote_code` | no | transformers only – allow custom repo code when required |
| `compile_model` | no | transformers only – enable `torch.compile` through `transformers serve` |
| `continuous_batching` | no | transformers only – enable continuous batching for supported LLMs |
| `attn_implementation` | no | transformers only – `auto`, `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flex_attention` |
| `model_timeout` | no | transformers only – backend-side idle timeout in seconds |
| `revision` | no | transformers only – HF revision rendered as `model@revision` |
| `tts_engine` | yes (tts only) | TTS engine name (`qwen`, `supertonic`). Required when `backend: tts` |
| `tool_parser` | no | openvino LLM only – parser for tool-call extraction (`llama3`, `hermes3`, `phi4`, `mistral`, `gptoss`, `qwen3coder`, `devstral`, `lfm2`) |
| `reasoning_parser` | no | openvino LLM only – parser for reasoning-content extraction (`qwen3`, `gptoss`) |
| `enable_tool_guided_generation` | no | openvino LLM only – guide generation to follow the tool-call schema |
| `draft_model` | no | openvino LLM only – path to a smaller OpenVINO IR draft model for speculative decoding |
| `draft_model_vram_mb` | no | openvino LLM only – extra VRAM (MB) used by the draft model; added to `vram_mb` for eviction math |
| `kv_cache_precision` | no | openvino LLM only – KV cache precision (`u8` for 8-bit cache, halves memory) |
| `cache_size` | no | openvino LLM only – fixed KV cache size in GB (default: dynamic) |
| `enable_prefix_caching` | no | openvino LLM only – cache repeated prompt prefixes (default: enabled in OVMS) |
| `max_num_seqs` | no | openvino LLM only – max sequences processed together (default: 256) |
| `max_num_batched_tokens` | no | openvino LLM only – max tokens per scheduler step |
| `dynamic_split_fuse` | no | openvino LLM only – split prefill/decode across batches (default: enabled in OVMS) |
| `model_distribution_policy` | no | openvino LLM only – `TENSOR_PARALLEL` or `PIPELINE_PARALLEL` for multi-device setups |
| `unload_time` | no | Per-model idle timeout (seconds). Overrides `idle_timeout_seconds`. Use `-1`, `inf`, `infinity`, or `never` to never auto-unload |

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
  - tts
  - privacy_filter

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

  # In-process TTS engine (Qwen3-TTS); model is loaded directly inside DynLLM
  - name: "qwen3-tts"
    path: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    backend: tts
    model_type: speech
    tts_engine: qwen
    target_device: xpu
    vram_mb: 3000

  # In-process privacy filter for PII masking (litellm guardrail API)
  - name: "privacy-filter"
    path: "/mnt/models/privacy-filter"   # local copy of openai/privacy-filter
    backend: privacy_filter
    model_type: classification
    vram_mb: 1024
    unload_time: -1                      # keep loaded

  # OpenVINO LLM with speculative decoding and KV cache optimization
  - name: "codellama-7b-sd"
    path: "/mnt/models/openvino/codellama-7b-instruct-ov"
    backend: openvino
    model_type: llm
    target_device: GPU
    vram_mb: 12000
    draft_model: "/mnt/models/openvino/llama-135m-draft-ov"
    draft_model_vram_mb: 500
    kv_cache_precision: u8
    cache_size: 8
    enable_prefix_caching: true
    max_num_seqs: 128

  - name: "qwen25-3b-hf"
    path: "/mnt/models/huggingface/Qwen2.5-3B-Instruct"
    backend: transformers
    model_type: llm
    device: xpu
    dtype: bfloat16
    quantization: bnb-4bit
    attn_implementation: sdpa
    model_timeout: 300
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

OpenAI-compatible speech-to-text endpoint. DynLLM accepts the standard multipart request and proxies it to OVMS or `transformers serve`, depending on the configured backend. On a cold start (model not loaded yet) transient failures are retried automatically so the client does not see a 400/404/503 while the backend is still initialising.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "model=whisper-large-v3-ov" \
  -F "file=@speech.wav"
```

### `POST /v1/audio/translations`

OpenAI-compatible speech translation endpoint. This uses the same OpenVINO transcription models and maps to OVMS `/v3/audio/translations`. Cold-start retries apply as for transcriptions.

```bash
curl http://localhost:8000/v1/audio/translations \
  -F "model=whisper-large-v3-ov" \
  -F "file=@speech-es.wav"
```

### `POST /v1/audio/speech`

OpenAI-compatible text-to-speech endpoint. Served by the in-process `tts` backend (`tts_engine: qwen` / `supertonic`) or, with `transformers`, proxied to `/v1/audio/speech`. The `tts` backend currently returns WAV audio (or the format requested via `response_format`).

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"Hello from DynLLM"}' \
  -o speech.wav
```

### `POST /v1/images/generations`

OpenAI-compatible image generation endpoint. Supported via OVMS with `model_type: image_generation`.

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"sd-xl-ov","prompt":"a cat wearing a hat","n":1,"size":"512x512"}'
```

### `POST /v1/embeddings`

OpenAI-compatible embeddings endpoint. Supported by llama.cpp (GGUF embedding models) and OVMS.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3-gguf","input":"Hello world"}'
```

### `POST /v1/rerank`

Cohere-compatible reranking endpoint. Supported by llama.cpp (GGUF reranker models) and OVMS.

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-v2-gguf","query":"what is AI?","documents":["AI is...","ML is..."]}'
```

### `POST /v1/systemone`

System One decision-model endpoint (JEV-compatible), served by llama.cpp for models with `model_type: decision`. The request sends a `state` and typed `questions`; the backend returns a probability per option in a single scoring pass (no generated tokens).

```bash
curl http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef",
    "state": "Customer message: I was charged twice and nobody replied.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments, refunds", "shipping": "delivery"}
      },
      "angry": {"type": "noul", "instructions": "Is the customer angry?"}
    }
  }'
```

Requires a llama.cpp build with decision-model support (PR #29818, merged 2026-10-02) and a decision-model GGUF. Use official `ggml-org` GGUFs (e.g. `ggml-org/Clef-GGUF`); some community conversions lack the custom `clef` architecture. Add `mmproj` for multimodal models that read images, and raise `--batch-size` via `extra_args` for many candidates.

### `POST /beta/litellm_basic_guardrail_api`

litellm [Generic Guardrail API](https://docs.litellm.ai/docs/proxy/guardrails) contract for PII masking. It accepts texts and returns masked versions through the in-process `privacy_filter` backend. Configure it as a litellm guardrail:

```yaml
# litellm config.yaml
guardrails:
  - guardrail_name: "dynllm-pii-filter"
    litellm_params:
      guardrail: generic_guardrail_api
      mode: pre_call
      api_base: "http://localhost:8000"
      additional_provider_specific_params:
        model: "privacy-filter"      # DynLLM model name
        mask_strategy: "replace"      # replace | redact | hash
        # categories: ["private_email"]   # optional subset of entities
```

The request is answered with `action: NONE` when no PII was found and `action: GUARDRAIL_INTERVENED` (with masked texts) otherwise.

### KServe API (`/v2/models/{name}/...`)

KServe v2 protocol passthrough for OpenVINO models. Supports detection, segmentation, OCR, embedding, and rerank models through OVMS. (`model_type: classification` is reserved for the `privacy_filter` backend.)

```bash
# Model metadata
curl http://localhost:8000/v2/models/resnet50-vm

# Readiness check
curl http://localhost:8000/v2/models/resnet50-vm/ready

# Inference
curl -X POST http://localhost:8000/v2/models/resnet50-vm/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[...]}'
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

When a request arrives for any configured endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/audio/*`, `/v1/images/generations`, `/v1/embeddings`, `/v1/rerank`, `/v1/systemone`, `/v2/models/*`):
1. DynLLM looks up the model in the config by name.
2. If not loaded: checks whether enough VRAM is free.
3. If not enough VRAM: evicts models in **LIFO order** (most recently loaded first),
   skipping any model currently serving a request.
4. Starts the backend (a subprocess for llama.cpp/OVMS/transformers, or an
   in-process engine for `tts` / `privacy_filter`) and waits for it to be ready.
5. Proxies the request to the backend and streams the response back.

### Idle auto-unload

A background scheduler checks loaded models every 30 seconds. Any model idle longer
than its effective timeout (per-model `unload_time` > global `idle_timeout_seconds`)
is stopped and its VRAM is freed. Models with active requests are never evicted.

### Per-model `unload_time`

Set `unload_time: -1` (or `inf`, `infinity`, `never`) on a model to keep it
permanently in VRAM unless VRAM pressure forces eviction or you manually unload
it via `/admin/models/unload`.

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
- Supported `model_type`: `llm`, `decision`, `embedding`, `rerank`.
- `decision` models (System One / JEV-compatible) are exposed at `/v1/systemone`. Requires a llama.cpp build with decision-model support (PR #29818+) and a decision-model GGUF (e.g. `ggml-org/Clef-GGUF`).
- Relevant config fields: `n_gpu_layers`, `context_size`, `mmproj`, `extra_args`.

### OpenVINO Model Server (OVMS)

- Serves **OpenVINO IR** model directories only (not GGUF).
- One `ovms` process per loaded model.
- gRPC is disabled (`--port 0`); only the REST API is used.
- How each `model_type` is started:
  - `llm` – OVMS **task mode** (`--task text_generation`) and exposes the OpenAI-compatible `/v3/` endpoints (`/v3/chat/completions`, `/v3/completions`). Tool-calling, reasoning, speculative decoding, KV-cache and batching options are passed as OVMS flags (see `tool_parser`, `reasoning_parser`, `enable_tool_guided_generation`, `draft_model`, `kv_cache_precision`, `cache_size`, `enable_prefix_caching`, `max_num_seqs`, `max_num_batched_tokens`, `dynamic_split_fuse`, `model_distribution_policy`).
  - `embedding` / `rerank` – standard single-model OVMS config flow (KServe v2 model registration) with the OpenAI-compatible `/v3/embeddings` and `/v1/rerank` endpoints.
  - `transcription` – OVMS **audio task mode** (`--task speech2text`), proxied at `/v3/audio/transcriptions` and `/v3/audio/translations`. Requires OVMS `2025.4+`.
  - `image_generation` – OVMS **image generation task mode** (`--task image_generation`), exposed at `/v3/images/generations` (OpenAI-compatible). Supports Stable Diffusion, SDXL, and FLUX.1 models in OpenVINO IR format.
  - `detection` / `segmentation` / `ocr` – standard OVMS config flow, served through the KServe v2 API (`/v2/models/{name}/infer`).
- `model_type: classification` is reserved for `backend: privacy_filter` and cannot be served via OVMS/KServe (see below).
- `model_type: speech` is **not** supported by OpenVINO – use `backend: tts` (or `transformers`) instead.
- Readiness is detected by model type:
  - Config-flow models (embedding/rerank/CV) and LLM task-mode models: KServe Model Readiness endpoint (`GET /v2/models/{name}/ready`).
  - Audio (`transcription`): a silent WAV probe against `/v3/audio/transcriptions`.
  - `image_generation`: an OPTIONS probe against `/v3/images/generations`.

### TTS (in-process)

- Serves `model_type: speech` with `backend: tts`. The model is loaded **directly in the DynLLM process** (no subprocess) via a `TTSEngine` plugin; synthesis runs in a thread pool.
- Selects the engine with `tts_engine`:
  - `qwen` – Qwen3-TTS (`uv pip install qwen-tts soundfile`). `path` is a Hugging Face model ID or local directory.
  - `supertonic` – Supertonic (`uv pip install supertonic soundfile`). `path` is ignored; models are auto-downloaded.
- `target_device` selects the execution device (e.g. `xpu`, `cpu`).
- Exposed via `POST /v1/audio/speech`.

### Privacy filter (in-process)

- Serves `model_type: classification` with `backend: privacy_filter`. Loads `openai/privacy-filter` (or a local copy) via the Hugging Face `token-classification` pipeline directly inside the proxy process.
- Detects PII entities (`private_person`, `private_email`, `private_phone`, `private_address`, `private_url`, `private_date`, `account_number`, `secret`, …).
- Mask strategies: `replace` (category tag such as `[EMAIL]`), `redact` (`[REDACTED]`), `hash` (`[REDACTED_<hash>]`).
- Consumed by litellm through `POST /beta/litellm_basic_guardrail_api`.
- Requires the `transformers` and `torch` packages in the DynLLM environment.

### Hugging Face transformers

- Serves local Hugging Face model directories through `transformers serve`.
- One `transformers serve` process per loaded model.
- DynLLM keeps the public model alias from `config.yaml` and rewrites backend requests to the local model path expected by `transformers serve`.
- Supports `model_type: llm`, `model_type: transcription`, and `model_type: speech`.
- Relevant config fields: `device`, `dtype`, `quantization`, `trust_remote_code`, `compile_model`, `continuous_batching`, `attn_implementation`, `model_timeout`, `revision`.
- For Intel GPUs, install torch from the XPU wheel index before installing `transformers[serving]`:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
uv pip install "transformers[serving]"
```

- For CUDA systems, install the CUDA-specific torch wheels first, then `transformers[serving]`.
- `quantization: bnb-4bit` and `quantization: bnb-8bit` map to bitsandbytes loading in `transformers serve`.
- In practice, bitsandbytes is most proven on CUDA. On Intel XPU it may work, but treat it as deployment-specific and validate the exact torch + bitsandbytes stack on your target machine before relying on it in production.
- Quantization is enabled only for `model_type: llm` in DynLLM.
- DynLLM still applies the same VRAM accounting, LIFO eviction, and idle unload rules used for llama.cpp and OVMS.

### NeutronStar (scaffold)

- Reserved for NeutronStar `ns-server`, a custom MoE inference engine for Intel Arc GPUs.
- **Scaffold only**: `backend: neutronstar` is validated and registered, but launching `ns-server` is not implemented yet (`start()` raises `NotImplementedError`). It is off by default and will be wired up once the server API is stable.
- Expected shape once enabled: one `ns-server <merged.gguf> --alias <name>` process per model, readiness via `GET /health`, inference proxied to `POST /v1/chat/completions` (OpenAI-compatible, streaming and non-streaming).

---

## License

See [LICENSE](LICENSE).
