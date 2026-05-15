from dynllm.backends.tts.base import TTSEngine
from dynllm.backends.tts.engines.qwen import QwenTTSEngine
from dynllm.backends.tts.engines.supertonic import SupertonicEngine

ENGINE_REGISTRY: dict[str, type[TTSEngine]] = {
    "qwen": QwenTTSEngine,
    "supertonic": SupertonicEngine,
}

__all__ = [
    "ENGINE_REGISTRY",
    "QwenTTSEngine",
    "SupertonicEngine",
    "TTSEngine",
]
