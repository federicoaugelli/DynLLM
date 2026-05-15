from __future__ import annotations

import logging
import os
import tempfile

from dynllm.backends.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class QwenTTSEngine(TTSEngine):
    """TTS engine for Qwen3-TTS models (no voice cloning, plain TTS)."""

    async def load(self) -> None:
        from qwen_tts import Qwen3TTSModel

        import torch

        logger.info(
            "Loading Qwen3-TTS model '%s' on device '%s' …",
            self.model_path,
            self.device,
        )
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        self._loaded = True
        logger.info("Qwen3-TTS model loaded")

    async def unload(self) -> None:
        if not self._loaded:
            return
        del self._model
        self._loaded = False
        logger.info("Qwen3-TTS model unloaded")

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        if not self._loaded:
            raise RuntimeError("Qwen3-TTS model not loaded")

        import soundfile as sf

        wavs, sr = self._model.generate(
            text=text,
            language="English",
        )

        tmp = tempfile.NamedTemporaryFile(
            suffix=f".{response_format.lower()}", delete=False
        )
        try:
            tmp.close()
            sf.write(tmp.name, wavs[0], sr)
            with open(tmp.name, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp.name)
