from __future__ import annotations

import logging
import os
import tempfile

from dynllm.backends.tts.base import TTSEngine

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "M1"


class SupertonicEngine(TTSEngine):
    """TTS engine for Supertonic TTS models."""

    async def load(self) -> None:
        from supertonic import TTS

        logger.info("Loading Supertonic TTS …")
        self._tts = TTS(auto_download=True)
        self._loaded = True
        logger.info("Supertonic TTS loaded")

    async def unload(self) -> None:
        if hasattr(self, "_tts"):
            del self._tts
        self._loaded = False

        import gc

        gc.collect()
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Supertonic TTS unloaded")

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        if not self._loaded:
            raise RuntimeError("Supertonic TTS not loaded")

        voice_name = voice or _DEFAULT_VOICE
        style = self._tts.get_voice_style(voice_name=voice_name)
        wav, _ = self._tts.synthesize(text, voice_style=style, lang="en")

        tmp = tempfile.NamedTemporaryFile(
            suffix=f".{response_format.lower()}", delete=False
        )
        try:
            tmp.close()
            self._tts.save_audio(wav, tmp.name)
            with open(tmp.name, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp.name)
