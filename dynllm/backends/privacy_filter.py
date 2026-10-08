from __future__ import annotations

import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dynllm.backends.base import InProcessBackend
from dynllm.backends.gpu import empty_torch_cache
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

MASK_TAGS: dict[str, str] = {
    "private_person": "[PERSON]",
    "private_email": "[EMAIL]",
    "private_phone": "[PHONE]",
    "private_address": "[ADDRESS]",
    "private_url": "[URL]",
    "private_date": "[DATE]",
    "account_number": "[ACCOUNT_NUMBER]",
    "secret": "[SECRET]",
}


class PrivacyFilterBackend(InProcessBackend):
    """In-process PII masking backend.

    Loads ``openai/privacy-filter`` (or a local copy) through the Hugging Face
    ``token-classification`` pipeline.  All inference runs in a dedicated
    thread-pool executor.
    """

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.privacy_filter

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    async def _load(self, model: ModelConfig) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for the privacy_filter "
                "backend. Install it with: uv pip install transformers torch"
            ) from exc

        model_path = str(model.path)
        logger.info("Loading privacy filter model '%s' …", model_path)

        def _load_sync() -> None:
            self._pipeline = pipeline(
                "token-classification",
                model=model_path,
                device_map="auto",
            )
            self._pipeline.model.eval()

        try:
            await self._run(_load_sync)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load privacy filter model '{model_path}': {exc}"
            ) from exc
        logger.info("Privacy filter model '%s' loaded successfully", model.name)

    async def _unload(self) -> None:
        if self._pipeline is None:
            return
        logger.info("Unloading privacy filter model …")
        self._pipeline = None
        await self._run(empty_torch_cache)
        logger.info("Privacy filter model unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def filter_text(
        self,
        text: str,
        *,
        mask_strategy: str = "replace",
        categories: list[str] | None = None,
    ) -> dict:
        """Detect and mask PII spans in *text*.

        ``mask_strategy`` is one of ``replace`` (category tag), ``redact``
        (``[REDACTED]``) or ``hash`` (``[REDACTED_<hash>]``).  ``categories``
        limits masking to the given ``entity_group`` values.
        """
        if self._pipeline is None:
            raise RuntimeError("Privacy filter model not loaded")

        spans = await self._run(self._classify_sync, text)
        if categories:
            allowed = set(categories)
            spans = [s for s in spans if s["entity_group"] in allowed]

        return {
            "masked_text": self._apply_mask(text, spans, mask_strategy),
            "spans": spans,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def _classify_sync(self, text: str) -> list[dict]:
        results = self._pipeline(text, aggregation_strategy="simple")
        return [
            {
                "entity_group": r["entity_group"],
                "score": round(float(r["score"]), 6),
                "word": r["word"],
                "start": int(r["start"]),
                "end": int(r["end"]),
            }
            for r in results
        ]

    def _apply_mask(self, text: str, spans: list[dict], strategy: str) -> str:
        if not spans:
            return text
        masked = text
        for span in sorted(spans, key=lambda s: s["start"], reverse=True):
            tag = self._mask_tag(span["entity_group"], strategy)
            masked = masked[: span["start"]] + tag + masked[span["end"] :]
        return masked

    @staticmethod
    def _mask_tag(entity_group: str, strategy: str) -> str:
        if strategy == "redact":
            return "[REDACTED]"
        if strategy == "hash":
            digest = hashlib.sha256(entity_group.encode()).hexdigest()[:8]
            return f"[REDACTED_{digest}]"
        return MASK_TAGS.get(entity_group, "[PII]")
