from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from dynllm.backends.base import Backend
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


class PrivacyFilterBackend(Backend):
    """
    In-process privacy-filter backend.

    Loads ``openai/privacy-filter`` (or a local copy) via the Hugging Face
    ``token-classification`` pipeline inside the proxy process.  All inference
    runs in a dedicated thread-pool executor to keep the event loop responsive.
    """

    def __init__(self) -> None:
        self._pipeline = None  # transformers pipeline
        self._pid: int = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="privacy",
        )

    # ------------------------------------------------------------------
    # Backend ABC
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.privacy_filter

    async def start(self, model: ModelConfig, port: int) -> int:
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "The 'transformers' package is required for the privacy_filter "
                "backend. Install it with: uv pip install transformers torch"
            )

        model_path = str(model.path)

        loop = asyncio.get_running_loop()
        logger.info("Loading privacy filter model '%s' …", model_path)

        def _load() -> None:
            self._pipeline = pipeline(
                "token-classification",
                model=model_path,
                device_map="auto",
            )
            self._pipeline.model.eval()

        try:
            await loop.run_in_executor(self._executor, _load)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load privacy filter model '{model_path}': {exc}"
            ) from exc

        self._pid = id(self._pipeline)
        logger.info("Privacy filter model '%s' loaded successfully", model.name)
        return self._pid

    async def stop(self, pid: int) -> None:
        if self._pipeline is not None:
            logger.info("Unloading privacy filter model …")
            del self._pipeline
            self._pipeline = None
            try:
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except ImportError:
                pass

    async def is_ready(
        self,
        port: int,
        model_name: str = "",
        timeout: float = 60.0,
        model_type: str = "llm",
    ) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    async def filter_text(
        self,
        text: str,
        *,
        mask_strategy: str = "replace",
        categories: list[str] | None = None,
    ) -> dict:
        """
        Detect and optionally mask PII spans in *text*.

        Parameters
        ----------
        text:
            Input text to scan for PII.
        mask_strategy:
            ``"replace"`` – substitute with a category-specific tag (``[EMAIL]`` …).
            ``"redact"``  – substitute with ``[REDACTED]``.
            ``"hash"``    – substitute with ``[REDACTED_<hash>]``.
        categories:
            If given, only report / mask spans whose ``entity_group`` is in this
            list.  ``None`` means all categories.

        Returns
        -------
        A dict with keys ``masked_text`` (str) and ``spans`` (list[dict]).
        """
        if self._pipeline is None:
            raise RuntimeError("Privacy filter model not loaded")

        loop = asyncio.get_running_loop()
        raw_spans = await loop.run_in_executor(
            self._executor,
            self._classify_sync,
            text,
        )

        if categories:
            categories_set = set(categories)
            raw_spans = [s for s in raw_spans if s["entity_group"] in categories_set]

        masked_text = self._apply_mask(text, raw_spans, mask_strategy)

        return {
            "masked_text": masked_text,
            "spans": raw_spans,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_sync(self, text: str) -> list[dict]:
        """Run the HF pipeline (blocking – call from executor)."""
        results = self._pipeline(text, aggregation_strategy="simple")
        spans: list[dict] = []
        for r in results:
            spans.append(
                {
                    "entity_group": r["entity_group"],
                    "score": round(float(r["score"]), 6),
                    "word": r["word"],
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                }
            )
        return spans

    def _apply_mask(self, text: str, spans: list[dict], strategy: str) -> str:
        """Replace detected PII spans with the chosen mask token."""
        if not spans:
            return text

        sorted_spans = sorted(spans, key=lambda s: s["start"], reverse=True)
        masked = text
        for span in sorted_spans:
            tag = self._mask_tag(span["entity_group"], strategy)
            masked = masked[: span["start"]] + tag + masked[span["end"] :]
        return masked

    def _mask_tag(self, entity_group: str, strategy: str) -> str:
        if strategy == "redact":
            return "[REDACTED]"
        if strategy == "hash":
            import hashlib

            h = hashlib.sha256(entity_group.encode()).hexdigest()[:8]
            return f"[REDACTED_{h}]"
        return MASK_TAGS.get(entity_group, "[PII]")
