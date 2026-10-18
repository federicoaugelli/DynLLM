"""In-process Laya decision-model backend.

Laya is a small (421M / 322M) non-autoregressive System One model: it answers
typed ``choice`` / ``score`` / ``noul`` questions about a state instead of
generating text.  The model is loaded directly inside DynLLM through the
``laya`` Python package, so no ``laya-serve`` process, port or binary is
involved.

The package is imported lazily and the Router itself is lazy by default: the
first ``/v1/systemone`` request downloads/builds the checkpoint selected by
Laya's language router.  Set ``laya_preload: true`` to pay that cost during
``load`` instead.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dynllm.backends.base import InProcessBackend
from dynllm.backends.gpu import empty_torch_cache
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Install torch for your hardware first (DynLLM never installs or\n"
    "replaces it), then run:\n"
    "  uv pip install 'laya>=0.4,<0.5'\n"
    "With auto_install: true DynLLM adds Laya automatically once torch is\n"
    "already present."
)


class LayaBackend(InProcessBackend):
    """Run Laya's :class:`laya.Router` inside the DynLLM process."""

    def __init__(self) -> None:
        self._router: Any | None = None
        self._model: ModelConfig | None = None
        self._executor: ThreadPoolExecutor | None = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.laya

    @property
    def loaded(self) -> bool:
        return self._router is not None

    @property
    def ready_timeout_seconds(self) -> float:
        # The Router is lazy; this budget only matters when
        # ``laya_preload: true`` makes the first checkpoint download happen
        # during load.
        return 900.0

    async def _load(self, model: ModelConfig) -> None:
        try:
            from laya import Router
        except Exception as exc:
            raise RuntimeError(
                "The 'laya' package (and its torch/transformers dependencies) "
                f"could not be imported: {exc}\n{_INSTALL_HINT}"
            ) from exc

        kwargs: dict[str, Any] = {"max_loaded": model.laya_max_loaded}
        if model.device != "auto":
            kwargs["device"] = model.device
        if model.laya_default_model:
            kwargs["default"] = model.laya_default_model

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya")
        logger.info(
            "Initialising Laya Router (device=%s, max_loaded=%d, lazy=%s) …",
            model.device,
            model.laya_max_loaded,
            not model.laya_preload,
        )
        router = None
        try:
            router = Router(**kwargs)
            if model.laya_preload:
                names = model.laya_preload_models or None
                await self._run(router.preload, names)
        except Exception as exc:
            if router is not None:
                try:
                    await self._run(router.unload)
                except Exception:
                    logger.debug("Laya cleanup after failed load failed", exc_info=True)
            self._shutdown_executor()
            raise RuntimeError(f"Failed to initialise the Laya Router: {exc}") from exc

        self._router = router
        self._model = model
        logger.info("Laya Router ready for model '%s'", model.name)

    async def _unload(self) -> None:
        router, self._router = self._router, None
        self._model = None
        if router is not None:
            logger.info("Unloading Laya Router …")
            try:
                await self._run(router.unload)
            except Exception:
                logger.warning("Error while unloading the Laya Router", exc_info=True)
            await self._run(empty_torch_cache)
        self._shutdown_executor()
        if router is not None:
            logger.info("Laya Router unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def predict(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        model: str | None = None,
        **controls: Any,
    ) -> dict[str, Any]:
        """Answer typed questions, loading the routed checkpoint on first use.

        ``controls`` are the per-request knobs Laya's HTTP API also accepts
        (``task``, ``lang``, ``lang_guess``, ``max_len``, ``head_max_len``,
        ``min_confidence``).  Unknown keys are ignored so a Jev-compatible
        client can send extra fields without breaking.
        """
        router = self._router
        if router is None:
            raise RuntimeError("Laya model not loaded")

        allowed = {
            "task",
            "lang",
            "lang_guess",
            "max_len",
            "head_max_len",
            "min_confidence",
        }
        kwargs = {
            key: value
            for key, value in controls.items()
            if key in allowed and value is not None
        }
        if model:
            kwargs["model"] = model

        return await self._run(lambda: router.predict(state, questions, **kwargs))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(self, fn, *args, **kwargs):
        executor = self._executor
        if executor is None:
            raise RuntimeError("Laya backend is not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: fn(*args, **kwargs),
        )

    def _shutdown_executor(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
