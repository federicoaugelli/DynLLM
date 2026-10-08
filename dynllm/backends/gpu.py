"""Best-effort GPU memory helpers shared by in-process backends."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def empty_torch_cache() -> None:
    """Run garbage collection and release the active torch GPU cache."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("GPU cache cleanup skipped (best-effort)", exc_info=True)
