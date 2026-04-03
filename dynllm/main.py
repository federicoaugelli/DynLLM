"""
DynLLM application entry point.

Usage (via uv):
    uv run dynllm
    uv run dynllm --config /path/to/config.yaml
    uv run python -m dynllm.main
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dynllm.api.routes import router, set_managers
from dynllm.core.config import load_config
from dynllm.core.scheduler import IdleScheduler
from dynllm.core.vram_manager import VRAMManager
from dynllm.db.manager import StateManager
from dynllm.installer import check_backends


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def create_app(config_path: str | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separate from ``run()`` so it can be imported by test fixtures or
    ASGI runners (e.g. ``uvicorn dynllm.main:app``).
    """
    settings = load_config(config_path)
    _configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("DynLLM starting – total VRAM budget: %d MB", settings.total_vram_mb)
    logger.info(
        "Enabled backends: %s",
        [b.value for b in settings.enabled_backends],
    )
    logger.info("Configured models: %s", [m.name for m in settings.models])

    # Check that requested binaries are available (skippable)
    check_backends(settings)

    state_manager = StateManager(settings.db_path)
    vram_manager = VRAMManager(settings, state_manager)
    scheduler = IdleScheduler(
        vram_manager,
        state_manager,
        idle_timeout=settings.idle_timeout_seconds,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        await state_manager.heal_stale_states()
        set_managers(vram_manager, state_manager)

        # Preload models listed in config
        for model_name in settings.preload_models:
            model_cfg = settings.model_by_name(model_name)
            if model_cfg is None:
                logger.warning(
                    "preload_models: model '%s' not found in config, skipping",
                    model_name,
                )
                continue
            logger.info("Preloading model '%s' as requested by config …", model_name)
            try:
                await vram_manager.ensure_loaded(model_cfg)
                await state_manager.touch(model_name)
                logger.info("Preloaded model '%s' successfully", model_name)
            except Exception as exc:
                logger.error("Failed to preload model '%s': %s", model_name, exc)

        scheduler_task = asyncio.create_task(scheduler.run())
        logger.info(
            "DynLLM proxy listening on %s:%d",
            settings.server.host,
            settings.server.port,
        )
        yield
        # Shutdown
        scheduler.stop()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

        # Unload all running models gracefully
        loaded = await state_manager.get_loaded()
        for model_state in loaded:
            logger.info("Shutdown: unloading model '%s'", model_state.name)
            try:
                await vram_manager.unload(model_state.name)
            except Exception as exc:
                logger.warning(
                    "Error unloading '%s' on shutdown: %s", model_state.name, exc
                )

    app = FastAPI(
        title="DynLLM",
        description="Agnostic OpenAI-compatible proxy with dynamic model loading",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Expose settings via dependency injection
    app.state.settings = settings

    return app


def run() -> None:
    """CLI entry point: parse args and start the server."""
    parser = argparse.ArgumentParser(description="DynLLM – dynamic model loading proxy")
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to config.yaml (default: $DYNLLM_CONFIG or ./config.yaml)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override server host from config",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port from config",
    )
    args = parser.parse_args()

    app = create_app(args.config)
    settings = app.state.settings

    host = args.host or settings.server.host
    port = args.port or settings.server.port

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # we configure logging ourselves
    )


if __name__ == "__main__":
    run()
