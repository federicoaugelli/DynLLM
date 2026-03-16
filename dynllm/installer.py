"""
Basic backend availability checker.

Checks whether the configured backend binaries are available on PATH.
If a binary is missing the user is warned but startup is NOT aborted –
they may only use the backends whose binaries are present, or may have
both binaries but not listed certain backends in enabled_backends.

A --install flag is NOT provided because both llama.cpp and OVMS require
custom compilation tuned to the user's hardware.  Instead we print
instructions for the user to follow.
"""

from __future__ import annotations

import logging
import shutil

from dynllm.core.config import BackendType, Settings

logger = logging.getLogger(__name__)

_INSTALL_HINTS: dict[BackendType, str] = {
    BackendType.llamacpp: (
        "llama-server not found on PATH.\n"
        "  Build llama.cpp from source:\n"
        "    git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp\n"
        "    cmake -B build -DLLAMA_CUDA=ON   # or -DLLAMA_VULKAN=ON etc.\n"
        "    cmake --build build --config Release -j$(nproc)\n"
        "    sudo cp build/bin/llama-server /usr/local/bin/\n"
        "  Or set 'backend.llamacpp_binary' in your config to the full path."
    ),
    BackendType.openvino: (
        "ovms not found on PATH.\n"
        "  Install OpenVINO Model Server:\n"
        "    https://docs.openvino.ai/2024/ovms_docs_deploying_server.html\n"
        "  Build from source:\n"
        "    https://github.com/openvinotoolkit/model_server\n"
        "  Or set 'backend.ovms_binary' in your config to the full path."
    ),
}


def check_backends(settings: Settings) -> None:
    """
    Warn if any enabled backend binary is not found.

    Does NOT raise – the proxy can still start; errors will surface
    when a model using the missing backend is first requested.
    """
    binary_map: dict[BackendType, str] = {
        BackendType.llamacpp: settings.backend.llamacpp_binary,
        BackendType.openvino: settings.backend.ovms_binary,
    }

    for backend_type in settings.enabled_backends:
        binary = binary_map.get(backend_type)
        if binary is None:
            continue

        if shutil.which(binary) is None:
            hint = _INSTALL_HINTS.get(backend_type, "")
            logger.warning(
                "Backend binary '%s' for '%s' not found on PATH.\n%s",
                binary,
                backend_type.value,
                hint,
            )
        else:
            logger.info(
                "Backend '%s' binary found: %s",
                backend_type.value,
                shutil.which(binary),
            )
