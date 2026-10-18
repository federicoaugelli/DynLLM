"""
Basic backend availability checker.

Checks whether the configured backend binaries are available on PATH.
If a binary is missing the user is warned but startup is NOT aborted —
they may only use the backends whose binaries are present, or may have
both binaries but not listed certain backends in enabled_backends.

A --install flag is NOT provided because both llama.cpp and OVMS require
custom compilation tuned to the user's hardware.  Instead we print
instructions for the user to follow.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from dynllm.core.builtins import BUILTIN_PACKAGE_SPECS, BUILTIN_REQUIREMENTS
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
    BackendType.transformers: (
        "transformers CLI not found on PATH.\n"
        "  Install the serving extras into your DynLLM environment:\n"
        '    uv pip install "transformers[serving]"\n'
        "  For Intel GPUs, install torch with XPU wheels first:\n"
        "    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu\n"
        "  Then set 'backend.transformers_binary' if the CLI is outside PATH."
    ),
    BackendType.privacy_filter: (
        "privacy_filter backend requires the 'transformers' and 'torch' packages.\n"
        "  Install them into your DynLLM environment:\n"
        "    uv pip install transformers torch\n"
        "  The model will be auto-downloaded from HuggingFace on first load."
    ),
    BackendType.neutronstar: (
        "ns-server not found on PATH.\n"
        "  NeutronStar requires a hardware-tuned build; from the NeutronStar\n"
        "  repo checkout:\n"
        "    cargo build --release --bin ns-server\n"
        "    sudo scripts/deploy.sh   # symlinks into /usr/local/bin\n"
        "  Requires the OpenVINO runtime and Intel GPU compute-runtime.\n"
        "  Or set 'backend.neutronstar_binary' in your config to the full path."
    ),
    BackendType.laya: (
        "The in-process Laya backend requires torch>=2.0 and the 'laya'\n"
        "Python package. DynLLM never installs or replaces torch itself:\n"
        "  1. Install the torch build for your hardware (CPU/XPU/CUDA).\n"
        "  2. Install Laya with:\n"
        "       uv pip install 'laya>=0.4,<0.5'\n"
        "  With auto_install: true DynLLM will add Laya automatically once\n"
        "  torch is present."
    ),
}


def detect_execution_backends() -> list[str]:
    """Return the torch execution backends available in this environment."""
    backends = ["cpu"]
    try:
        import torch
    except Exception:
        return backends

    checks = [
        ("cuda", getattr(torch, "cuda", None)),
        ("xpu", getattr(torch, "xpu", None)),
    ]
    for name, module in checks:
        is_available = getattr(module, "is_available", None)
        if callable(is_available):
            try:
                if is_available():
                    backends.append(name)
            except Exception:
                continue

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = getattr(mps_backend, "is_available", None)
    if callable(mps_available):
        try:
            if mps_available():
                backends.append("mps")
        except Exception:
            pass

    return backends


def _package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch_compatible() -> tuple[bool, str | None]:
    """Return whether an installed torch can be reused by Laya.

    DynLLM deliberately never installs torch as part of ``auto_install``:
    doing so would pull the default CUDA wheel and could replace a custom
    XPU/CUDA build that other backends depend on.  If torch is already there,
    ``uv pip install laya`` leaves it satisfied and only adds the small
    pure-Python dependencies.
    """
    if not _package_available("torch"):
        return False, "torch is not installed"
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return True, None
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return True, None
    if major < 2:
        return False, f"torch {version} is too old (Laya needs torch>=2.0)"
    return True, None


def install_python_package(package_spec: str) -> None:
    """Install *package_spec* into the interpreter running DynLLM.

    Uses ``uv pip --python <sys.executable>`` when uv is available (the project
    is uv-first) and falls back to ``python -m pip`` otherwise.  The caller
    decides whether automatic installation is allowed.
    """
    uv = shutil.which("uv")
    if uv is not None:
        cmd = [uv, "pip", "install", "--python", sys.executable, package_spec]
    else:
        cmd = [sys.executable, "-m", "pip", "install", package_spec]

    logger.info("Installing missing Python package: %s", package_spec)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to install '{package_spec}' (exit {result.returncode}):\n"
            f"{output[-4000:]}"
        )

    importlib.invalidate_caches()
    logger.info("Python package installed: %s", package_spec)


def _ensure_builtin_packages(settings: Settings, backend_type: BackendType) -> None:
    """Check (and optionally install) the Python packages for a built-in backend."""
    builtin_name = next(
        (
            model.builtin
            for model in settings.models
            if model.backend == backend_type and model.builtin
        ),
        backend_type.value,
    )
    requires = BUILTIN_REQUIREMENTS.get(builtin_name or "", ())
    if not requires:
        return

    missing = [package for package in requires if not _package_available(package)]
    if not missing:
        logger.info(
            "Backend '%s' packages found: %s",
            backend_type.value,
            ", ".join(requires),
        )
        return

    hint = _INSTALL_HINTS.get(backend_type, "")
    if not settings.auto_install:
        logger.warning(
            "Backend '%s' missing packages: %s\n%s",
            backend_type.value,
            ", ".join(missing),
            hint,
        )
        return

    # Installing the 'laya' wheel normally would also resolve torch.  On a
    # machine without torch, uv would pull the default CUDA build (several
    # GB) and could clobber a custom XPU build.  Only install when torch is
    # already present and usable; otherwise leave the decision to the user.
    torch_ok, torch_reason = _torch_compatible()
    if not torch_ok:
        logger.warning(
            "Backend '%s' cannot be installed automatically: %s.\n"
            "Install torch for this machine (for example a CPU or XPU wheel), "
            "then restart DynLLM or run:\n"
            "  %s",
            backend_type.value,
            torch_reason,
            hint,
        )
        return

    package_spec = BUILTIN_PACKAGE_SPECS.get(builtin_name, builtin_name)
    try:
        install_python_package(package_spec)
    except Exception as exc:
        logger.warning(
            "Automatic install of '%s' failed: %s\n%s",
            package_spec,
            exc,
            hint,
        )
        return

    still_missing = [package for package in requires if not _package_available(package)]
    if still_missing:
        logger.warning(
            "Backend '%s' still missing packages after install: %s\n%s",
            backend_type.value,
            ", ".join(still_missing),
            hint,
        )
    else:
        logger.info(
            "Backend '%s' packages ready: %s",
            backend_type.value,
            ", ".join(requires),
        )


def check_backends(settings: Settings) -> None:
    """
    Warn if any enabled backend binary is not found.

    Does NOT raise – the proxy can still start; errors will surface
    when a model using the missing backend is first requested.
    """
    binary_map: dict[BackendType, str] = {
        BackendType.llamacpp: settings.backend.llamacpp_binary,
        BackendType.openvino: settings.backend.ovms_binary,
        BackendType.transformers: settings.backend.transformers_binary,
        BackendType.neutronstar: settings.backend.neutronstar_binary,
    }

    for backend_type in settings.enabled_backends:
        if backend_type == BackendType.laya:
            _ensure_builtin_packages(settings, backend_type)
            continue

        if backend_type == BackendType.privacy_filter:
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401

                logger.info(
                    "Backend 'privacy_filter' packages found: transformers, torch"
                )
            except ImportError as exc:
                hint = _INSTALL_HINTS.get(backend_type, "")
                logger.warning(
                    "Backend 'privacy_filter' missing package: %s\n%s",
                    exc.name,
                    hint,
                )
            continue

        binary = binary_map.get(backend_type)
        if binary is None:
            continue

        resolved = shutil.which(binary)
        configured_path = Path(binary).expanduser()
        if resolved is None and configured_path.is_file():
            logger.warning(
                "Backend binary '%s' for '%s' exists but is not executable.",
                configured_path,
                backend_type.value,
            )
            continue

        if resolved is None:
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
                resolved,
            )
