from __future__ import annotations

import sys
import types

from dynllm import installer
from dynllm.core.config import BackendType, Settings


def _laya_settings(auto_install: bool) -> Settings:
    return Settings.model_validate(
        {"auto_install": auto_install, "models": [{"name": "laya"}]}
    )


def test_install_python_package_uses_uv_with_current_interpreter(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(installer.importlib, "invalidate_caches", lambda: None)

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)

    installer.install_python_package("laya>=0.4,<0.5")

    assert calls == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "laya>=0.4,<0.5",
        ]
    ]


def test_auto_install_runs_when_only_laya_is_missing(monkeypatch) -> None:
    installed: list[str] = []
    monkeypatch.setattr(installer, "_package_available", lambda name: name != "laya")
    monkeypatch.setattr(installer, "_torch_compatible", lambda: (True, None))
    monkeypatch.setattr(installer, "install_python_package", installed.append)

    installer._ensure_builtin_packages(
        _laya_settings(auto_install=True), BackendType.laya
    )

    assert installed == ["laya>=0.4,<0.5"]


def test_auto_install_never_pulls_torch(monkeypatch) -> None:
    installed: list[str] = []
    monkeypatch.setattr(installer, "_package_available", lambda name: False)
    monkeypatch.setattr(
        installer, "_torch_compatible", lambda: (False, "torch is not installed")
    )
    monkeypatch.setattr(installer, "install_python_package", installed.append)

    installer._ensure_builtin_packages(
        _laya_settings(auto_install=True), BackendType.laya
    )

    assert installed == []


def test_auto_install_disabled_only_warns(monkeypatch) -> None:
    installed: list[str] = []
    monkeypatch.setattr(installer, "_package_available", lambda name: False)
    monkeypatch.setattr(installer, "install_python_package", installed.append)

    installer._ensure_builtin_packages(
        _laya_settings(auto_install=False), BackendType.laya
    )

    assert installed == []
