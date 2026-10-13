"""Procedencia de una corrida: con qué código, librerías y hardware se produjo.

Se guarda dentro de cada `run.json` para que cualquier número del paper se pueda
rastrear hasta el commit y la máquina que lo generó, y para poder declarar el
entorno experimental sin reconstruirlo de memoria.
"""
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_HERE = Path(__file__).parent.resolve()
_PACKAGES = ("torch", "qpso-torch", "numpy", "scikit-learn", "scipy")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_HERE, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(pkg: str) -> Optional[str]:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def collect_provenance(device: torch.device) -> Dict[str, Any]:
    """Foto del entorno al momento de la corrida. Nunca lanza: lo que no se puede leer queda en None."""
    device = torch.device(device)
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(idx),
            "memory_total_mb": int(torch.cuda.get_device_properties(idx).total_memory // 2**20),
            "cuda": torch.version.cuda,
        }
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        # True = había cambios sin commitear en archivos versionados: el commit NO
        # describe por completo el código que corrió.
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "packages": {pkg: _version(pkg) for pkg in _PACKAGES},
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_threads_torch": torch.get_num_threads(),
        "device": str(device),
        "gpu": gpu,
    }
