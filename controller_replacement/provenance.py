"""Reproducibility fingerprints for controller-replacement rollouts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import mujoco
import numpy as np

from controller_replacement.output import sha256_file


def _command_output(command: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def git_provenance(repo_root: Path) -> dict[str, Any]:
    """Return commit/branch/dirty metadata without modifying the repository."""

    root = Path(repo_root).resolve()
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _command_output(
        ["git", "branch", "--show-current"], cwd=root
    )
    tracked_status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
    )
    full_status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
    )
    diff = _command_output(
        ["git", "diff", "--no-ext-diff", "--binary", "--"], cwd=root
    )
    tracked_paths = [] if not tracked_status else tracked_status.splitlines()
    all_paths = [] if not full_status else full_status.splitlines()
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(all_paths),
        "status": all_paths,
        "tracked_dirty": bool(tracked_paths),
        "tracked_status": tracked_paths,
        "tracked_diff_sha256": (
            None
            if diff is None
            else hashlib.sha256(diff.encode("utf-8")).hexdigest()
        ),
    }


def runtime_environment() -> dict[str, Any]:
    """Return versions that can affect model or MuJoCo execution."""

    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "mujoco", "onnxruntime"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    onnxruntime_available_providers: list[str] | None = None
    if packages["onnxruntime"] is not None:
        try:
            import onnxruntime as ort

            onnxruntime_available_providers = [
                str(value) for value in ort.get_available_providers()
            ]
        except (ImportError, RuntimeError):
            onnxruntime_available_providers = None
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "onnxruntime_available_providers": onnxruntime_available_providers,
    }


def source_tree_hashes(directory: Path) -> dict[str, str]:
    """Hash behavior-bearing local source files in stable relative order."""

    root = Path(directory).resolve()
    suffixes = {".py", ".sh", ".yaml", ".yml"}
    excluded_directories = {
        ".git",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "assets",
        "bin",
        "build",
        "data",
        "models",
    }
    result: dict[str, str] = {}
    candidates: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name not in excluded_directories
            and not (Path(current) / name).is_symlink()
        )
        for name in files:
            path = Path(current) / name
            if (
                path.suffix.lower() in suffixes
                and path.is_file()
                and not path.is_symlink()
            ):
                candidates.append(path)
    for path in sorted(candidates):
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def snapshot_xml_hashes(snapshot_root: Path) -> dict[str, str]:
    root = Path(snapshot_root).resolve()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*.xml"))
        if path.is_file() and not path.is_symlink()
    }


def snapshot_symlink_targets(snapshot_root: Path) -> dict[str, str]:
    """Describe external asset links without scanning the full 2.9 GB tree."""

    root = Path(snapshot_root).resolve()
    result: dict[str, str] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in (*directories, *files):
            path = base / name
            if path.is_symlink():
                result[path.relative_to(root).as_posix()] = str(path.resolve())
    return dict(sorted(result.items()))


def compiled_mujoco_model_fingerprint(model: mujoco.MjModel) -> dict[str, Any]:
    """Hash the compiled model, including assets actually loaded by MuJoCo.

    The MJB is serialized directly into a temporary memory buffer.  No large
    binary artifact is left on disk.
    """

    size = int(mujoco.mj_sizeModel(model))
    if size <= 0:
        raise ValueError("MuJoCo returned a non-positive compiled model size")
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return {
        "serialization": "MuJoCo MJB memory buffer",
        "mujoco_version": mujoco.__version__,
        "size_bytes": size,
        "sha256": hashlib.sha256(memoryview(buffer)).hexdigest(),
    }


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    root = Path(directory).resolve()
    result: dict[str, str] = {}
    for name in sorted(set(str(item) for item in names)):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256_file(path)
    return result


__all__ = [
    "artifact_hashes",
    "compiled_mujoco_model_fingerprint",
    "git_provenance",
    "runtime_environment",
    "snapshot_symlink_targets",
    "snapshot_xml_hashes",
    "source_tree_hashes",
]
