#!/usr/bin/env python3
"""Build the qpos-track source-history deploy without editing NVIDIA code.

The configured ``gear_sonic_deploy/build`` tree supplies this host's exact
compile and link options.  Only the official main object is replaced by the
wrapper in ``change_ckpt_track``; official source/object/release artifacts are
hashed before and after and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
BUILD_ROOT = DEPLOY_ROOT / "build"
COMPILE_DATABASE = BUILD_ROOT / "compile_commands.json"
OFFICIAL_SOURCE = (
    DEPLOY_ROOT / "src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
).resolve()
OFFICIAL_BINARY = (DEPLOY_ROOT / "target/release/g1_deploy_onnx_ref").resolve()
WRAPPER_SOURCE = (
    REPO_ROOT
    / "change_ckpt_track/source_history_deploy/g1_deploy_onnx_ref_source_history.cpp"
).resolve()
PRIVATE_BUILD_ROOT = (
    REPO_ROOT / "change_ckpt_track/build/source_history_deploy"
).resolve()
OUTPUT_BINARY = (
    REPO_ROOT / "change_ckpt_track/bin/g1_deploy_onnx_ref_source_history"
).resolve()
LINK_FILE = (
    BUILD_ROOT
    / "src/g1/g1_deploy_onnx_ref/CMakeFiles/g1_deploy_onnx_ref.dir/link.txt"
).resolve()
TRACKED_BUILD_INPUTS = {
    "wrapper_source": WRAPPER_SOURCE,
    "official_source": OFFICIAL_SOURCE,
    "state_logger_header": (
        DEPLOY_ROOT / "src/g1/g1_deploy_onnx_ref/include/state_logger.hpp"
    ).resolve(),
    "policy_parameters_header": (
        DEPLOY_ROOT / "src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    ).resolve(),
    "compile_database": COMPILE_DATABASE.resolve(),
    "link_file": LINK_FILE,
}


class BuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_compile_entry() -> dict[str, Any]:
    if not COMPILE_DATABASE.is_file():
        raise BuildError(
            f"compile database is missing: {COMPILE_DATABASE}\n"
            "Build gear_sonic_deploy once before building this wrapper."
        )
    try:
        entries = json.loads(COMPILE_DATABASE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read compile database: {exc}") from exc
    matches = [
        entry
        for entry in entries
        if Path(entry["file"]).resolve() == OFFICIAL_SOURCE
    ]
    if len(matches) != 1:
        raise BuildError(
            f"expected one compile entry for {OFFICIAL_SOURCE}, found {len(matches)}"
        )
    return matches[0]


def _entry_argv(entry: dict[str, Any]) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    if "command" in entry:
        return shlex.split(entry["command"])
    raise BuildError("compile entry has neither arguments nor command")


def _replace_compile_paths(
    argv: Sequence[str], *, source: Path, output: Path
) -> list[str]:
    result = list(argv)
    try:
        output_index = result.index("-o") + 1
        source_index = result.index("-c") + 1
    except ValueError as exc:
        raise BuildError("unexpected compile command: missing -o or -c") from exc
    if Path(result[source_index]).resolve() != OFFICIAL_SOURCE:
        raise BuildError(f"compile command source changed: {result[source_index]}")
    result[output_index] = str(output)
    result[source_index] = str(source)
    return result


def _link_argv(custom_object: Path, output: Path) -> tuple[list[str], Path]:
    if not LINK_FILE.is_file():
        raise BuildError(f"configured link command is missing: {LINK_FILE}")
    argv = shlex.split(LINK_FILE.read_text(encoding="utf-8").strip())
    suffix = "CMakeFiles/g1_deploy_onnx_ref.dir/src/g1_deploy_onnx_ref.cpp.o"
    matches = [i for i, token in enumerate(argv) if token.endswith(suffix)]
    if len(matches) != 1:
        raise BuildError("could not identify one official main object in link.txt")
    argv[matches[0]] = str(custom_object)
    try:
        argv[argv.index("-o") + 1] = str(output)
    except ValueError as exc:
        raise BuildError("unexpected link command: missing -o") from exc
    return argv, LINK_FILE.parents[2]


def build(*, dry_run: bool = False) -> dict[str, Any]:
    for required in (OFFICIAL_BINARY, *TRACKED_BUILD_INPUTS.values()):
        if not required.is_file():
            raise BuildError(f"required file is missing: {required}")
    official_source_hash = _sha256(OFFICIAL_SOURCE)
    official_binary_hash = _sha256(OFFICIAL_BINARY)

    entry = _load_compile_entry()
    PRIVATE_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_BINARY.parent.mkdir(parents=True, exist_ok=True)
    custom_object = PRIVATE_BUILD_ROOT / "g1_deploy_onnx_ref_source_history.cpp.o"
    compile_argv = _replace_compile_paths(
        _entry_argv(entry), source=WRAPPER_SOURCE, output=custom_object
    )
    with tempfile.NamedTemporaryFile(
        prefix="g1_deploy_onnx_ref_source_history.",
        dir=OUTPUT_BINARY.parent,
        delete=False,
    ) as temporary:
        temporary_output = Path(temporary.name)
    temporary_output.unlink()
    link_argv, link_cwd = _link_argv(custom_object, temporary_output)

    print("[compile]", shlex.join(compile_argv), flush=True)
    print("[link]", shlex.join(link_argv), flush=True)
    if dry_run:
        return {
            "dry_run": True,
            "compile": compile_argv,
            "link": link_argv,
            "output": str(OUTPUT_BINARY),
        }

    try:
        subprocess.run(compile_argv, cwd=Path(entry["directory"]), check=True)
        subprocess.run(link_argv, cwd=link_cwd, check=True)
        temporary_output.chmod(0o755)
        os.replace(temporary_output, OUTPUT_BINARY)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    if _sha256(OFFICIAL_SOURCE) != official_source_hash:
        raise BuildError("official deploy source changed during wrapper build")
    if _sha256(OFFICIAL_BINARY) != official_binary_hash:
        raise BuildError("official deploy binary changed during wrapper build")

    payload = {
        "dry_run": False,
        "output": str(OUTPUT_BINARY),
        "output_sha256": _sha256(OUTPUT_BINARY),
        "wrapper_source": str(WRAPPER_SOURCE),
        "wrapper_source_sha256": _sha256(WRAPPER_SOURCE),
        "official_source": str(OFFICIAL_SOURCE),
        "official_source_sha256": official_source_hash,
        "official_binary": str(OFFICIAL_BINARY),
        "official_binary_sha256": official_binary_hash,
        "compile_database": str(COMPILE_DATABASE),
        "link_file": str(LINK_FILE),
        "tracked_build_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in TRACKED_BUILD_INPUTS.items()
        },
    }
    manifest = PRIVATE_BUILD_ROOT / "build_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        build(dry_run=args.dry_run)
    except (BuildError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
