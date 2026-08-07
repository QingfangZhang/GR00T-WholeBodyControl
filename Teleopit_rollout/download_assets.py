#!/usr/bin/env python3
"""Download and verify the pinned Teleopit v0.5.0 rollout assets.

Only Python's standard library is used so this script can run before the
rollout environment has been created.  Downloads are installed atomically,
and the robot archive is extracted without accepting links, devices, absolute
paths, or parent-directory traversal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import BinaryIO
import urllib.error
import urllib.request


TELEOPIT_VERSION = "v0.5.0"
TELEOPIT_SOURCE_COMMIT = "f9263865c581802ad531854b8e547e2403a945f3"
MODEL_REPOSITORY = "12e21/Teleopit-models"
MODEL_REPOSITORY_VERSION = "v0.5.0"
MODEL_REPOSITORY_COMMIT = "94cf996444fea6894b87c28e86606cd4c2f1408f"

TRACK_G1_URL = (
    "https://huggingface.co/12e21/Teleopit-models/resolve/v0.5.0/"
    "checkpoints/track_g1.onnx"
)
TRACK_G1_SHA256 = (
    "1ebd341d9193e1c49a986450f6043ba1a9473ad46636ce0bcb1c7755c856e0de"
)
ROBOT_ASSETS_URL = (
    "https://huggingface.co/12e21/Teleopit-models/resolve/v0.5.0/"
    "archives/robot_assets.tar.gz"
)
ROBOT_ASSETS_SHA256 = (
    "fb5f1aeec3c57be6b26533c9a9aad0d095048a5fe8aa3c6915d8f6a67c35d4fc"
)
ROBOT_XML_SHA256 = (
    "512ccfe8b811e2bfaec0c2fc57960941371e189b365db5c1ff874474166800a3"
)

CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_EXTRACTED_BYTES = 8 * 1024**3
EXTRACTION_MARKER = ".teleopit_rollout_asset_root.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := source.read(CHUNK_BYTES):
        destination.write(chunk)
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def ensure_download(
    *, url: str, destination: Path, expected_sha256: str, offline: bool
) -> tuple[str, int, bool]:
    """Return ``(sha256, size, downloaded_now)`` for one verified asset."""

    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return actual, destination.stat().st_size, False
        if offline:
            raise RuntimeError(
                f"Offline mode: {destination} has SHA-256 {actual}, expected "
                f"{expected_sha256}."
            )
    elif offline:
        raise RuntimeError(f"Offline mode: required asset is missing: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Teleopit-rollout-assets/0.5.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    actual, byte_count = _copy_and_hash(response, output)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            output.flush()
            os.fsync(output.fileno())

        if actual != expected_sha256:
            raise RuntimeError(
                f"Refusing {url}: SHA-256 {actual}, expected {expected_sha256}."
            )
        os.replace(temporary_path, destination)
        return actual, byte_count, True
    finally:
        temporary_path.unlink(missing_ok=True)


def _safe_member_path(member_name: str) -> PurePosixPath:
    # Tar paths are POSIX paths even when this script runs on another platform.
    path = PurePosixPath(member_name)
    if path.is_absolute() or not path.parts:
        raise RuntimeError(f"Unsafe archive path: {member_name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"Unsafe archive path: {member_name!r}")
    return path


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only into an initially empty path."""

    destination.mkdir(parents=True, exist_ok=False)
    seen: set[PurePosixPath] = set()
    total_size = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise RuntimeError(
                f"Archive has {len(members)} entries; limit is {MAX_ARCHIVE_MEMBERS}."
            )
        for member in members:
            # A leading archive-root directory entry ("." or "./") carries
            # no data and is common in archives produced by ``tar -C ... .``.
            if member.isdir() and member.name.rstrip("/") in ("", "."):
                continue
            relative = _safe_member_path(member.name)
            if relative in seen:
                raise RuntimeError(f"Duplicate archive path: {member.name!r}")
            seen.add(relative)

            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    f"Unsupported archive entry type for {member.name!r}"
                )

            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            total_size += member.size
            if total_size > MAX_EXTRACTED_BYTES:
                raise RuntimeError(
                    "Archive expands beyond the configured 8 GiB safety limit."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read archive member {member.name!r}")
            with extracted, target.open("xb") as output:
                shutil.copyfileobj(extracted, output, length=CHUNK_BYTES)
            target.chmod(0o644)


def find_robot_xml(extraction_root: Path) -> Path:
    candidates = sorted(
        path
        for path in extraction_root.rglob("g1_29dof.xml")
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        choices = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "The robot archive contains multiple g1_29dof.xml files; refusing "
            f"to guess:\n  {choices}"
        )

    xml_names = sorted(
        str(path.relative_to(extraction_root))
        for path in extraction_root.rglob("*.xml")
        if path.is_file()
    )
    available = "\n  ".join(xml_names[:50]) or "(none)"
    raise RuntimeError(
        "The verified robot archive did not contain the expected official "
        f"g1_29dof.xml. XML files found:\n  {available}"
    )


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def write_manifest(
    *,
    assets_dir: Path,
    model_path: Path,
    model_sha256: str,
    model_bytes: int,
    archive_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    extraction_root: Path,
    robot_xml: Path,
) -> Path:
    manifest_path = assets_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "teleopit": {
            "version": TELEOPIT_VERSION,
            "source_commit": TELEOPIT_SOURCE_COMMIT,
            "model_repository": MODEL_REPOSITORY,
            "model_repository_version": MODEL_REPOSITORY_VERSION,
            "model_repository_commit": MODEL_REPOSITORY_COMMIT,
        },
        "assets": {
            "track_g1_onnx": {
                "path": _relative(model_path, assets_dir),
                "url": TRACK_G1_URL,
                "sha256": model_sha256,
                "expected_sha256": TRACK_G1_SHA256,
                "bytes": model_bytes,
            },
            "robot_assets_archive": {
                "path": _relative(archive_path, assets_dir),
                "url": ROBOT_ASSETS_URL,
                "sha256": archive_sha256,
                "expected_sha256": ROBOT_ASSETS_SHA256,
                "bytes": archive_bytes,
            },
            "robot_assets_root": _relative(extraction_root, assets_dir),
            "robot_xml": {
                "path": _relative(robot_xml, assets_dir),
                "sha256": sha256_file(robot_xml),
                "bytes": robot_xml.stat().st_size,
            },
        },
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".json", dir=assets_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=script_dir / "assets",
        help="destination directory (default: Teleopit_rollout/assets)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify already-downloaded files without network access",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir.expanduser().resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    model_path = assets_dir / "checkpoints" / "track_g1.onnx"
    archive_path = assets_dir / "downloads" / "robot_assets.tar.gz"
    extraction_root = assets_dir / "robot_assets"

    model_sha, model_bytes, model_downloaded = ensure_download(
        url=TRACK_G1_URL,
        destination=model_path,
        expected_sha256=TRACK_G1_SHA256,
        offline=args.offline,
    )
    archive_sha, archive_bytes, archive_downloaded = ensure_download(
        url=ROBOT_ASSETS_URL,
        destination=archive_path,
        expected_sha256=ROBOT_ASSETS_SHA256,
        offline=args.offline,
    )

    # Re-extract from the verified archive on every run.  This makes an
    # offline rerun repair (rather than silently trust) modified mesh/XML
    # files left in the extraction directory.
    with tempfile.TemporaryDirectory(
        prefix=".robot_assets.", dir=assets_dir
    ) as temporary_directory:
        staged_root = Path(temporary_directory) / "contents"
        safe_extract_tar(archive_path, staged_root)
        # Validate the exact controller XML before replacing an earlier tree.
        staged_xml = find_robot_xml(staged_root)
        staged_xml_sha = sha256_file(staged_xml)
        if staged_xml_sha != ROBOT_XML_SHA256:
            raise RuntimeError(
                "The verified archive contains an unexpected g1_29dof.xml: "
                f"SHA-256 {staged_xml_sha}, expected {ROBOT_XML_SHA256}."
            )
        (staged_root / EXTRACTION_MARKER).write_text(
            json.dumps(
                {
                    "owner": "Teleopit_rollout/download_assets.py",
                    "archive_sha256": ROBOT_ASSETS_SHA256,
                    "robot_xml_sha256": ROBOT_XML_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        backup_root: Path | None = None
        if extraction_root.exists():
            marker = extraction_root / EXTRACTION_MARKER
            if not marker.is_file():
                # Accept the marker-less tree produced by the earlier version
                # only if its pinned XML proves it is the same owned asset.
                try:
                    legacy_xml = find_robot_xml(extraction_root)
                    legacy_owned = sha256_file(legacy_xml) == ROBOT_XML_SHA256
                except RuntimeError:
                    legacy_owned = False
                if not legacy_owned:
                    raise RuntimeError(
                        f"Refusing to replace unowned directory {extraction_root}; "
                        f"expected marker {EXTRACTION_MARKER}."
                    )
            backup_root = Path(
                tempfile.mkdtemp(prefix=".robot_assets.backup.", dir=assets_dir)
            )
            backup_root.rmdir()
            os.replace(extraction_root, backup_root)
        try:
            os.replace(staged_root, extraction_root)
        except Exception:
            if backup_root is not None and not extraction_root.exists():
                os.replace(backup_root, extraction_root)
            raise
        if backup_root is not None:
            shutil.rmtree(backup_root)

    robot_xml = find_robot_xml(extraction_root)
    robot_xml_sha = sha256_file(robot_xml)
    if robot_xml_sha != ROBOT_XML_SHA256:
        raise RuntimeError(
            "The verified archive contains an unexpected g1_29dof.xml: "
            f"SHA-256 {robot_xml_sha}, expected {ROBOT_XML_SHA256}."
        )
    manifest_path = write_manifest(
        assets_dir=assets_dir,
        model_path=model_path,
        model_sha256=model_sha,
        model_bytes=model_bytes,
        archive_path=archive_path,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        extraction_root=extraction_root,
        robot_xml=robot_xml,
    )

    print(f"Teleopit {TELEOPIT_VERSION} assets are ready:")
    print(f"  model:   {model_path}")
    print(f"  robot:   {robot_xml}")
    print(f"  manifest:{manifest_path}")
    if not model_downloaded and not archive_downloaded:
        print("  status:  verified existing downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
