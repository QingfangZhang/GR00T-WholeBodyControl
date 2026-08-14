from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import mujoco

from controller_replacement.provenance import (
    artifact_hashes,
    compiled_mujoco_model_fingerprint,
    snapshot_symlink_targets,
    snapshot_xml_hashes,
    source_tree_hashes,
)


class ProvenanceTest(unittest.TestCase):
    def test_compiled_model_fingerprint_is_repeatable(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint type='free'/>"
            "<geom type='sphere' size='.1'/></body></worldbody></mujoco>"
        )
        first = compiled_mujoco_model_fingerprint(model)
        second = compiled_mujoco_model_fingerprint(model)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["size_bytes"], second["size_bytes"])

    def test_xml_links_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            xml = snapshot / "scene.xml"
            xml.write_text("<mujoco/>", encoding="utf-8")
            target = root / "assets"
            target.mkdir()
            (snapshot / "assets").symlink_to(target, target_is_directory=True)
            self.assertEqual(
                snapshot_xml_hashes(snapshot)["scene.xml"],
                hashlib.sha256(xml.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                snapshot_symlink_targets(snapshot)["assets"], str(target.resolve())
            )
            self.assertEqual(
                artifact_hashes(snapshot, ["scene.xml"])["scene.xml"],
                hashlib.sha256(xml.read_bytes()).hexdigest(),
            )

    def test_source_hashes_prune_runtime_data_and_virtual_environments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            for excluded in ("data", ".venv", "models", "__pycache__"):
                target = root / excluded
                target.mkdir()
                (target / "hidden.py").write_text("VALUE = 2\n", encoding="utf-8")
            hashes = source_tree_hashes(root)
            self.assertEqual(set(hashes), {"module.py"})


if __name__ == "__main__":
    unittest.main()
