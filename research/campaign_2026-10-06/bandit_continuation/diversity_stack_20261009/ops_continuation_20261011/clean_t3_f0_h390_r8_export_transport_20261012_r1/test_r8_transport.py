from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import build_r8_transport


class R8TransportTests(unittest.TestCase):
    def _import_kernel(self, path: Path):
        spec = importlib.util.spec_from_file_location("r8_kernel_under_test", path)
        if spec is None or spec.loader is None:
            raise AssertionError("could not import generated R8 kernel")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_single_file_refuses_pending_before_materializing_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pending"
            receipt = Path(temporary) / "pending_receipt.json"
            result = build_r8_transport.build(Path.cwd(), output, None, receipt)
            self.assertEqual(result["status"], "pending_root_release")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"kernel.py", "kernel-metadata.json"},
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], build_r8_transport.R8_ID)
            self.assertEqual(metadata["title"], build_r8_transport.R8_TITLE)
            self.assertEqual(metadata["dataset_sources"], result["r7_dataset_sources"])
            self.assertEqual(metadata["kernel_sources"], result["r7_kernel_sources"])
            self.assertTrue(receipt.is_file())
            kernel = self._import_kernel(output / "kernel.py")
            control_dir = Path(temporary) / "synthetic_working" / "control"
            control_dir.parent.mkdir()
            self.assertEqual(list(control_dir.parent.iterdir()), [])
            with patch.object(kernel.runpy, "run_path") as run_path:
                with self.assertRaisesRegex(PermissionError, "pending"):
                    kernel.run_embedded(control_dir)
                run_path.assert_not_called()
            self.assertFalse(control_dir.exists())

    def test_admitted_transport_receipt_materializes_exact_controls_then_intercepts(self) -> None:
        source_root = Path.cwd()
        r7_kernel = source_root / build_r8_transport.R7_PACKET
        bindings = json.loads((r7_kernel / "R7_EXPORT_BINDINGS_R1.json").read_text())
        pending = json.loads((r7_kernel / "ROOT_EXPORT_RELEASE_R7.json").read_text())
        admitted = dict(pending)
        admitted["status"] = "admitted"
        admitted["authorized"] = True
        admitted["binding"] = bindings["root_release_binding"]
        admitted["logical_recipe_target_kernel_id"] = bindings["root_release_binding"][
            "target_kernel_id"
        ]
        admitted["transport_target_kernel_id"] = build_r8_transport.R8_ID
        with tempfile.TemporaryDirectory() as temporary:
            root_file = Path(temporary) / "synthetic_root_release.json"
            root_file.write_text(json.dumps(admitted, sort_keys=True), encoding="utf-8")
            generated = Path(temporary) / "admitted_kernel"
            build_receipt = Path(temporary) / "admitted_build_receipt.json"
            build_r8_transport.build(source_root, generated, root_file, build_receipt)
            self.assertEqual(
                {path.name for path in generated.iterdir()},
                {"kernel.py", "kernel-metadata.json"},
            )
            kernel = self._import_kernel(generated / "kernel.py")
            control_dir = Path(temporary) / "working" / "control"
            control_dir.parent.mkdir()
            with patch.object(kernel.runpy, "run_path") as run_path:
                kernel.run_embedded(control_dir)
            expected = {
                "kernel.py": (r7_kernel / "kernel.py").read_bytes(),
                "R7_EXPORT_BINDINGS_R1.json": (
                    r7_kernel / "R7_EXPORT_BINDINGS_R1.json"
                ).read_bytes(),
                "ROOT_EXPORT_RELEASE_R7.json": root_file.read_bytes(),
            }
            self.assertEqual({path.name for path in control_dir.iterdir()}, set(expected))
            for name, content in expected.items():
                self.assertEqual((control_dir / name).read_bytes(), content)
            run_path.assert_called_once_with(str(control_dir / "kernel.py"), run_name="__main__")

    def test_bad_transport_target_is_rejected_by_builder(self) -> None:
        source_root = Path.cwd()
        r7_kernel = source_root / build_r8_transport.R7_PACKET
        bindings = json.loads((r7_kernel / "R7_EXPORT_BINDINGS_R1.json").read_text())
        pending = json.loads((r7_kernel / "ROOT_EXPORT_RELEASE_R7.json").read_text())
        pending.update(
            status="admitted",
            authorized=True,
            binding=bindings["root_release_binding"],
            logical_recipe_target_kernel_id=bindings["root_release_binding"]["target_kernel_id"],
            transport_target_kernel_id="vladkochetov07/incorrect-target",
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "incorrect.json"
            receipt.write_text(json.dumps(pending), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                build_r8_transport.build(
                    source_root,
                    Path(temporary) / "out",
                    receipt,
                    Path(temporary) / "out_receipt.json",
                )
            self.assertFalse((Path(temporary) / "out").exists())


if __name__ == "__main__":
    unittest.main()
