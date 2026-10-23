from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


class NativeIdBoundary(unittest.TestCase):
    """Exercise only the new exact systemd Id guard, before helper or file access."""

    def test_mismatched_native_id_fails_before_metadata_import(self) -> None:
        path = Path(__file__).parent / "ACCEPT_EXPORT_CAPTURE_V3.py"
        spec = importlib.util.spec_from_file_location("root_export_accept_v3_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        observation = {
            "schema": "fg19.first-cell-export-bounded-status.v1",
            "unit": "expected-export.service",
            "invocation_id": "a" * 32,
            "systemd": {
                "Id": "other.service",
                "InvocationID": "a" * 32,
                "ActiveState": "active",
                "SubState": "exited",
                "MainPID": "0",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "Result": "success",
            },
        }
        with (
            patch.object(
                module, "validate_scorer_data_identity", side_effect=AssertionError("import")
            ),
            self.assertRaisesRegex(ValueError, "systemd Id"),
        ):
            module.validate_metadata(observation, {}, {})


if __name__ == "__main__":
    unittest.main()
