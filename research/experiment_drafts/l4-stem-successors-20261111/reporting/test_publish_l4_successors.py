"""Small fixtures for the successor publisher's endpoint and no-auth gates."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("publish_l4_successors.py")
SPEC = importlib.util.spec_from_file_location("publish_l4_successors_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


def _baseline() -> dict[str, object]:
    return {
        "sha256": "a" * 64,
        "controls": {
            203301: {"t0": 0.68, "t1": 0.69, "wp": 0.685},
            203302: {"t0": 0.67, "t1": 0.68, "wp": 0.675},
        },
    }


def _cells(complete: bool = True) -> list[dict[str, object]]:
    result = []
    for index, name in enumerate(PUBLISHER.CELLS):
        result.append(
            {
                "cell": name,
                "complete": complete,
                "latest_event": {
                    "epoch": 6,
                    "step": PUBLISHER.UPDATES,
                    "metrics": {
                        "valid/wp/t0": 0.681 + index / 1000,
                        "valid/wp/t1": 0.692 + index / 1000,
                    },
                },
            }
        )
    return result


class SuccessorPublisherTests(unittest.TestCase):
    def test_pinned_historical_readout_loads_two_k16_controls(self) -> None:
        baseline = PUBLISHER._load_baseline()
        self.assertEqual(baseline["sha256"], PUBLISHER.BASELINE_FILE_SHA256)
        self.assertAlmostEqual(baseline["controls"][203301]["wp"], 0.6830194975829329)
        self.assertAlmostEqual(baseline["controls"][203302]["wp"], 0.6826293715900864)

    def test_local_successor_source_packet_is_pinned(self) -> None:
        PUBLISHER._source_identity()

    def test_actual_root_admission_binds_six_profiles_and_configs(self) -> None:
        path = PUBLISHER.ROOT / "ROOT-fit-admission-r1.json"
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        admission = PUBLISHER._admission(path, pin, PUBLISHER.ROOT / "configs")
        self.assertEqual(set(admission["profiles"]), set(PUBLISHER.CELLS))
        self.assertEqual(admission["admission_sha256"], pin)

    def test_pairs_only_after_six_verified_fixed_e6_endpoints(self) -> None:
        cells = _cells()
        rows = PUBLISHER.paired_rows(cells, _baseline())
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(len(rows), 6)
        first = rows[0]
        self.assertAlmostEqual(first["delta_t0"], 0.001)
        self.assertAlmostEqual(first["delta_t1"], 0.002)
        self.assertAlmostEqual(first["delta_wp"], 0.0015)

        cells[-1]["complete"] = False
        self.assertIsNone(PUBLISHER.paired_rows(cells, _baseline()))

    def test_real_candidate_configs_match_the_declared_one_factor_arms(self) -> None:
        for cell in PUBLISHER.CELLS:
            config = json.loads(
                (PUBLISHER.ROOT / "configs" / f"{cell}.json").read_text()
            )
            self.assertTrue(PUBLISHER._expected_config(cell, config), cell)

    def test_dry_run_never_constructs_github_client_or_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            admission_path = Path(temporary) / "admission.json"
            admission_path.write_text("{}", encoding="utf-8")
            pin = hashlib.sha256(admission_path.read_bytes()).hexdigest()
            args = SimpleNamespace(
                pr=79,
                max_seconds=60,
                config_root=PUBLISHER.ROOT / "configs",
                admission=admission_path,
                admission_sha256=pin,
                dry_run=True,
                once=True,
                monitor_root=PUBLISHER.DEFAULT_MONITOR_ROOT,
                state_file=Path(temporary) / "state.json",
            )
            fake_admission = {
                "admission_sha256": pin,
                "source_sha256": PUBLISHER.SOURCE_SHA256,
                "profiles": {},
            }
            empty_cells = [
                {"cell": name, "identity_ok": False, "complete": False}
                for name in PUBLISHER.CELLS
            ]
            output = io.StringIO()
            with (
                patch.object(PUBLISHER, "_admission", return_value=fake_admission),
                patch.object(PUBLISHER, "_load_baseline", return_value=_baseline()),
                patch.object(PUBLISHER, "inspect_cell", side_effect=empty_cells),
                patch.object(
                    PUBLISHER.P,
                    "_source_clock",
                    return_value={"local_now": 1_794_361_800.0, "age_seconds": 1.0},
                ),
                patch.object(
                    PUBLISHER, "_client", side_effect=AssertionError("client loaded")
                ),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(PUBLISHER.run(args), 0)
            self.assertIn("monitor comparisons", output.getvalue())
            self.assertFalse(args.state_file.exists())


if __name__ == "__main__":
    unittest.main()
