from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
GATE_PATH = Path(__file__).with_name("fold1_gru384_postfit_gate.py")
SPEC = importlib.util.spec_from_file_location("fold1_postfit_gate", GATE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not import the frozen postfit gate")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

from wnn.run_identity import group_identity  # noqa: E402
from wnn.splits import CampaignSplit  # noqa: E402


class PostfitGateTests(unittest.TestCase):
    """Exercise fail-closed gates and bind the snapshot-order assumption to a real run."""

    def test_actual_fixed_epoch_checkpoint_embeds_preselection_provenance(self) -> None:
        output = ROOT / "artifacts/runs/bd26_w384raw_s3_f0"
        sidecar = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        checkpoint = gate.finite_checkpoint(output / "model_ep3.pt")
        selected = gate.finite_checkpoint(output / "model.pt")
        self.assertEqual(sidecar["selected_epoch"], 3)
        self.assertNotIn("selected_epoch", checkpoint["provenance"])
        self.assertEqual(checkpoint["args"], sidecar["args"])
        self.assertEqual(selected["args"], sidecar["args"])
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(selected["epoch"], 3)
        self.assertEqual(checkpoint["model"].keys(), selected["model"].keys())
        self.assertTrue(
            all(
                torch.equal(checkpoint["model"][key], selected["model"][key])
                for key in checkpoint["model"]
            )
        )

    def test_attempt_command_and_oom_are_hard_gates(self) -> None:
        command = ["timeout", "7200", "trainer", "--seed", "3"]
        step = {
            "attempt_id": "fold1-gru384-s3-fit-r1",
            "command": command,
            "command_sha256_canonical_json": gate.canonical_sha256(command),
        }
        record = {
            "id": step["attempt_id"],
            "status": "finished",
            "exit_status": 0,
            "command": command,
            "contract_sha256": "a" * 64,
            "memory_event_deltas": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
            "peak_ram_bytes": 8_000_000_000,
            "peak_observed_vram_bytes": 2_000_000_000,
            "unit": "fit.service",
            "finished_utc": "2026-10-09T00:00:00+00:00",
        }
        gate.verify_attempt_record(record, step, "a" * 64)
        record["memory_event_deltas"]["oom_kill"] = 1
        with self.assertRaisesRegex(ValueError, "OOM"):
            gate.verify_attempt_record(record, step, "a" * 64)

    def test_fit_evaluation_component_overlap_is_rejected(self) -> None:
        split = CampaignSplit(
            Path("fold1.json"),
            "b" * 64,
            1,
            4,
            (0,),
            (1,),
            (2,),
            (
                ("train", 0, 10),
                ("valid", 1, 20),
                ("valid", 2, 30),
                ("valid", 3, 30),
            ),
        )
        manifest_hash = "c" * 64
        source_pins = {
            "research/campaign_2026-09-29/artifacts/splits/fold1.json": manifest_hash,
            "src/wnn/run_identity.py": gate.sha256_file(ROOT / "src/wnn/run_identity.py"),
        }
        args = {
            "name": "fixture_seed3",
            "device": "cuda",
            "arch": "gru",
            "cell": "gru",
            "hidden": 384,
            "layers": 2,
            "dropout": 0.0,
            "seed": 3,
            "raw_only": True,
            "loss": "wmse",
            "loss_norm": "chunk",
            "train_source": "train+valid",
            "train_seqs": 10607,
            "valid_fold": 1,
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "no_eval_during_fit": True,
            "batch": 16,
            "chunk": 2000,
            "lr": 0.001,
            "wd": 0.01,
            "optim": "adamw",
            "sched": "onecycle",
            "precision": "fp16",
            "save_training_state": True,
            "split_manifest_sha256": manifest_hash,
            "split_manifest": "fold1.json",
        }
        provenance = {
            "args": args,
            "split_manifest_sha256": manifest_hash,
            "selected_epoch": 3,
            "run_identity_source_sha256": source_pins["src/wnn/run_identity.py"],
            "fit_row_groups": {"train": [0], "valid": [1]},
            "evaluation_row_groups": [2],
            "fit_membership_identity": {"train": group_identity([0]), "valid": group_identity([1])},
        }
        step = {"run_name": "fixture_seed3"}
        gate.verify_provenance_membership(step, provenance, split, source_pins, True)
        leaky_split = split._replace(valid_groups=(1, 3))
        provenance["fit_row_groups"]["valid"] = [1, 3]
        provenance["fit_membership_identity"]["valid"] = group_identity([1, 3])
        with self.assertRaisesRegex(ValueError, "component overlap"):
            gate.verify_provenance_membership(step, provenance, leaky_split, source_pins, True)

    def test_nonfinite_checkpoint_and_output_overage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "bad.pt"
            torch.save({"epoch": 3, "model": {"weight": torch.tensor([float("nan")])}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                gate.finite_checkpoint(checkpoint)
            output = root / "out"
            output.mkdir()
            (output / "large.bin").write_bytes(b"x" * 16)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                gate.directory_bytes(output, 15)


if __name__ == "__main__":
    unittest.main()
