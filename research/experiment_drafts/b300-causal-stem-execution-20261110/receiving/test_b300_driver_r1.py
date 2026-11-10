"""CPU-only fixtures for the B300 receiving gates; no torch, CUDA, SSH, or child run."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

from b300_driver_core_r1 import (
    CELLS,
    COMMIT,
    FIT_ADMISSION_SCHEMA,
    FIT_WAVES,
    SOURCE_SHA256,
    SPLIT_SHA256,
    TRAIN_MEMBERS,
    VALID_MEMBERS,
    canonical_sha256,
    parse_proc_stat,
    sha256_file,
    validate_authority,
    validate_clock,
    validate_config,
    validate_fit_admission,
    validate_profile_receipt,
)
from b300_driver_support_r1 import _load_native_config

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST_SHA = "1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936"
NORMALIZED_CONFIG_SHAS = {
    "plain_s203201": "85028e3f3b81c5c888713a961ee61a8d50f2f90a353834183b3da8cb48ce8608",
    "plain_s203202": "594f38a43fd886f670abe22597be6da6a0d640bcfc91ad0c8a0860b3b44fd92d",
    "k1_s203201": "5c4783ef4c019c2e5736176a479e3d85db389ec1892fa3c72a1658bbe01b4816",
    "k1_s203202": "c93a88846f8d71bcd0f0d08795ba1eb75e69c06860b8a642a2894e8aa0e9c6bd",
    "k16_s203201": "9f21f962ff0c11a101103ebbc98f408d84d4f262283ce312cb0a66ad21ac8974",
    "k16_s203202": "eaf2102e1a4ff8c5eccc16c03db166e68ca741699c132d88e219c601a251b60e",
}


class ReceiverGateTests(unittest.TestCase):
    """Prove authority, fit-wave, profile, and process-parsing gates on synthetic bytes."""

    def test_authority_and_six_configs_match_frozen_bytes(self) -> None:
        authority = json.loads((ARTIFACT_ROOT / "AUTHORITY_R2.json").read_text())
        root = Path(authority["native_root"])
        configs = validate_authority(
            authority, root=root, manifest_sha=PINNED_MANIFEST_SHA, commit=COMMIT
        )
        self.assertEqual(
            FIT_WAVES,
            (
                ("plain_s203201", "k1_s203201", "k16_s203201", "plain_s203202"),
                ("k1_s203202", "k16_s203202"),
            ),
        )
        archive = ARTIFACT_ROOT / "source-r1.tar.gz"
        self.assertEqual(sha256_file(archive), authority["source"]["archive_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with tarfile.open(archive, "r:gz") as packaged:
                packaged.extractall(source, filter="data")
            for cell in CELLS:
                config_path = (
                    source
                    / "research/experiment_drafts/transformer-causal-input-stem-20261109/configs"
                    / f"{cell}.json"
                )
                configs[cell]["path"] = str(config_path)
                normalized = _load_native_config(source, config_path)
                config, digest = validate_config(
                    config_path, configs[cell], cell, normalized
                )
                self.assertEqual(config["seed"], configs[cell]["seed"])
                self.assertEqual(digest, canonical_sha256(config))
                self.assertEqual(digest, NORMALIZED_CONFIG_SHAS[cell])
        self.assertEqual(SOURCE_SHA256, authority["source"]["source_sha256"])
        self.assertEqual(SPLIT_SHA256, authority["split_sha256"])

    def test_clock_requires_original_plus_five_and_six_hours(self) -> None:
        self.assertEqual(
            validate_clock(
                {
                    "first_profile_epoch": 1000,
                    "fit_deadline_epoch": 19_000,
                    "wave_deadline_epoch": 22_600,
                }
            ),
            (1000.0, 19_000.0, 22_600.0),
        )
        with self.assertRaises(ValueError):
            validate_clock(
                {
                    "first_profile_epoch": 1000,
                    "fit_deadline_epoch": 19_010,
                    "wave_deadline_epoch": 22_600,
                }
            )

    def test_proc_stat_parser_handles_parentheses_in_comm(self) -> None:
        tail = [
            "R",
            "9",
            "417",
            "417",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "1",
            "2",
            "0",
            "0",
            "20",
            "0",
            "1",
            "0",
            "77",
            "0",
            "3",
        ]
        parsed = parse_proc_stat(417, f"417 (trainer ) worker) {' '.join(tail)}", 4096)
        self.assertEqual(parsed.pgid, 417)
        self.assertEqual(parsed.session, 417)
        self.assertEqual(parsed.start_ticks, 77)
        self.assertEqual(parsed.rss_bytes, 12_288)

    def test_profile_receipt_requires_both_rank_reload_and_fit_only_members(
        self,
    ) -> None:
        uuid_pair = ["a" * 36, "b" * 36]
        files = {"src/wnn/training/sequence_fit.py": "c" * 64}
        fit = [["train", index] for index in range(TRAIN_MEMBERS)]
        evaluation = [["valid", index] for index in range(VALID_MEMBERS)]
        identity: dict[str, Any] = {
            "source_sha256": SOURCE_SHA256,
            "source_files": files,
            "config_sha256": "d" * 64,
            "world_size": 2,
            "data": {
                "split_sha256": SPLIT_SHA256,
                "fit_members": fit,
                "evaluation_members": evaluation,
                "normalization": {"scope": "all_fit", "sequence_count": TRAIN_MEMBERS},
            },
            "runtime": {
                "devices": [{"uuid": item} for item in uuid_pair],
                "precision": {"mode": "bf16_bf16x9"},
            },
        }
        receipt = {
            "receipt_schema": "wnn.sequence-profile.v1",
            "status": "profile_complete",
            "approved": False,
            "updates": 50,
            "errors": [],
            "error_count": 0,
            "scientific_checkpoint_written": False,
            "identity": identity,
            "memory_by_rank": [
                {
                    "rank": rank,
                    "peak_reserved_bytes": 40,
                    "device_total_memory_bytes": 100,
                }
                for rank in range(2)
            ],
            "engineering_reload": {
                "status": "passed",
                "by_rank": [
                    {"rank": rank, "model_optimizer_scheduler_rng_equal": True}
                    for rank in range(2)
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile_receipt.json"
            path.write_text(json.dumps(receipt))
            accepted = validate_profile_receipt(
                path,
                config={"epochs": 4},
                config_sha=identity["config_sha256"],
                source_sha=SOURCE_SHA256,
                source_files=files,
                expected_split=SPLIT_SHA256,
                expected_uuids=uuid_pair,
            )
            self.assertEqual(accepted["updates"], 50)
            receipt["engineering_reload"]["by_rank"][1][
                "model_optimizer_scheduler_rng_equal"
            ] = False
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                validate_profile_receipt(
                    path,
                    config={"epochs": 4},
                    config_sha=identity["config_sha256"],
                    source_sha=SOURCE_SHA256,
                    source_files=files,
                    expected_split=SPLIT_SHA256,
                    expected_uuids=uuid_pair,
                )

    def test_fit_admission_binds_all_profile_receipt_bytes(self) -> None:
        authority = json.loads((ARTIFACT_ROOT / "AUTHORITY_R2.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-r1"
            configs = authority["configs"]
            admission_profiles: dict[str, dict[str, Any]] = {}
            for cell in CELLS:
                config = configs[cell]
                config["config_sha256"] = "e" * 64
                entries: dict[str, Any] = {
                    "config_sha256": config["config_sha256"],
                    "raw_config_sha256": config["raw_sha256"],
                    "pair": config["pair"],
                    "gpu_uuids": config["gpu_uuids"],
                }
                for name, relative in (
                    ("production_receipt", Path("profile") / "profile_receipt.json"),
                    ("peak_receipt", Path("profile-peak") / "profile_receipt.json"),
                    ("approved_receipt", Path("approved_profile.json")),
                ):
                    path = root / "runs" / cell / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{cell}:{name}")
                    entries[f"{name}_path"] = str(path)
                    entries[f"{name}_sha256"] = sha256_file(path)
                admission_profiles[cell] = entries
            admission = {
                "schema": FIT_ADMISSION_SCHEMA,
                "approved": True,
                "max_fresh_fits": 6,
                "source_sha256": SOURCE_SHA256,
                "source_root": str(source),
                "authority_sha256": "f" * 64,
                "split_sha256": SPLIT_SHA256,
                "profiles": admission_profiles,
            }
            accepted = validate_fit_admission(
                admission,
                expected_sha="1" * 64,
                actual_sha="1" * 64,
                authority_sha="f" * 64,
                source_sha=SOURCE_SHA256,
                source_root=source,
                configs=configs,
                root=root,
            )
            self.assertEqual(set(accepted), set(CELLS))
            admission_profiles[CELLS[0]]["approved_receipt_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                validate_fit_admission(
                    admission,
                    expected_sha="1" * 64,
                    actual_sha="1" * 64,
                    authority_sha="f" * 64,
                    source_sha=SOURCE_SHA256,
                    source_root=source,
                    configs=configs,
                    root=root,
                )


if __name__ == "__main__":
    unittest.main()
