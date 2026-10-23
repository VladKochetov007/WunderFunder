from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType

sys.dont_write_bytecode = True
sys.pycache_prefix = None

PREPARATION = Path(__file__).resolve().parent
BINDING_PATH = PREPARATION.parent / "first_cell_evaluation_export_v1/EXPORT_BINDING_V1.json"
CAPTURE_PATH = PREPARATION / "ACCEPT_EXPORT_CAPTURE_V2.py"
SCORE_DIRECTORY = PREPARATION.parent / "first_cell_evaluation_score_v1"
CAPTURE_BYTES = 18_187
CAPTURE_SHA256 = "dd009a0ff018e4541934aad27811ae3b3aa65b1f44456cf93932306c0e8abdc2"
BINDING_BYTES = 15_822
BINDING_SHA256 = "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c"
SCORER_PINS = {
    "first_cell_score_core": (
        27_226,
        "e115d12868b8d9c6ac85e1b497b0bc514559d78fff5dada0b0c350e18b69669a",
    ),
    "first_cell_score_contract": (
        19_485,
        "09155537feec1d9cd398f6a729ab88201b82e814c1a8c4ac796fe4283940a90d",
    ),
}
ARMS = ("control", "ema", "sparse", "nll", "allthree")
GROUPS = list(range(468))
PARENT_INODE = 1_234_567_890


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def load_capture_module() -> ModuleType:
    """Load only the frozen ROOT metadata validator, never its CLI entrypoint."""
    raw = CAPTURE_PATH.read_bytes()
    if (len(raw), sha256(raw)) != (CAPTURE_BYTES, CAPTURE_SHA256):
        raise RuntimeError("ROOT acceptance V2 source pin differs")
    spec = importlib.util.spec_from_file_location("accept_export_capture_v2_fixture", CAPTURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ROOT acceptance V2 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = load_capture_module()
BINDING_RAW = BINDING_PATH.read_bytes()
if (len(BINDING_RAW), sha256(BINDING_RAW)) != (BINDING_BYTES, BINDING_SHA256):
    raise RuntimeError("frozen export binding pin differs")
BINDING = json.loads(BINDING_RAW)
BINDING_PIN: dict[str, object] = {
    "path": str(BINDING_PATH),
    "bytes": len(BINDING_RAW),
    "sha256": sha256(BINDING_RAW),
}


def capture_bytes(raw: bytes) -> dict[str, object]:
    """Encode a small synthetic metadata receipt with exact byte joins."""
    return {"bytes": len(raw), "sha256": sha256(raw), "base64": base64.b64encode(raw).decode()}


def json_bytes(value: dict[str, object]) -> bytes:
    """Serialize a synthetic JSON metadata object deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_observations(identities: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Project the six admitted source identities to their exact stat-only fields."""
    keys = ("path", "resolved", "bytes", "device", "inode", "islink", "mtime_ns")
    return {name: {key: item[key] for key in keys} for name, item in identities.items()}


def build_data_identity(binding: dict[str, object]) -> dict[str, object]:
    """Build synthetic before/after stat observations without opening source data."""
    registered = binding["registered_data"]
    identities = copy.deepcopy(registered["source_identities"])
    valid = identities["datasets/valid.parquet"]
    leaf = {key: valid[key] for key in ("path", "resolved", "bytes", "device", "inode", "mtime_ns")}
    leaf["islink"] = False
    parent = {
        "path": str(Path(str(valid["path"])).parent),
        "resolved": str(Path(str(valid["resolved"])).parent),
        "device": valid["device"],
        "inode": PARENT_INODE,
    }
    valid_observation = {**leaf, "parent": parent}
    source_stats = source_observations(identities)
    portable = {
        **registered["portable_manifest"],
        "source_names": list(identities),
        "split_manifest_sha256": binding["split"]["sha256"],
        "version": 1,
    }
    return {
        "schema": "fg19.registered-valid-data-identity.v1",
        "admission": copy.deepcopy(registered["admission"]),
        "portable_manifest": portable,
        "registered_source_identities": identities,
        "registered_valid_parquet": copy.deepcopy(registered["valid_parquet"]),
        "observed_sources_before_split": copy.deepcopy(source_stats),
        "observed_sources_after_split": copy.deepcopy(source_stats),
        "observed_before_sequence_open": copy.deepcopy(valid_observation),
        "observed_after_export": copy.deepcopy(valid_observation),
    }


def build_fixture(binding: dict[str, object]) -> dict[str, object]:
    """Build a complete synthetic five-arm export metadata capture, with no arrays."""
    membership = sha256("".join(f"{group}\n" for group in GROUPS).encode("ascii"))
    split = {**binding["split"], "role": "fit-excluded-exposed-development-evaluation"}
    manifest: dict[str, object] = {
        "schema": "fg19.first-cell-evaluation-export-manifest.v1",
        "status": "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED",
        "created_utc": "2026-10-21T12:00:00+00:00",
        "binding_sha256": BINDING_PIN["sha256"],
        "source": copy.deepcopy(binding["source"]),
        "split": split,
        "data_identity": build_data_identity(binding),
        "ordered_groups": GROUPS.copy(),
        "membership_sha256": membership,
        "arms": [],
        "quality_or_metric_claim": False,
    }
    metadata: dict[str, object] = {"EXPORT_MANIFEST_V1.json": capture_bytes(json_bytes(manifest))}
    arm_rows: list[dict[str, object]] = []
    for arm, recipe in zip(ARMS, binding["arms"], strict=True):
        checkpoint = {**recipe["model"], "loaded_sha256": recipe["model"]["sha256"]}
        prediction = {
            "path": "predictions.npy",
            "bytes": 74_880_128,
            "npy_header_bytes": 128,
            "data_bytes": 74_880_000,
            "sha256": "a" * 64,
            "dtype": "float32-little-endian",
            "shape": [468, 20_000, 2],
        }
        sidecar: dict[str, object] = {
            "schema": "fg19.first-cell-feature-only-prediction.v1",
            "status": "PREDICTION_EXPORT_ONLY_NOT_SCORED",
            "binding_sha256": BINDING_PIN["sha256"],
            "arm": arm,
            "run_name": recipe["run_name"],
            "unit": recipe["unit"],
            "invocation_id": recipe["invocation_id"],
            "selected_epoch": 5,
            "successful_updates": 44_760,
            "terminal_acceptance": copy.deepcopy(recipe["terminal_acceptance"]),
            "exact_stop": copy.deepcopy(recipe["exact_stop"]),
            "checkpoint": checkpoint,
            "representation_mode": recipe["representation_mode"],
            "source": copy.deepcopy(binding["source"]),
            "split": copy.deepcopy(split),
            "ordered_groups": GROUPS.copy(),
            "membership_encoding": "ASCII decimal row-group IDs, one per line, final LF",
            "membership_sha256": membership,
            "feature_only": {"targets_read": False, "scored_mask_read": False},
            "representation": {
                "mode": recipe["representation_mode"],
                **copy.deepcopy(binding["representation"]),
            },
            "inference": {
                "precision": "strict_fp32",
                "autocast": False,
                "tf32": False,
                "chunk_rows": 2_000,
                "sequence_batch_size": 1,
            },
            "prediction": prediction,
            "scoring_performed": False,
        }
        sidecar_raw = json_bytes(sidecar)
        sidecar_pin = capture_bytes(sidecar_raw)
        metadata[f"{arm}/PREDICTION_SIDECAR.json"] = sidecar_pin
        arm_rows.append(
            {
                "arm": arm,
                "directory": arm,
                "run_name": recipe["run_name"],
                "unit": recipe["unit"],
                "invocation_id": recipe["invocation_id"],
                "representation_mode": recipe["representation_mode"],
                "selected_epoch": 5,
                "successful_updates": 44_760,
                "checkpoint": checkpoint,
                "membership_sha256": membership,
                "prediction_path": f"{arm}/predictions.npy",
                "sidecar_path": f"{arm}/PREDICTION_SIDECAR.json",
                "prediction_bytes": prediction["bytes"],
                "prediction_sha256": prediction["sha256"],
                "sidecar_bytes": sidecar_pin["bytes"],
                "sidecar_sha256": sidecar_pin["sha256"],
            }
        )
    manifest["arms"] = arm_rows
    metadata["EXPORT_MANIFEST_V1.json"] = capture_bytes(json_bytes(manifest))
    return {
        "schema": "fg19.first-cell-export-bounded-status.v1",
        "unit": "computer-vision-homework-fg19-first-cell-export-v1.service",
        "invocation_id": binding["arms"][0]["invocation_id"],
        "systemd": {
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
            "InvocationID": binding["arms"][0]["invocation_id"],
        },
        "array_payloads_read_or_hashed": False,
        "terminal_metadata": metadata,
    }


class RootExportAcceptanceV2MetadataTests(unittest.TestCase):
    """Check the root capture producer against the frozen score metadata contract."""

    def setUp(self) -> None:
        self.binding = copy.deepcopy(BINDING)
        self.observation = build_fixture(self.binding)

    def call_validate(
        self, observation: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        return CAPTURE.validate_metadata(observation, self.binding, BINDING_PIN)

    def assert_no_wnn_imports(self) -> None:
        self.assertFalse(any(name == "wnn" or name.startswith("wnn.") for name in sys.modules))

    def test_complete_synthetic_five_arm_capture_passes_scorer_leaf(self) -> None:
        for name in SCORER_PINS:
            self.assertNotIn(name, sys.modules)
        manifest, outputs = self.call_validate(self.observation)
        self.assertEqual(len(manifest["ordered_groups"]), 468)
        self.assertEqual(list(outputs), list(ARMS))
        for module_name, (expected_bytes, expected_sha) in SCORER_PINS.items():
            module_path = Path(sys.modules[module_name].__file__).resolve(strict=True)
            self.assertEqual(
                module_path, (SCORE_DIRECTORY / f"{module_name}.py").resolve(strict=True)
            )
            source = module_path.read_bytes()
            self.assertEqual((len(source), sha256(source)), (expected_bytes, expected_sha))
        self.assert_no_wnn_imports()

    def test_empty_per_source_observation_is_rejected(self) -> None:
        observation = copy.deepcopy(self.observation)
        manifest = json.loads(
            base64.b64decode(observation["terminal_metadata"]["EXPORT_MANIFEST_V1.json"]["base64"])
        )
        manifest["data_identity"]["observed_sources_before_split"] = {}
        observation["terminal_metadata"]["EXPORT_MANIFEST_V1.json"] = capture_bytes(
            json_bytes(manifest)
        )
        with self.assertRaisesRegex(ValueError, "observed_sources_before_split membership differs"):
            self.call_validate(observation)
        self.assert_no_wnn_imports()

    def test_missing_registered_valid_parent_is_rejected(self) -> None:
        observation = copy.deepcopy(self.observation)
        manifest = json.loads(
            base64.b64decode(observation["terminal_metadata"]["EXPORT_MANIFEST_V1.json"]["base64"])
        )
        del manifest["data_identity"]["observed_before_sequence_open"]["parent"]
        observation["terminal_metadata"]["EXPORT_MANIFEST_V1.json"] = capture_bytes(
            json_bytes(manifest)
        )
        with self.assertRaisesRegex(ValueError, "observed_before_sequence_open fields differ"):
            self.call_validate(observation)
        self.assert_no_wnn_imports()

    def test_sidecar_membership_tamper_is_rejected(self) -> None:
        observation = copy.deepcopy(self.observation)
        del observation["terminal_metadata"]["allthree/PREDICTION_SIDECAR.json"]
        with self.assertRaisesRegex(ValueError, "metadata-only capture membership differs"):
            self.call_validate(observation)
        self.assert_no_wnn_imports()


if __name__ == "__main__":
    unittest.main()
