"""Build the identified V4 profile packet with the frozen raw-manifest schema fix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from kaggle_assets import prepare

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
V3_PACKET = (
    CAMPAIGN
    / "gru512_768_capacity_v3_terminal_20261008/downloaded_outputs"
    / "gru512_768_muon_capacity_v3_profile_packet"
)
OUT = CAMPAIGN / "gru512_768_muon_capacity_v4_manifest_profile_20261008/packet"
SOURCE_DATASET = "vladkochetov07/e529-gru512-768-muon-capacity-v4-source-20261008"
KERNEL_SLUG = "vladkochetov07/e529-gru512-768-muon-capacity-v4-profile-20261008-runner"
PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V4_20261008"
RECIPE_NAME = "GRU512_768_MUON_PROFILE_RECIPE_V4.json"
AUTH_NAME = "ROOT_PROFILE_AUTHORIZATION.json"
REVIEW_NAME = "ROOT_PROFILE_SOURCE_REVIEW.json"
PACKET_FILES = (
    "capacity_worker.py",
    "kernel.py",
    "train_entry.py",
    "verified_digest_handoff.py",
    "README.md",
    "kernel-metadata.json",
    "source/source_archive.tar",
    "source/source_archive_manifest.json",
    "fold0.json",
    "portable_inputs.json",
    "prior_input_identity_v2.json",
    RECIPE_NAME,
    AUTH_NAME,
    REVIEW_NAME,
)
V3_RECIPE = "GRU512_768_MUON_PROFILE_RECIPE_V3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _v3_driver_template() -> str:
    builder_path = ROOT / "scripts/build_gru_capacity_v3_profile_packet.py"
    spec = importlib.util.spec_from_file_location("gru_capacity_v3_builder_template", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V3 Kaggle driver template")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return str(module.DRIVER_TEMPLATE)


def build_packet() -> dict[str, object]:
    if OUT.exists():
        raise FileExistsError("V4 output already exists; preserve it and use a new identity")
    if not V3_PACKET.is_dir():
        raise FileNotFoundError("committed V3 terminal packet is required as the frozen base")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(V3_PACKET, OUT, ignore=shutil.ignore_patterns("__pycache__"))

    old_recipe_path = OUT / V3_RECIPE
    recipe = json.loads(old_recipe_path.read_text(encoding="utf-8"))
    old_recipe_sha = sha256_file(old_recipe_path)
    old_recipe_path.unlink()

    worker_path = OUT / "capacity_worker.py"
    worker = worker_path.read_text(encoding="utf-8")
    for old, new in (
        ("E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008", PACKET_ID),
        ("GRU512_768_MUON_PROFILE_RECIPE_V3.json", RECIPE_NAME),
        ("V3 profile-only packet", "V4 profile-only packet"),
        ("gru512_768_muon_capacity_v3_profile_receipts", "gru512_768_muon_capacity_v4_manifest_profile_receipts"),
        ("gru512_768_muon_capacity_v3_profile_project", "gru512_768_muon_capacity_v4_manifest_profile_project"),
        ("capacityv3profile", "capacityv4manifestprofile"),
    ):
        worker = worker.replace(old, new)
    marker = "def _verify_raw_source(\n"
    if worker.count(marker) != 1:
        raise ValueError("V3 raw verification function changed; V4 patch needs review")
    binding_helper = '''def _raw_source_expected_binding(
    expected: dict[str, object],
    prior_observation: dict[str, object] | None,
    logical_name: str,
) -> tuple[str, int]:
    """Validate the producer's nested size/hash schema against the frozen prior digest."""
    if set(expected) != {"original_identity", "sha256"}:
        raise ValueError(f"portable raw entry schema changed: {logical_name}")
    identity = expected.get("original_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"portable raw identity is malformed: {logical_name}")
    expected_sha = expected.get("sha256")
    expected_bytes = identity.get("bytes")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        raise ValueError(f"portable raw fingerprint or size is malformed: {logical_name}")
    identity_sha = identity.get("sha256")
    if identity_sha is not None and identity_sha != expected_sha:
        raise ValueError(f"portable raw nested fingerprint conflicts: {logical_name}")
    if (
        not isinstance(prior_observation, dict)
        or prior_observation.get("sha256") != expected_sha
        or prior_observation.get("bytes") != expected_bytes
    ):
        raise ValueError(
            f"prior raw fingerprint or size differs from portable manifest: {logical_name}"
        )
    return expected_sha, expected_bytes


'''
    worker = worker.replace(marker, binding_helper + marker)
    old_binding = '''    expected_sha = str(expected["sha256"])
    expected_bytes = int(expected["bytes"])
    prior = prior_observation or {}
'''
    new_binding = '''    expected_sha, expected_bytes = _raw_source_expected_binding(
        expected, prior_observation, logical_name
    )
    prior = prior_observation
'''
    if worker.count(old_binding) != 1:
        raise ValueError("V3 raw manifest consumer binding changed; V4 patch needs review")
    worker = worker.replace(old_binding, new_binding)
    worker_path.write_text(worker, encoding="utf-8")

    recipe["packet_id"] = PACKET_ID
    recipe["status"] = "profile-only-source-prepared-pending-root-review-and-admission"
    recipe["source"]["v3_base_recipe_sha256"] = old_recipe_sha
    recipe["source"]["manifest_schema_fix"] = (
        "Read raw size from original_identity.bytes, validate the portable SHA/size against "
        "the frozen prior observed SHA/size, then use the existing exact-stat or streaming-hash path."
    )
    recipe["source"]["lineage"] = (
        "V3 physical-GPU, source-qualified membership, checkpoint producer and private transport "
        "with the V4 portable raw-manifest schema correction; scientific recipe unchanged."
    )
    metadata = json.loads((OUT / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = KERNEL_SLUG
    metadata["title"] = "GRU512/768 Muon V4 manifest-schema profile only"
    metadata["dataset_sources"][0] = SOURCE_DATASET
    write_json(OUT / "kernel-metadata.json", metadata)

    recipe["runtime"]["requested_kernel"] = KERNEL_SLUG
    recipe["runtime"]["requested_source_dataset"] = SOURCE_DATASET
    recipe["runtime"]["initial_execution_scope"] = "profile-only; conditional fit authorization is false"
    recipe["authorization"] = {
        "profile": "requires independent root source review and fresh Kaggle quota admission",
        "conditional_fit": "disabled in this profile-only packet; requires a separately authorized successor",
        "grading": False,
        "provider_push": False,
    }
    recipe_path = OUT / RECIPE_NAME
    write_json(recipe_path, recipe)
    write_json(
        OUT / REVIEW_NAME,
        {
            "schema": "wnn-gru-capacity-independent-profile-source-review-v1",
            "packet_id": PACKET_ID,
            "status": "pending_independent_review",
            "authorized": False,
        },
    )
    (OUT / "README.md").write_text(
        "# GRU512/768 Muon capacity V4 profile-only packet\n\n"
        "V4 preserves V3's model, seed, clean-fold split, objective, sampler and measured profile shapes. "
        "It corrects the consumer to use the producer's `original_identity.bytes` field and checks the "
        "expected portable SHA/size against the prior observed SHA/size before taking the stat-reuse path. "
        "Profile-only; no fit is authorized. V3's terminal failure remains preserved as execution evidence.\n\n"
        "The private source dataset contains only packet files. Raw train/valid and the named fold0 assets "
        "remain separate read-only mounts. This V4 candidate has not been uploaded or pushed.\n",
        encoding="utf-8",
    )

    bindings = {}
    for name in PACKET_FILES:
        if name in {RECIPE_NAME, AUTH_NAME, REVIEW_NAME}:
            continue
        path = OUT / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"V4 packet member is missing or not regular: {name}")
        bindings[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    recipe["source"]["file_bindings"] = bindings
    write_json(recipe_path, recipe)
    recipe_sha = sha256_file(recipe_path)
    write_json(
        OUT / AUTH_NAME,
        {
            "schema": "wnn-gru-capacity-root-conditional-authorization-v2",
            "packet_id": PACKET_ID,
            "status": "pending_root_review",
            "authorized": False,
            "profile_authorized": False,
            "conditional_fit_authorized": False,
            "fit_arm_ids": [],
            "selected_batch": None,
            "max_worker_seconds": 5200,
            "recipe_sha256": recipe_sha,
            "source_archive_sha256": bindings["source/source_archive.tar"]["sha256"],
            "source_manifest_sha256": bindings["source/source_archive_manifest.json"]["sha256"],
            "portable_manifest_sha256": bindings["portable_inputs.json"]["sha256"],
            "split_manifest_sha256": bindings["fold0.json"]["sha256"],
            "prior_input_identity_sha256": bindings["prior_input_identity_v2.json"]["sha256"],
            "kernel_metadata_sha256": bindings["kernel-metadata.json"]["sha256"],
            "worker_sha256": bindings["capacity_worker.py"]["sha256"],
            "entry_sha256": bindings["kernel.py"]["sha256"],
            "train_entry_sha256": bindings["train_entry.py"]["sha256"],
            "digest_handoff_sha256": bindings["verified_digest_handoff.py"]["sha256"],
            "root_review_receipt": REVIEW_NAME,
            "root_review_receipt_sha256": sha256_file(OUT / REVIEW_NAME),
            "notes": "V4 remains profile-only until separate root authorization and fresh provider admission.",
        },
    )
    upload_root = OUT / "source_dataset_upload"
    prepare(OUT, list(PACKET_FILES), upload_root, SOURCE_DATASET)
    driver = _v3_driver_template()
    driver = driver.replace("__SOURCE_DATASET_SLUG__", SOURCE_DATASET)
    driver = driver.replace("E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008", PACKET_ID)
    driver = driver.replace(
        "gru512_768_muon_capacity_v3_profile_packet",
        "gru512_768_muon_capacity_v4_manifest_profile_packet",
    )
    driver = driver.replace("__ASSET_MANIFEST_SHA256__", sha256_file(upload_root / "assets-manifest.json"))
    driver = driver.replace("__REQUIRED_PACKET_FILES__", repr(PACKET_FILES))
    if "__SOURCE_DATASET_SLUG__" in driver or "__ASSET_MANIFEST_SHA256__" in driver:
        raise ValueError("V4 driver template placeholders were not fully bound")
    kernel_upload = OUT / "kernel_upload"
    kernel_upload.mkdir()
    (kernel_upload / "kernel.py").write_text(driver, encoding="utf-8")
    write_json(kernel_upload / "kernel-metadata.json", metadata)

    result = {
        "packet_id": PACKET_ID,
        "packet_path": str(OUT),
        "packet_source_dataset": SOURCE_DATASET,
        "packet_source_manifest_sha256": sha256_file(upload_root / "assets-manifest.json"),
        "recipe_sha256": recipe_sha,
        "worker_sha256": sha256_file(worker_path),
        "portable_manifest_sha256": sha256_file(OUT / "portable_inputs.json"),
        "prior_input_identity_sha256": sha256_file(OUT / "prior_input_identity_v2.json"),
        "source_archive_sha256": sha256_file(OUT / "source/source_archive.tar"),
        "upload_driver_sha256": sha256_file(kernel_upload / "kernel.py"),
        "fit_authorized": False,
        "upload_or_push_performed": False,
        "v3_base_recipe_sha256": old_recipe_sha,
        "v3_base_worker_sha256": sha256_file(V3_PACKET / "capacity_worker.py"),
        "v3_terminal_download_manifest_sha256": sha256_file(
            CAMPAIGN / "gru512_768_capacity_v3_terminal_20261008/downloaded_file_manifest.json"
        ),
        "v3_packet_builder_sha256": sha256_file(
            ROOT / "scripts/build_gru_capacity_v3_profile_packet.py"
        ),
        "v4_packet_builder_sha256": sha256_file(Path(__file__)),
    }
    write_json(OUT / "PACKET_BUILD_RECEIPT.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(build_packet(), sort_keys=True, indent=2))
