from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
STAGE = PACKET / "kernel_r4"
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-source")
ASSET_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-assets")
PROFILE_FIXTURE = STAGE / "actual_t4_profile_guard_fixture_r1.json"


def load_kernel():
    spec = importlib.util.spec_from_file_location("treatment_kernel_r4", STAGE / "kernel.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("R4 generated kernel module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_payload(input_root: Path, dataset: str, layout: str) -> Path:
    owner, slug = dataset.split("/", 1)
    if layout == "flat":
        return input_root / slug
    if layout == "nested":
        return input_root / "datasets" / owner / slug / "version_1" / "payload"
    raise ValueError(layout)


def fixture_source(dataset: str, filename: str, plan: dict) -> Path:
    if dataset == "vladkochetov07/wnn-clean-t3-f0-treatment-source-20261012-r22":
        return SOURCE_STAGE / filename
    if dataset == "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22":
        return ASSET_STAGE / filename
    for logical, mapping in plan["external_mounts"].items():
        mapped_dataset = "vladkochetov07/wunder-funder" if isinstance(mapping, str) else mapping["dataset"]
        mapped_filename = mapping if isinstance(mapping, str) else mapping["filename"]
        if mapped_dataset == dataset and mapped_filename == filename:
            return MAIN / logical
    raise FileNotFoundError(f"no producer source for {dataset}/{filename}")


def create_input_fixture(input_root: Path, layout: str, kernel, plan: dict) -> dict[str, Path]:
    source_expected = kernel.SOURCE_EXPECTED_FILES
    asset_expected = {
        name: descriptor
        for name, descriptor in {
            **{
                name: {"size": item["size"], "sha256": item["sha256"]}
                for name, item in plan["asset_files"].items()
            },
            "assets-manifest.json": {"size": 949, "sha256": kernel.ASSET_MANIFEST_SHA256},
        }.items()
    }
    grouped = kernel.external_dataset_expectations(plan)
    expected_by_dataset = {
        kernel.SOURCE_DATASET: source_expected,
        kernel.ASSET_DATASET: asset_expected,
        **{name: files for name, files in grouped.items() if name != kernel.ASSET_DATASET},
    }
    roots = {}
    for dataset, expected_files in expected_by_dataset.items():
        payload = mounted_payload(input_root, dataset, layout)
        payload.mkdir(parents=True)
        roots[dataset] = payload
        for filename in expected_files:
            source = fixture_source(dataset, filename, plan)
            if not source.is_file():
                raise FileNotFoundError(f"offline fixture producer is missing: {source}")
            target = payload / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)

    if layout == "nested":
        # A same-leaf-name decoy under the wrong owner must never be considered.
        decoy = input_root / "datasets" / "unrelated-owner" / "wunder-funder" / "payload"
        decoy.mkdir(parents=True)
        (decoy / "train.parquet").symlink_to(MAIN / "datasets/train.parquet")
        (decoy / "valid.parquet").symlink_to(MAIN / "datasets/valid.parquet")
    return roots


def check_duplicate_resolution(kernel) -> None:
    with tempfile.TemporaryDirectory(prefix="wnn-r4-ambiguous-") as temporary:
        input_root = Path(temporary) / "input"
        flat = input_root / "sample"
        nested = input_root / "datasets" / "org" / "sample" / "payload"
        flat.mkdir(parents=True)
        nested.mkdir(parents=True)
        good = b"bound-file"
        bad = b"other-file"
        (flat / "item.bin").write_bytes(good)
        (nested / "item.bin").write_bytes(bad)
        expected = {"item.bin": {"size": len(good), "sha256": hashlib.sha256(good).hexdigest()}}
        chosen, receipt = kernel.discover_dataset_mount(input_root, "org/sample", expected)
        if chosen != flat or not receipt["sha_disambiguation"]:
            raise AssertionError("duplicate candidates were not resolved by the declared content identity")
        (nested / "item.bin").write_bytes(good)
        try:
            kernel.discover_dataset_mount(input_root, "org/sample", expected)
        except RuntimeError as exc:
            if "ambiguous" not in str(exc):
                raise
        else:
            raise AssertionError("two identical matching payloads were accepted as an arbitrary first match")


def check_actual_profile_guard(kernel) -> dict:
    fixture = json.loads(PROFILE_FIXTURE.read_text())
    if fixture.get("resource_qualification") is not False or fixture.get("fixture_only") is not True:
        raise AssertionError("actual profile fixture must remain explicitly schema-only")
    with tempfile.TemporaryDirectory(prefix="wnn-r4-profile-") as temporary:
        output = Path(temporary)
        worker = fixture["finished_record"]
        worker_dir = output / worker["name"]
        worker_dir.mkdir()
        log_lines = [
            key + json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for key, value in fixture["markers"].items()
        ]
        (worker_dir / "worker.log").write_text("\n".join(log_lines) + "\n")
        guard = kernel.profile_guard({"worker": worker}, output)
        if guard["status"] != "pass" or guard["completed_batches"] != 30:
            raise AssertionError("actual producer summary/marker fixture did not satisfy the existing profile guard")
        if guard["optimizer_attempts"] != 600 or guard["gradient_nonfinite_steps"] != 1:
            raise AssertionError("actual producer update/gradient marker values were not retained")
        try:
            kernel.profile_guard({"worker": {"name": worker["name"], "status": "complete"}}, output)
        except (RuntimeError, KeyError):
            pass
        else:
            raise AssertionError("abbreviated heartbeat record was incorrectly accepted as a terminal profile")
        return guard


def main() -> None:
    kernel = load_kernel()
    plan = json.loads((SOURCE_STAGE / "pair-plan.json").read_text())
    if "Path('/kaggle/input/wunder-funder')" in (STAGE / "kernel.py").read_text():
        raise AssertionError("kernel retained the flat-only raw data path")
    check_duplicate_resolution(kernel)

    discovered = {}
    nested_consumer = None
    for layout in ("flat", "nested"):
        with tempfile.TemporaryDirectory(prefix=f"wnn-r4-{layout}-") as temporary:
            input_root = Path(temporary) / "input"
            input_root.mkdir()
            create_input_fixture(input_root, layout, kernel, plan)
            input_roots, observed_plan, receipt = kernel.discover_all_dataset_mounts(input_root)
            if observed_plan != plan or receipt["status"] != "pass":
                raise AssertionError(f"{layout} payload discovery did not preserve the bound plan")
            expected_datasets = {
                kernel.SOURCE_DATASET,
                kernel.ASSET_DATASET,
                "vladkochetov07/wunder-funder",
                "vladkochetov07/wnn-c20261004-fold0-assets",
            }
            if set(input_roots) != expected_datasets:
                raise AssertionError(f"{layout} discovery dataset set changed: {set(input_roots)}")
            expected_layout = "flat_slug" if layout == "flat" else "datasets_owner_slug"
            if any(item["layout"] != expected_layout for item in receipt["mounts"]):
                raise AssertionError(f"{layout} directory ancestry was not recorded consistently")
            if layout == "nested" and any(item["payload_depth"] != 2 for item in receipt["mounts"]):
                raise AssertionError("nested version/payload layers were not traversed exactly")
            if layout == "nested" and any("unrelated-owner" in item["mount_path"] for item in receipt["mounts"]):
                raise AssertionError("same-leaf-name dataset under an unrelated owner was selected")
            if layout == "nested":
                with tempfile.TemporaryDirectory(prefix="wnn-r4-nested-consumer-") as temporary_root:
                    bundle_manifest = json.loads((SOURCE_STAGE / "bundle-manifest.json").read_text())
                    consumer = kernel.validate_and_mount_external(
                        Path(temporary_root) / "experiment", input_roots, bundle_manifest, plan
                    )
                raw_paths = {
                    item["path"] for item in consumer["observations"]
                    if item["status"] == "deferred_to_frozen_trainer"
                }
                if consumer["status"] != "pass" or raw_paths != {
                    "datasets/train.parquet", "datasets/valid.parquet"
                }:
                    raise AssertionError("actual external-input consumer rejected nested payloads or changed raw hash ownership")
                nested_consumer = {
                    "status": consumer["status"],
                    "observation_count": len(consumer["observations"]),
                    "verified_nonparquet_count": sum(
                        item["status"] == "verified" for item in consumer["observations"]
                    ),
                    "raw_parquet_paths_deferred": sorted(raw_paths),
                }
            discovered[layout] = receipt

    profile_guard = check_actual_profile_guard(kernel)
    print(json.dumps({
        "schema": "wnn.clean_t3_f0_treatment_mount_profile_fixture.r4",
        "status": "pass",
        "provider_mutation": False,
        "trainer_import_or_cuda": False,
        "fit_or_score": False,
        "flat_dataset_ids": sorted(item["dataset_id"] for item in discovered["flat"]["mounts"]),
        "nested_dataset_ids": sorted(item["dataset_id"] for item in discovered["nested"]["mounts"]),
        "wrong_owner_same_leaf_decoy_ignored": True,
        "nested_actual_external_consumer": nested_consumer,
        "duplicate_content_identity_disambiguation": True,
        "identical_duplicate_payload_refused": True,
        "raw_parquet_hashing": "not performed; the frozen trainer remains the full-byte verifier",
        "actual_producer_profile_fixture": {
            "origin_summary_sha256": json.loads(PROFILE_FIXTURE.read_text())["source"]["summary_sha256"],
            "origin_worker_log_sha256": json.loads(PROFILE_FIXTURE.read_text())["source"]["worker_log_sha256"],
            "resource_qualification": False,
            "completed_batches": profile_guard["completed_batches"],
            "optimizer_attempts": profile_guard["optimizer_attempts"],
            "amp_skipped_updates": profile_guard["amp_skipped_updates"],
            "gradient_nonfinite_steps": profile_guard["gradient_nonfinite_steps"],
            "heartbeat_abbreviation_rejected": True,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
