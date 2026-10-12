from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-source")
ASSET_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-assets")


def load_kernel():
    """Load the exact generated kernel without entering its runtime main."""
    spec = importlib.util.spec_from_file_location("treatment_kernel_r22", PACKET / "kernel" / "kernel.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("generated kernel module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    kernel = load_kernel()
    plan = json.loads((SOURCE_STAGE / "pair-plan.json").read_text())
    bundle_manifest = json.loads((SOURCE_STAGE / "bundle-manifest.json").read_text())
    release = json.loads((SOURCE_STAGE / "ROOT_TREATMENT_RELEASE_R2.json").read_text())

    if len(plan["arms"]) != 1 or plan["arms"][0]["arm"] != "treatment":
        raise AssertionError("cloud job list is not treatment-only")
    arm = plan["arms"][0]
    if [stage["stop_after_epoch_inclusive"] for stage in arm["fit_stages"]] != [1, 3, 5]:
        raise AssertionError("treatment resume schedule changed")
    if arm["export"]["run"] != arm["fit_stages"][-1]["name"] or arm["export"]["epoch"] != 5:
        raise AssertionError("export is not bound to the completed treatment p3 CP5")
    if any("combo_parent" in stage["name"] or "c20261003_combo_s3_f0@ep3" in stage["argv"] for stage in arm["fit_stages"]):
        raise AssertionError("local control was duplicated in the cloud job list")

    try:
        kernel.validate_root_release(release, plan, plan["source_manifest_sha256"])
    except PermissionError:
        pass
    else:
        raise AssertionError("pending root release did not refuse before worker setup")

    with tempfile.TemporaryDirectory(prefix="wnn-r22-fixture-") as temporary:
        fixture_root = Path(temporary)
        input_root = fixture_root / "inputs"
        input_root.mkdir()
        source_mount = input_root / SOURCE_STAGE.name
        source_mount.mkdir()
        expected_source_files = {
            "bundle.wnn", "bundle-manifest.json", "pair-plan.json", "ROOT_TREATMENT_RELEASE_R2.json"
        }
        observed_source_files = {path.name for path in SOURCE_STAGE.iterdir() if path.is_file()}
        if observed_source_files != expected_source_files | {"dataset-metadata.json"}:
            raise AssertionError(f"source upload staging differs from expected files: {observed_source_files}")
        for name in expected_source_files:
            (source_mount / name).symlink_to(SOURCE_STAGE / name)

        asset_mount = input_root / ASSET_STAGE.name
        asset_mount.mkdir()
        expected_asset_payload = {"assets-manifest.json", *plan["asset_files"]}
        observed_asset_payload = {path.name for path in ASSET_STAGE.iterdir() if path.is_file()}
        if observed_asset_payload != expected_asset_payload | {"dataset-metadata.json"}:
            raise AssertionError(f"asset upload staging differs from expected files: {observed_asset_payload}")
        for name in expected_asset_payload:
            (asset_mount / name).symlink_to(ASSET_STAGE / name)

        input_roots = {
            kernel.SOURCE_DATASET: source_mount,
            kernel.ASSET_DATASET: asset_mount,
            "vladkochetov07/wunder-funder": input_root / "wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets": input_root / "wnn-c20261004-fold0-assets",
        }
        for logical, mapping in plan["external_mounts"].items():
            if isinstance(mapping, str):
                dataset = "vladkochetov07/wunder-funder"
                filename = mapping
                source = MAIN / logical
            else:
                dataset = mapping["dataset"]
                filename = mapping["filename"]
                if dataset == kernel.ASSET_DATASET:
                    source = ASSET_STAGE / filename
                elif dataset == kernel.SOURCE_DATASET:
                    source = SOURCE_STAGE / filename
                else:
                    source = MAIN / logical
            if not source.is_file():
                raise FileNotFoundError(f"fixture source missing for {logical}: {source}")
            mount_root = input_roots.setdefault(dataset, input_root / dataset.rsplit("/", 1)[-1])
            target = mount_root / filename
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            target.symlink_to(source)

        extracted = fixture_root / "experiment"
        kernel.extract_bundle(SOURCE_STAGE / "bundle.wnn", bundle_manifest, extracted)
        receipt = kernel.validate_and_mount_external(extracted, input_roots, bundle_manifest, plan)
        if receipt["status"] != "pass" or len(receipt["observations"]) != len(plan["external_mounts"]):
            raise AssertionError("actual nested source mount fixture did not validate")
        raw = {item["path"] for item in receipt["observations"] if item["status"] == "deferred_to_frozen_trainer"}
        if raw != {"datasets/train.parquet", "datasets/valid.parquet"}:
            raise AssertionError(f"raw parquet identity was not deferred exactly to the frozen trainer: {raw}")
        stage = arm["fit_stages"][0]
        spec_path = kernel.build_spec(extracted, plan, stage["name"], stage["argv"], stage["artifacts"], stage["limits"])
        emitted = json.loads(spec_path.read_text())
        if emitted["name"] != stage["name"] or emitted["environment"]["WNN_PROJECT_ROOT"] != str(extracted):
            raise AssertionError("emitted treatment worker spec is not bound to extracted source root")
        if any(item["path"].startswith("artifacts/runs/c20261003_combo_s3_f0/") for item in emitted["inputs"]):
            raise AssertionError("control parent was included in treatment worker inputs")

    print(json.dumps({
        "schema": "wnn.clean_f0_student_treatment_offline_fixture.r2",
        "status": "pass",
        "provider_mutation": False,
        "cuda_or_trainer_import": False,
        "fit_or_score": False,
        "arm_count": 1,
        "fit_stage_epochs": [1, 3, 5],
        "export": {"run": arm["export"]["run"], "epoch": arm["export"]["epoch"], "shape": arm["export"]["shape"]},
        "external_mount_count": len(receipt["observations"]),
        "raw_parquet_full_hash_owner": "frozen trainer portable-source verification",
        "pending_root_release_refused": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
