"""Build an identified resource-telemetry successor from the immutable R4 bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(CHECKOUT / "scripts"))
from kaggle_common import canonical_sha256, sha256_file, write_json  # noqa: E402

R4_DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4"
R5_DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5"
R4_KERNEL = f"{R4_DATASET}-runner"
R5_KERNEL = f"{R5_DATASET}-runner"
R3_PACKET = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort"
)
R5_PACKET = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort"
)
R4_ARCHIVE_SHA = "ae8796a9c0cca1ed64d45341c14c917fc19f6889a1df20ff1873a87a200570dc"
R4_TERMINAL_RECEIPT_SHA = "833570ab8908a119811bed53a11477ba515db511eb65d55f08de659be61afac0"
R4_DIAGNOSTICS_SHA = "121a317b9bd75117939af3901116e496a3466a6d477c6d5095f2be1ad8734ad8"
R5_ALLOWED_DELTA = [
    "embed_exact_root_conditional_release_receipt_r5",
    "authorize_only_the_existing_profile_gated_teacher_fit_branch",
    "rebind_bootstrap_constants_for_r5_slug_archive_gate_and_release_receipt",
    "add_host_and_cgroup_memory_snapshots_around_the_existing_fit_admission",
    "advise_dontneed_only_for_hash_verified_declared_raw_and_own_cache_inputs_after_profiles_exit",
    "rebind_worker_attempt_names_and_package_identity_without_changing_science_or_resource_limits",
]
CACHE_PATH = re.compile(r"^(?:datasets/(?:train|valid)\.parquet|cache/[^/]+)$")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_tar(archive: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or member.name in files
            ):
                raise ValueError(f"unsafe R4 archive member: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"missing R4 member content: {member.name}")
            files[member.name] = stream.read()
    return files


def write_tar(files: dict[str, bytes], archive_path: Path) -> None:
    with (
        archive_path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def changed_spec(payload: bytes) -> dict:
    spec = json.loads(payload)

    def rename_attempt(value: str) -> str:
        value = re.sub(r"(_s[345])_r3(?=$|[^A-Za-z0-9])", r"\1_r5", value)
        return re.sub(r"(_paired_t4)_r3(?=$|[^A-Za-z0-9])", r"\1_r5", value)

    spec["name"] = rename_attempt(spec["name"])
    spec["argv"] = [rename_attempt(value) for value in spec["argv"]]
    spec["artifacts"] = [rename_attempt(value) for value in spec["artifacts"]]
    spec["code_sha256"] = canonical_sha256(sorted(spec["code"], key=lambda row: row["path"]))
    return spec


def release_delta() -> list[str]:
    return list(R5_ALLOWED_DELTA)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one anchor for {label}, found {text.count(old)}")
    return text.replace(old, new, 1)


def get_input_rows(files: dict[str, bytes], manifest: dict) -> list[dict]:
    binding = json.loads(files[f"{R3_PACKET}/INPUT_BINDING_R3.json"])
    rows = binding.get("external_inputs")
    if (
        binding.get("schema") != "wnn.kaggle.clean_t3_f0_inputs_binding.r3"
        or not isinstance(rows, list)
        or {row.get("path") for row in rows} != set(manifest["external_mounts"])
    ):
        raise ValueError("R3 external-input binding differs from its sealed mount map")
    for row in rows:
        name = row["path"]
        if not CACHE_PATH.fullmatch(name) and name not in {
            "artifacts/runs/c20260929_prop_f0/model.pt",
            "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "research/campaign_2026-10-04/portable_inputs.json",
        }:
            raise ValueError(f"unexpected external input: {name}")
    return rows


def resource_contract(rows: list[dict]) -> dict:
    selected = [
        {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in rows
        if CACHE_PATH.fullmatch(row["path"])
    ]
    if len(selected) != 10:
        raise ValueError(f"expected 10 exact raw/cache inputs, observed {len(selected)}")
    return {
        "schema": "wnn.kaggle.teacher_cache_handoff_contract.r5",
        "status": "frozen",
        "advice": "POSIX_FADV_DONTNEED",
        "advice_semantics": "best-effort kernel hint; receipt never claims bytes were released",
        "fit_admission": "unchanged kaggle_gpu_runner.admit_specs; each fit retains its 8,000,000,000-byte RAM reservation",
        "allowed_paths": sorted(selected, key=lambda row: row["path"]),
        "prohibited": [
            "global/privileged drop_caches",
            "advice to code, model checkpoints, split/portable manifests, propensity model, or undeclared paths",
            "writes/deletes/data mutation, cap changes, reservation reductions, or serial-fit fallback",
        ],
    }


def patch_kernel(
    old: str,
    archive_sha: str,
    manifest: dict,
    bindings: dict,
    gate: dict,
    pending_release: dict,
    driver_path: str,
) -> str:
    old = replace_once(
        old,
        f"BUNDLE_SLUG = '{R4_DATASET.split('/')[-1]}'",
        f"BUNDLE_SLUG = '{R5_DATASET.split('/')[-1]}'",
        "R5 slug",
    )
    old = replace_once(
        old,
        f"EXPECTED_ARCHIVE_SHA256 = '{R4_ARCHIVE_SHA}'",
        f"EXPECTED_ARCHIVE_SHA256 = '{archive_sha}'",
        "R5 archive hash",
    )
    constants = (
        f"R5_GATE_RELATIVE_PATH = '{R5_PACKET}/CONDITIONAL_FIT_ADMISSION_R5.json'\n"
        f"R5_GATE_SHA256 = '{sha_bytes(json_bytes(gate))}'\n"
        f"R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH = '{R5_PACKET}/ROOT_CONDITIONAL_RELEASE_R5.json'\n"
        f"R5_ROOT_RELEASE_RECEIPT_SHA256 = '{sha_bytes(json_bytes(pending_release))}'\n"
        f"R5_PRESEAL_BINDINGS = {bindings!r}\n"
        f"R5_RELEASE_TARGET = {{'dataset': '{R5_DATASET}', 'kernel_id': '{R5_KERNEL}'}}\n"
        f"R5_ALLOWED_DELTA = {R5_ALLOWED_DELTA!r}\n"
    )
    anchor = "R3_GATE_RELATIVE_PATH = "
    start = old.index(anchor)
    end = old.index("\n", start)
    old = old[: end + 1] + constants + old[end + 1 :]
    check = r"""    r5_gate_path = root / R5_GATE_RELATIVE_PATH
    if not r5_gate_path.is_file() or hashlib.sha256(r5_gate_path.read_bytes()).hexdigest() != R5_GATE_SHA256:
        raise PermissionError("R5 conditional gate bytes are missing or changed")
    r5_gate = json.loads(r5_gate_path.read_text())
    if r5_gate.get("status") != "admitted" or r5_gate.get("release_authorized") is not True:
        raise PermissionError("R5 root conditional admission is pending; refusing before external mounts")
    if r5_gate.get("root_release_receipt_path") != R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH or r5_gate.get("root_release_receipt_sha256") != R5_ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R5 gate does not bind the exact root receipt")
    r5_release_path = root / R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH
    if not r5_release_path.is_file() or hashlib.sha256(r5_release_path.read_bytes()).hexdigest() != R5_ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R5 root release receipt bytes are missing or changed")
    r5_release = json.loads(r5_release_path.read_text())
    if r5_release.get("schema") != "wnn.root_conditional_teacher_cohort_release.r5" or r5_release.get("status") != "admitted" or r5_release.get("authorized") is not True:
        raise PermissionError("R5 root release receipt is not authorized")
    if r5_release.get("binding", {}).get("preseal_r5") != R5_PRESEAL_BINDINGS or r5_release.get("target") != R5_RELEASE_TARGET or r5_release.get("allowed_delta") != R5_ALLOWED_DELTA:
        raise PermissionError("R5 root receipt does not bind the exact source, guards, and target")
    quota = r5_release.get("binding", {}).get("quota_budget", {})
    _require_fixed_quota_integer(quota.get("wall_ceiling_seconds"), 14400, "R5 wall ceiling")
    _require_fixed_quota_integer(quota.get("two_device_gpu_seconds_upper_bound"), 28800, "R5 GPU upper bound")
    _require_fixed_quota_integer(quota.get("reserved_gpu_seconds"), 28800, "R5 GPU reservation")
    if quota.get("paid_spending_authorized") is not False:
        raise PermissionError("R5 release may not authorize paid spending")
    _require_finite_quota_headroom(quota.get("campaign_headroom_gpu_seconds"), "R5 campaign GPU headroom")
    _require_finite_quota_headroom(quota.get("provider_headroom_gpu_seconds"), "R5 provider GPU headroom")
"""
    old = replace_once(
        old,
        '    for logical, mapping in manifest["external_mounts"].items():\n',
        check + '    for logical, mapping in manifest["external_mounts"].items():\n',
        "R5 pre-mount release check",
    )
    old_entry = "command = [sys.executable, str(root / 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/kernel_support/teacher_cohort_r3_driver.py'), \"--root\", str(root)]"
    new_entry = f"command = [sys.executable, str(root / '{driver_path}'), \"--root\", str(root)]"
    old = replace_once(old, old_entry, new_entry, "R5 driver entrypoint")
    old = replace_once(
        old,
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n',
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n'
        '    environment["WNN_R5_GATE_SHA256"] = R5_GATE_SHA256\n'
        '    environment["WNN_R5_ROOT_RELEASE_SHA256"] = R5_ROOT_RELEASE_RECEIPT_SHA256\n',
        "R5 auth environment",
    )
    return old


def build(r4_packet: Path, terminal_receipt: Path, diagnostics: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"R5 output exists: {output}")
    manifest_path = r4_packet / "bundle_teacher_cohort_r3/bundle-manifest.json"
    archive_path = r4_packet / "bundle_teacher_cohort_r3/bundle.wnn"
    manifest_bytes = manifest_path.read_bytes()
    parent_manifest = json.loads(manifest_bytes)
    if (
        parent_manifest.get("dataset") != R4_DATASET
        or parent_manifest.get("archive_sha256") != R4_ARCHIVE_SHA
    ):
        raise ValueError("sealed R4 parent manifest identity mismatch")
    if sha256_file(archive_path) != R4_ARCHIVE_SHA:
        raise ValueError("R4 archive SHA-256 mismatch")
    if sha256_file(terminal_receipt) != R4_TERMINAL_RECEIPT_SHA:
        raise ValueError("R4 terminal receipt SHA-256 mismatch")
    if sha256_file(diagnostics) != R4_DIAGNOSTICS_SHA:
        raise ValueError("R4 diagnostic receipt SHA-256 mismatch")
    files = read_tar(archive_path)
    mounted_paths = set(parent_manifest["external_mounts"])
    declared = {
        row["path"] for row in parent_manifest["files"] if row["path"] not in mounted_paths
    } | set(parent_manifest["spec_paths"])
    if declared != set(files):
        raise ValueError("R4 archive members do not exactly match its manifest")
    for row in parent_manifest["files"]:
        if row["path"] in mounted_paths:
            continue
        payload = files[row["path"]]
        if len(payload) != row["size"] or sha_bytes(payload) != row["sha256"]:
            raise ValueError(f"R4 member hash/bytes mismatch: {row['path']}")
    rows = get_input_rows(files, parent_manifest)
    contract = resource_contract(rows)

    specs = []
    for old_path in parent_manifest["spec_paths"]:
        spec = changed_spec(files[old_path])
        if (
            spec["limits"]["ram_bytes"] != 8_000_000_000
            and spec["kind"] == "gpu"
            and spec["name"].startswith("fit_")
        ):
            raise ValueError(f"R5 fit RAM reservation changed: {spec['name']}")
        specs.append(spec)
    files = {name: value for name, value in files.items() if not name.startswith("kaggle_specs/")}
    for spec in specs:
        files[f"kaggle_specs/{spec['name']}.json"] = json_bytes(spec)

    driver_path = f"{R5_PACKET}/kernel_support/teacher_cohort_r5_driver.py"
    driver_template = Path(__file__).resolve().parent / "kernel_support/teacher_cohort_r5_driver.py"
    driver = driver_template.read_bytes()
    compile(driver.decode("utf-8"), str(driver_template), "exec")
    plan_path = r4_packet / "TEACHER_COHORT_PLAN_R3.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    bindings = {
        "schema": "wnn.kaggle.teacher_cohort_r5_preseal_bindings.r1",
        "r4_dataset": R4_DATASET,
        "r4_kernel_id": R4_KERNEL,
        "r4_archive_sha256": R4_ARCHIVE_SHA,
        "r4_bundle_manifest_sha256": sha_bytes(manifest_bytes),
        "r4_terminal_receipt_sha256": R4_TERMINAL_RECEIPT_SHA,
        "r4_diagnostics_sha256": R4_DIAGNOSTICS_SHA,
        "r4_actual_failure_fixture_sha256": sha256_file(
            Path(__file__).resolve().parent
            / "fixtures/r4_attempt_001/R4_ACTUAL_FAILURE_FIXTURE_R1.json"
        ),
        "r3_teacher_plan_sha256": sha_bytes(plan_bytes),
        "r4_failure_classification": {
            "provider_terminal": "ERROR",
            "paired_profiles_completed": True,
            "teacher_fits_started": False,
            "error": "ValueError: Worker reservation exceeds host operational budget",
            "exact_post_profile_memory_inventory": "not persisted",
            "file_cache_cause": "unproven hypothesis",
            "inference_only": "given the recorded 32,212,254,720-byte capacity remained unchanged, the runner error implies budget <8,000,000,000B and therefore available <14,442,450,944B; exact available bytes were not captured",
        },
        "source_manifest_sha256": sha_bytes(
            files[f"{R3_PACKET}/source_snapshot/SOURCE_MANIFEST_H390.json"]
        ),
        "source_binding_sha256": sha_bytes(files[f"{R3_PACKET}/SOURCE_BINDING_R3.json"]),
        "input_binding_sha256": sha_bytes(files[f"{R3_PACKET}/INPUT_BINDING_R3.json"]),
        "producer_preflight_sha256": sha_bytes(
            files[f"{R3_PACKET}/ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json"]
        ),
        "profile_guard_contract_sha256": sha_bytes(
            files[f"{R3_PACKET}/PROFILE_GUARD_CONTRACT_R3.json"]
        ),
        "profile_reference_sha256": sha_bytes(files[f"{R3_PACKET}/PROFILE_REFERENCE_R3.json"]),
        "resource_handoff_contract_sha256": sha_bytes(json_bytes(contract)),
        "r5_driver_sha256": sha_bytes(driver),
        "worker_specs_sha256": {spec["name"]: sha_bytes(json_bytes(spec)) for spec in specs},
        "science_recipe_sha256": canonical_sha256(plan["recipe"]),
        "fit_worker_ram_reservation_bytes": 8_000_000_000,
        "allowed_delta": release_delta(),
    }
    gate = {
        "schema": "wnn.kaggle.conditional_teacher_fit_admission.r5",
        "status": "pending_root_review_and_release",
        "release_authorized": False,
        "root_release_receipt_path": f"{R5_PACKET}/ROOT_CONDITIONAL_RELEASE_R5.json",
        "root_release_receipt_sha256": None,
        "preseal_bindings_sha256": sha_bytes(json_bytes(bindings)),
        "profile_guard_contract_sha256": bindings["profile_guard_contract_sha256"],
        "profile_reference_sha256": bindings["profile_reference_sha256"],
        "resource_handoff_contract_sha256": bindings["resource_handoff_contract_sha256"],
        "required": [
            "independent review binds the exact R5 archive/bootstrap/inputs/guard/driver",
            "root creates an R5 preseal receipt for the exact R5 target and fresh quota",
            "the paired T4 profile guard passes",
            "actual host/cgroup snapshots and bounded cache-advice outcomes are persisted",
            "unchanged admit_specs returns concurrency two before any fit worker starts",
        ],
        "note": "Pending R5 state blocks at bootstrap before external mounts; R4 authorization is not authority for R5.",
    }
    pending = {
        "schema": "wnn.root_conditional_teacher_cohort_release.r5",
        "status": "pending_root_review_and_release",
        "authorized": False,
        "binding": {"preseal_r5": bindings},
        "target": {"dataset": R5_DATASET, "kernel_id": R5_KERNEL},
        "allowed_delta": release_delta(),
        "binding_note": "placeholder only; packet builder cannot create root authority",
    }
    extras = {
        driver_path: driver,
        f"{R5_PACKET}/R5_RESOURCE_HANDOFF_CONTRACT_R1.json": json_bytes(contract),
        f"{R5_PACKET}/R5_PRESEAL_BINDINGS_R1.json": json_bytes(bindings),
        f"{R5_PACKET}/CONDITIONAL_FIT_ADMISSION_R5.json": json_bytes(gate),
        f"{R5_PACKET}/ROOT_CONDITIONAL_RELEASE_R5.json": json_bytes(pending),
    }
    for name, payload in extras.items():
        if name in files:
            raise ValueError(f"R5 package path collision: {name}")
        files[name] = payload

    output.mkdir(parents=True)
    bundle_dir = output / "bundle_teacher_cohort_r5"
    bundle_dir.mkdir()
    for name, payload in (
        ("kernel_support/teacher_cohort_r5_driver.py", driver),
        ("R5_RESOURCE_HANDOFF_CONTRACT_R1.json", json_bytes(contract)),
        ("R5_PRESEAL_BINDINGS_R1.json", json_bytes(bindings)),
        ("CONDITIONAL_FIT_ADMISSION_R5.json", json_bytes(gate)),
        ("ROOT_CONDITIONAL_RELEASE_R5.json", json_bytes(pending)),
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    new_rows = [
        {"path": name, "size": len(payload), "sha256": sha_bytes(payload)}
        for name, payload in sorted(files.items())
        if name not in parent_manifest["external_mounts"]
    ]
    spec_paths = [f"kaggle_specs/{spec['name']}.json" for spec in specs]
    manifest = {
        **parent_manifest,
        "dataset": R5_DATASET,
        "files": new_rows,
        "spec_paths": spec_paths,
        "expanded_bytes": sum(row["size"] for row in new_rows),
    }
    write_tar(files, bundle_dir / "bundle.wnn")
    manifest["archive_sha256"] = sha256_file(bundle_dir / "bundle.wnn")
    manifest["archive_bytes"] = (bundle_dir / "bundle.wnn").stat().st_size
    write_json(bundle_dir / "bundle-manifest.json", manifest)
    write_json(
        bundle_dir / "dataset-metadata.json",
        {
            "id": R5_DATASET,
            "title": R5_DATASET.split("/")[-1],
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )

    old_kernel = (r4_packet / "bundle_teacher_cohort_r3/kernel/kernel.py").read_text()
    kernel = patch_kernel(
        old_kernel, manifest["archive_sha256"], manifest, bindings, gate, pending, driver_path
    )
    (bundle_dir / "kernel").mkdir()
    (bundle_dir / "kernel/kernel.py").write_text(kernel)
    kernel_metadata = json.loads(
        (r4_packet / "bundle_teacher_cohort_r3/kernel/kernel-metadata.json").read_text()
    )
    kernel_metadata["id"] = R5_KERNEL
    kernel_metadata["title"] = f"{R5_DATASET.split('/')[-1]} runner"
    old_sources = kernel_metadata["dataset_sources"][1:]
    kernel_metadata["dataset_sources"] = [R5_DATASET, *old_sources]
    write_json(bundle_dir / "kernel/kernel-metadata.json", kernel_metadata)

    prep = {
        "schema": "wnn.kaggle.teacher_cohort_r5_preparation_receipt.r1",
        "status": "prepared_pending_independent_review_and_root_release",
        "r4_archive_sha256": R4_ARCHIVE_SHA,
        "r4_manifest_sha256": sha_bytes(manifest_bytes),
        "r4_terminal_receipt_sha256": R4_TERMINAL_RECEIPT_SHA,
        "r4_diagnostics_sha256": R4_DIAGNOSTICS_SHA,
        "r5_dataset": R5_DATASET,
        "r5_kernel_id": R5_KERNEL,
        "r5_archive_sha256": manifest["archive_sha256"],
        "r5_manifest_sha256": sha256_file(bundle_dir / "bundle-manifest.json"),
        "r5_bootstrap_sha256": sha256_file(bundle_dir / "kernel/kernel.py"),
        "r5_driver_sha256": sha_bytes(driver),
        "r5_preseal_bindings_sha256": sha_bytes(json_bytes(bindings)),
        "r5_resource_handoff_contract_sha256": sha_bytes(json_bytes(contract)),
        "fit_ram_reservations_bytes": sorted(
            {
                spec["limits"]["ram_bytes"]
                for spec in specs
                if spec["kind"] == "gpu" and spec["name"].startswith("fit_")
            }
        ),
        "science_recipe_changed": False,
        "fit_resource_limits_changed": False,
        "root_authority_created": False,
        "provider_action_performed": False,
    }
    write_json(output / "R5_PREPARATION_RECEIPT_R1.json", prep)
    write_json(output / "R5_PRESEAL_BINDINGS_R1.json", bindings)
    write_json(output / "R5_RESOURCE_HANDOFF_CONTRACT_R1.json", contract)
    (output / "README_R5.md").write_text(
        "# R5 teacher cohort: measured resource handoff\\n\\n"
        "R5 is a new pending derivative of the immutable R4 source and recipe. R4 completed two paired T4 profile workers, then failed before any fit at the unchanged 8 GB per-worker operational admission. R4 did not persist the exact post-profile host/cgroup inventory; the file-cache explanation remains unproven.\\n\\n"
        "R5 retains the H384 two-layer GRU, Muon-all, fold-0, propensity-power-2, seeds 3/4/5, fixed epoch-2 checkpoints, exact source/input ancestry, and 8 GB per-worker fit reservation. It adds before/after host and cgroup memory snapshots. Before advising POSIX_FADV_DONTNEED on the exact 10 raw/cache paths, it checks this attempt's runner SHA receipts for eight cache files, the trainer full-content hash events for both Parquet files, and unchanged mounted device/inode/size/mtime/ctime from before profiles. It does not reread the large inputs for advice, and a returned advice call is not claimed to release bytes. R5 then invokes the unchanged kaggle_gpu_runner.admit_specs gate and stops if it refuses or does not return concurrency two. No cache/global drop, cap change, or serial fallback.\\n\\n"
        "The embedded R5 gate is pending. R4 authorization does not authorize R5. Root review, fresh quota/session, physical short-stage hash/READY checks, and separate root release are still required. This preparation made no provider changes and started no fits.\\n"
    )
    return prep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-packet", type=Path, required=True)
    parser.add_argument("--r4-terminal-receipt", type=Path, required=True)
    parser.add_argument("--r4-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.r4_packet, args.r4_terminal_receipt, args.r4_diagnostics, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
