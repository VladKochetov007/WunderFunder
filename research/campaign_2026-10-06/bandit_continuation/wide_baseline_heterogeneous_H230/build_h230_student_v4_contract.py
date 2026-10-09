#!/usr/bin/env python3
"""Create the metadata-only H230 student contract successor after real load checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230"
RUNTIME = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts"
V3_PATH = LINEAGE / "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V3_20261008.json"
V4_PATH = LINEAGE / "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V4_20261008.json"
V4_MARKDOWN = LINEAGE / "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V4_20261008.md"
LOAD_CHECK_PATH = LINEAGE / "H230_ALLDATA_STUDENT_PARENT_LOAD_CHECK_V1_20261008.json"
PARENT_TERMINAL_PATH = RUNTIME / "h230-all-data-parent-fit-b16-ep3-retry3-v1-20261008.json"
CACHE_TERMINAL_PATH = RUNTIME / "h230-all-data-teacher-cache-v2-20261008.json"
CACHE_METADATA_PATH = ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json"
PARENT_CHECKPOINT_PATH = ROOT / "artifacts/runs/bd26_h230_all_data_parent_b16_ep3_retry3_v1/model_ep3.pt"
V4_APPROVAL_PATH = LINEAGE / "H230_ALLDATA_STUDENT_PAIR_ROOT_AUTHORIZATION_V4_20261008.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args()

    for path in (V3_PATH, LOAD_CHECK_PATH, PARENT_TERMINAL_PATH, CACHE_TERMINAL_PATH, CACHE_METADATA_PATH, PARENT_CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    v3 = read_json(V3_PATH)
    load_check = read_json(LOAD_CHECK_PATH)
    parent_terminal = read_json(PARENT_TERMINAL_PATH)
    cache_terminal = read_json(CACHE_TERMINAL_PATH)
    cache_metadata = read_json(CACHE_METADATA_PATH)

    if load_check.get("status") != "passed_actual_source_v3_checkpoint_load_and_seed_specific_head_extension":
        raise AssertionError("the source-v3 parent-load/head-extension gate has not passed")
    if load_check.get("parent", {}).get("checkpoint_sha256") != sha256_file(PARENT_CHECKPOINT_PATH):
        raise AssertionError("load-check receipt does not bind the current parent checkpoint")
    if parent_terminal.get("exit_status") != 0 or parent_terminal.get("status") != "finished":
        raise AssertionError("parent terminal receipt is not a successful terminal fit")
    if cache_terminal.get("exit_status") != 0 or cache_terminal.get("status") != "finished":
        raise AssertionError("all-data cache terminal receipt is not successful")
    if cache_metadata.get("membership_mode") != "all_supplied_train_valid":
        raise AssertionError("cache metadata does not declare all-supplied membership")
    if cache_metadata.get("heldout_claim") is not False:
        raise AssertionError("all-data cache metadata unexpectedly makes a heldout claim")

    expected_shapes = {item["file"]: item["shape"] for item in v3["teacher_cache_dependency"]["expected_arrays"]}
    cache_files = cache_metadata["files"]
    if len(cache_files) != 8:
        raise AssertionError(f"expected eight finalized cache arrays, found {len(cache_files)}")
    for item in v3["teacher_cache_dependency"]["expected_arrays"]:
        entry = cache_files[item["file"].removesuffix(".npy")]
        if entry["path"] != item["file"] or entry["shape"] != expected_shapes[item["file"]]:
            raise AssertionError(f"cache metadata differs from frozen expected array {item['file']}")
        if entry["dtype"] != item["dtype"]:
            raise AssertionError(f"cache dtype differs for {item['file']}")

    v4 = json.loads(json.dumps(v3))
    v4["contract_id"] = "H230_ALLDATA_STUDENT_PAIR_RECIPE_V4_20261008"
    v4["created_utc"] = args.created_utc
    v4["status"] = "root_authorized_metadata_corrected_pending_fresh_per_student_resource_admission"
    v4["launch_authorized"] = True
    v4["authorization_basis"] = [
        "Root explicitly authorized a V4 metadata-only successor correcting the actual source/checkpoint-proven parent-to-student output width, after the complete all-data cache and real checkpoint-load/head-extension checks pass.",
        "Root authorized launching the exact student pair sequentially after those checks and fresh admission, with no additional confirmation required; current resource admission remains a hard per-student controller gate.",
        "V3 is preserved as historical evidence of the stale output-row metadata. Scientific training argv, targets, losses, optimizer, precision, sampler, checkpoint epoch, and runtime caps are unchanged.",
    ]
    v4["supersedes_contract"] = {
        "path": str(V3_PATH.relative_to(ROOT)),
        "sha256": sha256_file(V3_PATH),
        "reason": "V4 corrects only the parent/student output-width metadata using the actual ep3 checkpoint and source-v3 trainer/helper replay. The exact worker science argv is unchanged.",
    }

    v4["source"]["source_relation_to_parent"] = (
        "Students use sealed source-v3. Parent retry3 was trained from sealed source-v2. The actual model_ep3.pt was loaded through source-v3 make_model, append_family_output_head, extend_family_output_head, and strict load_state_dict for both student seeds. The 23 compatible parent rows are preserved exactly and eight seed-specific family rows remain freshly initialized."
    )
    v4["source"]["actual_parent_load_check_path"] = str(LOAD_CHECK_PATH.relative_to(ROOT))
    v4["source"]["actual_parent_load_check_sha256"] = sha256_file(LOAD_CHECK_PATH)
    v4["source"]["actual_parent_load_check_status"] = load_check["status"]

    v4["parent_dependency"].update(
        {
            "fit_status_at_packet_freeze": "finished_successful_fixed_ep3",
            "checkpoint_sha256": load_check["parent"]["checkpoint_sha256"],
            "terminal_receipt_path": str(PARENT_TERMINAL_PATH.relative_to(ROOT)),
            "terminal_receipt_sha256": sha256_file(PARENT_TERMINAL_PATH),
            "checkpoint_load_status": "passed_actual_source_v3_load_for_both_student_seeds",
            "actual_load_check_path": str(LOAD_CHECK_PATH.relative_to(ROOT)),
            "actual_load_check_sha256": sha256_file(LOAD_CHECK_PATH),
            "model_output_layout": {
                "inheritance": "All 23 ordinary main/auxiliary rows and the full recurrent trunk are copied from the shared parent; the appended eight family output rows remain freshly initialized per student seed.",
                "parent_source_v2": "family_head=false; actual head.2 weight shape (23,256) and bias shape (23,); 23 ordinary main/auxiliary output rows.",
                "student_source_v3": "family_head=true; actual student head.2 weight shape (31,256) and bias shape (31,); 23 ordinary main/auxiliary rows plus eight appended family rows.",
                "verification_gate": f"Passed; see {LOAD_CHECK_PATH.relative_to(ROOT)} SHA256 {sha256_file(LOAD_CHECK_PATH)}. Exact source-v3 functions were replayed for seeds3 and4 against the fixed parent ep3 checkpoint.",
            },
        }
    )

    v4["recipe"]["student_architecture"].update(
        {
            "parent_main_aux_outputs": 23,
            "ordinary_main_target_outputs": 23,
            "family_outputs_appended_after_parent": 8,
            "student_total_outputs": 31,
            "parent_family_head": False,
            "family_head_initialization": "fresh seed-specific eight-row head; actual source-v3 initialization and parent extension verified for seeds3 and4",
        }
    )
    v4["recipe"]["student_pair_ancestry"] = (
        "Both train from the same own all-data parent ep3 and same all-data teacher cache. The actual source-v3 loader copies the parent's 23 recurrent/main/auxiliary output rows into each 31-row student and retains a fresh, separately initialized eight-row family head per seed. Pair members are not independent whole-model draws."
    )

    v4["teacher_cache_dependency"].update(
        {
            "launch_authorized": True,
            "cache_metadata_sha256": sha256_file(CACHE_METADATA_PATH),
            "cache_build_receipt_path": str(CACHE_TERMINAL_PATH.relative_to(ROOT)),
            "cache_build_receipt_sha256": sha256_file(CACHE_TERMINAL_PATH),
            "cache_validation": {
                "status": "passed_all_arrays_and_three_target_variants",
                "variants": ["source", "normalized", "homogeneous"],
                "array_count": 8,
                "validation_method": "source-v3 validate_all_data_distillation_cache; full output hashes, metadata/shape/dtype, row-group identity/order, ancestry, and finite chunk checks passed.",
                "execution": "CPU-only, one thread, 1.5GB memory cap; previous completed check wall time 16.687 seconds.",
            },
            "finalized_array_hashes": {
                key: {"path": value["path"], "sha256": value["sha256"], "bytes": value["bytes"], "shape": value["shape"], "dtype": value["dtype"]}
                for key, value in cache_files.items()
            },
        }
    )
    for array in v4["teacher_cache_dependency"]["expected_arrays"]:
        metadata_entry = cache_files[array["file"].removesuffix(".npy")]
        array["sha256"] = metadata_entry["sha256"]
        array["bytes"] = metadata_entry["bytes"]

    v4["launch_preconditions"] = [
        f"Passed parent dependency: successful fixed ep3 terminal {PARENT_TERMINAL_PATH.relative_to(ROOT)} and exact checkpoint SHA256 {load_check['parent']['checkpoint_sha256']}.",
        f"Passed actual source-v3 parent/student checkpoint load and seed-specific family-head extension for seeds3 and4; receipt {LOAD_CHECK_PATH.relative_to(ROOT)} SHA256 {sha256_file(LOAD_CHECK_PATH)}.",
        f"Passed complete all-data cache build and validation; terminal {CACHE_TERMINAL_PATH.relative_to(ROOT)}, metadata SHA256 {sha256_file(CACHE_METADATA_PATH)}, all eight finalized array hashes are embedded in this contract.",
        "Before each student, the durable resource controller must perform fresh admission: no active GPU worker, effective slice limits intact, aggregate CPU/RAM/swap/VRAM and disk headroom pass, worker reservation 9GB, two CPU cores, zero loader workers, and one GPU lock.",
        "Launch seed3 first, preserve its terminal receipt and fixed ep1 artifact; only then perform a separate fresh admission for seed4. Do not launch concurrently.",
        "All supplied train and valid groups are fitting data. The 468 exposed valid groups are not held out; no local heldout or public-score claim is authorized.",
    ]
    for run in v4["student_runs"]:
        run["resource_controller_command"] = run["resource_controller_command"].replace(
            "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V3_20261008.json",
            "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V4_20261008.json",
        )
    V4_PATH.write_text(json.dumps(v4, indent=2, sort_keys=False) + "\n")

    contract_sha = sha256_file(V4_PATH)
    approval = {
        "identity": "H230_ALLDATA_STUDENT_PAIR_ROOT_AUTHORIZATION_V4_20261008",
        "created_utc": args.created_utc,
        "approver": "root controller; explicit authorization recorded by assigned actor",
        "approval_basis": (
            "Root message: preserve V3, create the distinct V4 metadata-only student contract with correct parent23-to-student31 ordinary-plus-eight-family mapping and actual initialized tensor-row evidence, unchanged science argv/horizon/targets/loss/caps; launch only after cache/load checks and fresh admission, with no second confirmation needed."
        ),
        "launch_authorized": True,
        "scope": "Sequential seed3 then seed4 fits only, under each exact V4 resource-controller command and fresh per-student admission; no prediction, grading, packaging, or public score claim is authorized by this receipt.",
        "contract": {"path": str(V4_PATH.relative_to(ROOT)), "sha256": contract_sha},
        "supersedes": {"path": str(V3_PATH.relative_to(ROOT)), "sha256": sha256_file(V3_PATH)},
        "science_argv_unchanged_from_v3": True,
        "parent_dependency": {
            "checkpoint_path": str(PARENT_CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(PARENT_CHECKPOINT_PATH),
            "terminal_receipt_path": str(PARENT_TERMINAL_PATH.relative_to(ROOT)),
            "terminal_receipt_sha256": sha256_file(PARENT_TERMINAL_PATH),
            "actual_load_check_path": str(LOAD_CHECK_PATH.relative_to(ROOT)),
            "actual_load_check_sha256": sha256_file(LOAD_CHECK_PATH),
        },
        "cache_dependency": {
            "terminal_receipt_path": str(CACHE_TERMINAL_PATH.relative_to(ROOT)),
            "terminal_receipt_sha256": sha256_file(CACHE_TERMINAL_PATH),
            "metadata_sha256": sha256_file(CACHE_METADATA_PATH),
            "array_sha256_by_name": {key: value["sha256"] for key, value in cache_files.items()},
            "validation": "passed for all eight arrays and source/normalized/homogeneous variants before V4 freeze",
        },
        "per_student_fresh_resource_admission_still_required": True,
        "fresh_resource_admission_not_claimed_here": True,
    }
    V4_APPROVAL_PATH.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")

    V4_MARKDOWN.write_text(
        "# H230 all-data student pair recipe contract V4\n\n"
        f"V4 supersedes [V3](./{V3_PATH.name}) (SHA256 `{sha256_file(V3_PATH)}`), preserving it unchanged. "
        "V3's student head dimensions were stale. The actual parent ep3 checkpoint has 23 main/auxiliary output rows; "
        "the exact source-v3 trainer builds those 23 rows and appends eight family rows, giving 31 student outputs. "
        f"The actual source-v3 model construction, family-head append, extension helper, and strict load passed for both seeds; see [the load-check receipt](./{LOAD_CHECK_PATH.name}) "
        f"(SHA256 `{sha256_file(LOAD_CHECK_PATH)}`).\n\n"
        "All scientific worker arguments and the fixed two-epoch/fixed-ep1 schedule are unchanged from V3. The completed all-data cache's "
        "metadata and eight array hashes are bound in the JSON contract. Root authorized the sequential pair after these checks; each fit still "
        "requires a separate fresh resource-controller admission. Seed4 may launch only after seed3 reaches terminal status. The students share the "
        "same parent and teacher cache and are not independent whole-model draws. All supplied train+valid groups, including exposed valid groups, are fit data; "
        "there is no heldout or public-score claim.\n\n"
        f"Contract SHA256: `{contract_sha}`. Root authorization record: `{V4_APPROVAL_PATH.name}`.\n"
    )
    print(json.dumps({"contract": str(V4_PATH.relative_to(ROOT)), "contract_sha256": contract_sha, "approval": str(V4_APPROVAL_PATH.relative_to(ROOT)), "approval_sha256": sha256_file(V4_APPROVAL_PATH), "status": v4["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
