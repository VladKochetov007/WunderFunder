from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any  # Primary gate inputs are heterogeneous JSON mappings.

from reproduce_h232_v4 import file_sha256, frozen_gate_arguments, reproduce_digest_scope


def profile_argv_ast(path: Path) -> str:
    """Compare the exact profile argv, allowing only its operational run name."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "argv" for target in node.targets):
            continue
        if (
            not node.value.elts
            or not isinstance(node.value.elts[0], ast.Constant)
            or node.value.elts[0].value != "--name"
        ):
            continue
        normalized = copy.deepcopy(node.value)
        normalized.elts[1] = ast.Constant(value="operational-profile-name")
        return ast.dump(normalized, include_attributes=False)
    raise ValueError("profile argv assignment missing")


def inspect_assembled_packet(packet: Path) -> dict[str, Any]:
    """Decode the built transport without executing it and verify every member."""
    wrapper = packet / "h232_profile_kernel_bundle.py"
    constants = {}
    for node in ast.parse(wrapper.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "PAYLOAD_ZIP_SHA256",
                    "PAYLOAD_FILE_SHA256",
                    "PAYLOAD_B64",
                ):
                    constants[target.id] = ast.literal_eval(node.value)
    archive = base64.b64decode(constants["PAYLOAD_B64"], validate=True)
    assert hashlib.sha256(archive).hexdigest() == constants["PAYLOAD_ZIP_SHA256"]
    with zipfile.ZipFile(io.BytesIO(archive)) as payload:
        assert sorted(payload.namelist()) == sorted(constants["PAYLOAD_FILE_SHA256"])
        assert sum(member.file_size for member in payload.infolist()) < 2_000_000
        for name, expected in constants["PAYLOAD_FILE_SHA256"].items():
            observed = hashlib.sha256(payload.read(name)).hexdigest()
            assert observed == expected == file_sha256(packet / name)
    meta = json.loads((packet / "packet_meta.json").read_text())
    for name, expected in meta["reviewed_hashes"].items():
        assert file_sha256(packet / name) == expected
    assert meta["external_review_sha256"] == file_sha256(packet / "external_review.json")
    manifest = json.loads((packet / "source_archive_manifest.json").read_text())
    with tarfile.open(packet / "source_archive.tar", "r:") as source:
        assert len(source.getmembers()) == manifest["member_count"]
        for member in manifest["members"]:
            stream = source.extractfile(member["path"])
            assert stream is not None
            data = stream.read()
            assert len(data) == member["bytes"]
            assert hashlib.sha256(data).hexdigest() == member["sha256"]
    kernel = json.loads((packet / "kernel-metadata.json").read_text())
    assert kernel["enable_internet"] is False
    assert kernel["enable_gpu"] is True
    assert kernel["is_private"] is True
    return {
        "wrapper_sha256": file_sha256(wrapper),
        "embedded_zip_sha256": constants["PAYLOAD_ZIP_SHA256"],
        "embedded_members": len(constants["PAYLOAD_FILE_SHA256"]),
        "verified_source_archive_members": manifest["member_count"],
        "plan_sha256": file_sha256(packet / "H232_PROFILE_PLAN.json"),
        "review_status_at_inspection": meta["status"],
        "provider_push_authorized_at_inspection": meta["provider_push_authorized"],
        "dataset_sources": kernel["dataset_sources"],
        "private_offline_gpu_metadata": True,
    }


def main() -> None:
    """Independently replay actual profile evidence and fail-closed mutations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = (
        args.main_root
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    )
    v4 = campaign / "h232_conditional_v4/packet"
    v5 = campaign / "h232_conditional_v5/packet"
    primary = campaign / "downloads/h232_conditional_v4/h232_conditional_v4"
    inspected = (
        "profile_worker.py",
        "verified_digest_handoff.py",
        "gpu_memory_domains.py",
        "source_archive.tar",
    )
    hashes_before = {name: file_sha256(v5 / name) for name in inspected}
    assert file_sha256(v5 / "source_archive.tar") == file_sha256(v4 / "source_archive.tar")
    assert file_sha256(v5 / "verified_digest_handoff.py") == file_sha256(
        v4 / "verified_digest_handoff.py"
    )
    assert profile_argv_ast(v4 / "profile_worker.py") == profile_argv_ast(v5 / "profile_worker.py")
    sys.path.insert(0, str(v5))
    source_root = campaign / "downloads/h232_conditional_v4/h232_source_v4"
    sys.path.insert(0, str(source_root / "src"))
    spec = importlib.util.spec_from_file_location("e538_v5_worker", v5 / "profile_worker.py")
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    recorded_gate = json.loads((primary / "profile_gate_result.json").read_text())
    profile = json.loads((primary / "profile_result.json").read_text())
    trace = json.loads((primary / "profile_update_trace.json").read_text())
    probe = worker._probe_source_input_width(SimpleNamespace(N_FEATURES=112))
    tiny_handoff = reproduce_digest_scope(worker)
    raw_path_delta = {name: tiny_handoff["raw"][f"{name}.parquet"] for name in ("train", "valid")}
    inputs: dict[str, Any] = {
        "trace": trace,
        "timing": recorded_gate["timing"],
        "loader_budget": recorded_gate["loader_budget"],
        "trainer_args": frozen_gate_arguments(v4 / "profile_worker.py"),
        "n_in": 211,
        "feature_width_probe": probe,
        "resources": profile["resources_final"],
        "export_probe": recorded_gate["export_shape_probe"],
        "raw_input_identity": recorded_gate["raw_input_identity"],
        "profile_handoff_delta": recorded_gate["profile_digest_handoff_delta"],
        "raw_digest_path_profile_delta": raw_path_delta,
        "elapsed_seconds": recorded_gate["lifetime_projection"][
            "worker_elapsed_before_fit_seconds"
        ],
    }
    cases = []

    def check_case(name: str, candidate: dict[str, Any], expected: str) -> None:
        result = worker.evaluate_conditional_fit_gate(**candidate)
        assert result["status"] == expected, (name, result)
        cases.append({"name": name, "status": result["status"], "reasons": result["reasons"]})

    check_case(
        "actual_v4_shape_trace_resources_with_scoped_path_evidence", copy.deepcopy(inputs), "pass"
    )
    wrong_width = copy.deepcopy(inputs)
    wrong_width["n_in"] = 160
    check_case("160_width_rejects", wrong_width, "reject_no_fit")
    missing_path = copy.deepcopy(inputs)
    del missing_path["raw_digest_path_profile_delta"]["train"]
    check_case("missing_raw_path_evidence_rejects", missing_path, "reject_no_fit")
    no_profile_lookups = copy.deepcopy(inputs)
    for metrics in (
        no_profile_lookups["profile_handoff_delta"],
        *no_profile_lookups["raw_digest_path_profile_delta"].values(),
    ):
        metrics.update({key: 0 for key in metrics})
    check_case("zero_profile_lookups_rejects", no_profile_lookups, "reject_no_fit")
    bad_aux_fallback = copy.deepcopy(inputs)
    bad_aux_fallback["profile_handoff_delta"]["fallback_digest_calls"] = 3
    check_case("auxiliary_miss_without_fallback_rejects", bad_aux_fallback, "reject_no_fit")
    changed_identity = copy.deepcopy(inputs)
    changed_identity["raw_input_identity"]["all_unchanged"] = False
    check_case("changed_raw_stat_identity_rejects", changed_identity, "reject_no_fit")
    skipped_update = copy.deepcopy(inputs)
    skipped_update["trace"]["timed"]["amp_skipped_updates"] = 1
    check_case("amp_skip_rejects_under_current_frozen_policy", skipped_update, "reject_no_fit")
    loader_mismatch = copy.deepcopy(inputs)
    loader_mismatch["loader_budget"]["batch_tensor_bytes"] += 1
    check_case("loader_bytes_mismatch_rejects", loader_mismatch, "reject_no_fit")
    raw_fallback = copy.deepcopy(inputs)
    raw_fallback["raw_digest_path_profile_delta"]["train"] = {
        "verifier_lookups": 1,
        "digest_reuse_hits": 0,
        "digest_reuse_misses": 1,
        "fallback_digest_calls": 1,
    }
    raw_fallback["profile_handoff_delta"].update(
        {"digest_reuse_hits": 1, "digest_reuse_misses": 5, "fallback_digest_calls": 5}
    )
    check_case("one_raw_full_verifier_fallback_remains_valid", raw_fallback, "pass")
    hashes_after = {name: file_sha256(v5 / name) for name in inspected}
    assert hashes_before == hashes_after, "actor changed reviewed source during replay"
    result = {
        "status": "bounded_independent_code_review_pass",
        "v5_file_sha256": hashes_after,
        "scientific_profile_argv_unchanged": True,
        "trainer_source_archive_unchanged": True,
        "assembled_transport": inspect_assembled_packet(v5),
        "tiny_handoff": tiny_handoff,
        "feature_width_probe": probe,
        "cases": cases,
        "limits": [
            "No CUDA profile or fit executed by reviewer",
            "V4 actual aggregate evidence combined with separately executed tiny scoped path evidence; V4 postprofile per-path counters were not persisted",
            "Provider quota, spending authority and final assembled packet sealing remain controller-owned",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "cases": len(cases), "v5_file_sha256": hashes_after},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
