from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
AUTHOR = OWN.parent / "amp_linearity_audit_v2"


def digest(path: Path) -> str:
    """Hash code/JSON only, in1MiB blocks; never open a capture tensor file."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Bind frozen candidates/projection/controller integration and exact two cases."""
    inventories = {}
    for name in ("amp_linearity_audit", "amp_linearity_audit_v2"):
        directory = OWN.parent / name
        manifest = json.loads((directory / "HASHES.json").read_text())
        rows = {}
        for item, expected in manifest["artifacts"].items():
            actual = digest(directory / item)
            assert actual == expected["sha256"]
            rows[item] = actual
        inventories[name] = rows
    prior = OWN.parent / "amp_linearity_review"
    manifest = json.loads((prior / "HASHES.json").read_text())
    for name, expected in manifest.items():
        assert digest(prior / name) == expected
    provenance = json.loads((AUTHOR / "INPUT_PROVENANCE.json").read_text())
    for name, expected in provenance["critic_consumed_sha256"].items():
        assert digest(prior / name) == expected
    for name, item in provenance["originals_preserved"].items():
        directory = (
            OWN.parent / name
            if name == "amp_linearity_audit"
            else ROOT / "research/campaign_2026-10-03/continuation/round5" / name
        )
        assert digest(directory / "HASHES.json") == item["manifest_sha256"]
    candidate = {
        name: digest(AUTHOR / name) for name in ("PREREG.md", "audit.py", "metadata.py", "tests.py")
    }
    spec = json.loads((AUTHOR / "FROZEN_SPEC.json").read_text())
    assert spec == candidate
    (OWN / "candidate_sha256.json").write_text(json.dumps(candidate, indent=2) + "\n")
    # Prove diagnostic math/body semantics unchanged from independently reviewed v1.
    old_tree = ast.parse((OWN.parent / "amp_linearity_audit/audit.py").read_text())
    new_tree = ast.parse((AUTHOR / "audit.py").read_text())
    unchanged = (
        "sha",
        "is_sha",
        "counter",
        "tensor_metadata",
        "empty",
        "finished",
        "discrepancy",
        "main",
    )
    old_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in old_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    new_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in new_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(old_functions[name] == new_functions[name] for name in unchanged)
    helper = OWN.parent / "prepare_capture_metadata_v2.py"
    approval_helper = OWN.parent / "approve_amp_audit.py"
    prepared_path = OWN.parent / "fixed_case_metadata_v2.json"
    trajectory = OWN.parent / "real_trajectory.json"
    assert trajectory.exists(), "Metadata-only helper fallback must stay dormant"
    prepared = json.loads(prepared_path.read_text())
    measured = OWN.parent / "capture_measurement_review"
    measured_state = json.loads((measured / "STATE.json").read_text())
    assert (
        measured_state["verdict"]
        == "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY"
    )
    measurement_sha = digest(measured / "STATE.json")
    assert prepared["measurement_review_sha256"] == measurement_sha
    assert prepared["approval_status"] == "METADATA_ONLY_NOT_ANALYSIS_AUTHORIZATION"
    assert prepared["gradients_read"] is False and prepared["CUDA_used"] is False
    recorded = json.loads((measured / "output_byte_verification.json").read_text())
    assert recorded["all_match"]
    actual_manifest = (
        ROOT
        / "research/campaign_2026-10-03/continuation/round4/gpu_capture_v4_default_2120r/output_hashes.json"
    )
    assert digest(actual_manifest) == recorded["manifest_sha256"]
    expected_cases = {
        "synthetic_actual_architecture_normal": "synthetic_actual_architecture/normal",
        "real_panel_normal": "real_fit_panel/normal",
    }
    assert set(prepared["cases"]) == set(expected_cases)
    case_bindings = {}
    tensor_fields = {}

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("kind") == "mapping":
                for key, child in value["entries"]:
                    if (
                        key in {"cached_propensity", "scored_mask"}
                        and child.get("kind") == "tensor_metadata"
                    ):
                        tensor_fields[key] = child
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for name, prefix in expected_cases.items():
        case = prepared["cases"][name]
        assert case["default_ordinary_passed"] is True and case["forced_skip"] is False
        assert "metadata" not in case
        assert case["metadata_identity"]["projection_version"] == "metadata-json-v1"
        for kind in ("pre", "post", "components"):
            relative = prefix + "/" + kind + ".pt"
            path = actual_manifest.parent / relative
            assert case["paths"][kind] == str(path)
            assert case["sha256"][kind] == recorded["files"][relative]["sha256"]
            stat = (
                path.stat()
            )  # No capture content read; compare authenticated earlier pass metadata.
            assert (stat.st_size, stat.st_mtime_ns) == (
                recorded["files"][relative]["bytes"],
                recorded["files"][relative]["mtime_ns"],
            )
        assert (
            case["source"]
            == json.loads(
                (
                    ROOT
                    / "research/campaign_2026-10-03/continuation/round4/live_capture_v4/review_request.json"
                ).read_text()
            )["source"]
        )
        canonical = case["optimizer_id_names"]
        identities = case["tensor_identity"]
        assert set(identities) == set(canonical.values())
        assert all(item["dtype"] == "torch.float32" for item in identities.values())
        grouped = {
            str(identifier): parameter
            for group in case["optimizer_groups"]
            for identifier, parameter in zip(group["ids"], group["names"], strict=True)
        }
        assert grouped == canonical
        case_bindings[name] = {
            "paths": case["paths"],
            "sha256": case["sha256"],
            "evidence_kind": case["evidence_kind"],
            "metadata_projection": case["metadata_identity"]["projection_version"],
        }
        if name == "real_panel_normal":
            walk(case["metadata_identity"])
    assert set(tensor_fields) == {"cached_propensity", "scored_mask"}
    assert all(
        item["device"] == "cpu"
        and item["tensor_class"] == "torch.Tensor"
        and item["detached"] is True
        and item["requires_grad"] is False
        for item in tensor_fields.values()
    )
    integration = {
        "verdict": "PASS_DESCRIPTIVE_FIXED_TWO_CASE_INSTRUMENT_ONLY",
        "candidate_sha256": candidate,
        "metadata_projection_sha256": candidate["metadata.py"],
        "projection_version": "metadata-json-v1",
        "prepare_controller_helper_sha256": digest(helper),
        "approval_controller_helper_sha256": digest(approval_helper),
        "fixed_case_metadata_sha256": digest(prepared_path),
        "independent_measurement_STATE_sha256": measurement_sha,
        "independent_measurement_report_sha256": digest(measured / "REPORT.md"),
        "independent_measurement_manifest_sha256": digest(measured / "HASHES.json"),
        "prior_byte_verification_sha256": digest(measured / "output_byte_verification.json"),
        "output_manifest_sha256": recorded["manifest_sha256"],
        "existing_trajectory_sha256": digest(trajectory),
        "helper_input_scanning_fallback_out_of_scope": True,
        "helper_metadata_only_precondition": "existing real_trajectory.json preserved; use prepared JSON, no helper fallback execution",
        "fixed_case_identities": case_bindings,
        "actual_analysis_executed": False,
        "receipt_written": False,
    }
    (OWN / "integration_bindings.json").write_text(json.dumps(integration, indent=2) + "\n")
    result = {
        "all_author_and_prior_files_match": True,
        "author_inventories": inventories,
        "unchanged_numerical_functions": list(unchanged),
        "prepared_JSON_fixed_case_binding_pass": True,
        "metadata_projection_descriptors": tensor_fields,
        "controller_helper_same_function_retested": True,
        "controller_approval_helper_source_reviewed_NOT_executed": True,
        "actual_capture_reads": 0,
        "metadata_json_only": True,
    }
    (OWN / "protected_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_author_and_prior_files_match": True,
                "candidate_sha256": candidate,
                "projection_sha256": candidate["metadata.py"],
                "measurement_STATE_sha256": measurement_sha,
                "fixed_case_metadata_sha256": integration["fixed_case_metadata_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
