from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROUND = OWN.parent
AUTHOR = ROUND / "amp_linearity_audit_v2"
PRIOR = ROUND / "amp_linearity_review_v2"
UNIT = "wnn-cpu-0-controller-1791055382-3566680"
BINDINGS: dict[str, str] = {}


def sha(path: Path) -> str:
    """Hash only bounded metadata/code/log files, never tensor containers."""
    assert path.suffix != ".pt" and path.stat().st_size < 4 * 1024**2
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    BINDINGS[str(path.resolve())] = result.hexdigest()
    return result.hexdigest()


def load(path: Path) -> dict:
    """Read JSON after binding bytes; heterogeneous schema checked by callers."""
    sha(path)
    return json.loads(path.read_text())


def close(observed: float, expected: float) -> None:
    """Allow scalar reduction/reporting roundoff, not gradient acceptance."""
    assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-24)


def main() -> None:
    """Check authorized output provenance and arithmetic without tensor reads."""
    result = load(AUTHOR / "actual_discrepancy_2222.json")
    approval_path = ROUND / "amp_audit_approved_2222.json"
    approval = load(approval_path)
    integration = load(PRIOR / "integration_bindings.json")
    candidates = load(PRIOR / "candidate_sha256.json")
    prior_manifest = load(PRIOR / "HASHES.json")
    assert prior_manifest == approval["independent_audit_review_bindings"]
    for name, expected in prior_manifest.items():
        assert sha(PRIOR / name) == expected
    assert candidates == approval["candidate_sha256"] == integration["candidate_sha256"]
    assert set(candidates) == {"PREREG.md", "audit.py", "metadata.py", "tests.py"}
    for name, expected in candidates.items():
        assert sha(AUTHOR / name) == expected
    for name, record in load(AUTHOR / "HASHES.json")["artifacts"].items():
        assert sha(AUTHOR / name) == record["sha256"]
    assert result["approval_sha256"] == sha(approval_path)
    assert result["prereg_sha256"] == candidates["PREREG.md"]
    assert approval["schema_version"] == 4
    assert approval["controller_analysis_authorized"] is True
    assert approval["independent_measurement_review_passed"] is True
    fixed = load(ROUND / "fixed_case_metadata_v2.json")
    assert fixed["cases"] == approval["cases"]
    assert sha(ROUND / "fixed_case_metadata_v2.json") == approval["fixed_metadata_sha256"]
    measurement = ROUND / "capture_measurement_review"
    checks = {
        "fixed_case_metadata_sha256": ROUND / "fixed_case_metadata_v2.json",
        "independent_measurement_STATE_sha256": measurement / "STATE.json",
        "independent_measurement_report_sha256": measurement / "REPORT.md",
        "independent_measurement_manifest_sha256": measurement / "HASHES.json",
        "prior_byte_verification_sha256": measurement / "output_byte_verification.json",
        "existing_trajectory_sha256": ROUND / "real_trajectory.json",
        "prepare_controller_helper_sha256": ROUND / "prepare_capture_metadata_v2.py",
        "approval_controller_helper_sha256": ROUND / "approve_amp_audit.py",
        "output_manifest_sha256": ROUND.parent
        / "round4/gpu_capture_v4_default_2120r/output_hashes.json",
    }
    for key, path in checks.items():
        assert sha(path) == integration[key]
    assert (
        approval["independent_measurement_review_receipt_sha256"]
        == integration["independent_measurement_STATE_sha256"]
    )
    names = {"synthetic_actual_architecture_normal", "real_panel_normal"}
    assert set(result["cases"]) == set(approval["cases"]) == names
    totals = {}
    for name, case in result["cases"].items():
        declared = approval["cases"][name]
        assert declared["forced_skip"] is False and declared["default_ordinary_passed"] is True
        assert declared["paths"] == integration["fixed_case_identities"][name]["paths"]
        assert (
            case["approved_file_sha256"]
            == declared["sha256"]
            == integration["fixed_case_identities"][name]["sha256"]
        )
        assert case["status"] == "DESCRIPTIVE_DIFFERENT_OPERATION_DISCREPANCY_ONLY"
        for key in (
            "pre_values_or_inputs_accessed",
            "coefficient_or_scale_reapplied",
            "acceptance_bound_inferred",
            "update_attribution",
            "scientific_conflict_or_score_claim",
        ):
            assert case[key] is False
        assert case["calibrated_allowance"] is None
        assert case["max_converted_workspace_bytes"] == 8 * 131072 * 8 == 8388608
        assert case["max_converted_workspace_bytes"] <= 32 * 1024**2
        rows = case["parameters"]
        assert len(rows) == 14 and len({row["name"] for row in rows}) == 14
        assert {row["name"] for row in rows} == set(declared["tensor_identity"])
        for row in rows:
            close(
                row["metrics"]["elements"],
                math.prod(declared["tensor_identity"][row["name"]]["shape"]),
            )
        for metrics in [*case["scopes"].values(), *(row["metrics"] for row in rows)]:
            for key, value in metrics.items():
                if type(value) in (int, float):
                    assert math.isfinite(value) and value >= 0, key
            assert metrics["zero_reference"] is False and metrics["reference_l2"] > 0
            close(metrics["relative_error_l2"], metrics["error_l2"] / metrics["reference_l2"])
            close(metrics["error_rms"], metrics["error_l2"] / math.sqrt(metrics["elements"]))
        for scope, metrics in case["scopes"].items():
            selected = [
                row["metrics"]
                for row in rows
                if scope == "total"
                or scope in (row["optimizer"], row["serialized_group"], row["name_scope"])
            ]
            assert selected
            for key in ("elements", "active_parameters", "unused_parameters"):
                assert metrics[key] == sum(row[key] for row in selected)
            for key in ("error_l2", "reference_l2", "component_fp64_sum_l2"):
                close(metrics[key] ** 2, math.fsum(row[key] ** 2 for row in selected))
            close(metrics["error_max_abs"], max(row["error_max_abs"] for row in selected))
        totals[name] = case["scopes"]["total"]
        assert totals[name]["elements"] == 981015 and totals[name]["active_parameters"] == 14
    logs = ROUND.parent / "logs"
    terminal = logs / f"{UNIT}.terminal.log"
    sha(terminal)
    text = terminal.read_text()
    assert (
        "audit.py --approval research/campaign_2026-10-03/continuation/round6/amp_audit_approved_2222.json"
        in text
    )
    assert "actual_discrepancy_2222.json" in text and "1.293s wall clock" in text
    assert "286.5M memory peak" in text
    commands = logs / "cpu_commands.log"
    matching = [line for line in commands.read_text().splitlines() if f"unit={UNIT} " in line]
    assert len(matching) == 1 and matching[0].endswith("exit=0")
    (OWN / "executed_command.txt").write_text(matching[0] + "\n" + text)
    (OWN / "candidate_sha256.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (OWN / "provenance.json").write_text(
        json.dumps(
            {
                "sha256": BINDINGS,
                "approved_capture_declarations_not_rehashed": {
                    name: approval["cases"][name]["sha256"] for name in names
                },
                "actual_command": matching[0],
            },
            indent=2,
        )
        + "\n"
    )
    verdict = "PASS_AUTHORIZED_FIXED_TWO_CASE_DESCRIPTIVE_RESULT"
    findings = {
        "verdict": verdict,
        "totals": totals,
        "finite_aggregates": True,
        "all_scope_scalar_identities": True,
        "candidate_and_integration_hashes_verified": True,
        "frozen_author_and_prior_review_preserved": True,
        "converted_workspace_bytes": 8388608,
        "actual_gradient_reads_by_reviewer": 0,
        "capture_replay": False,
        "blockers": [],
    }
    (OWN / "results.json").write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
