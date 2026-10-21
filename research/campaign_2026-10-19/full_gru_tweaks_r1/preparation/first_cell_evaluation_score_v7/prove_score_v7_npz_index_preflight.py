"""Prove the V7 scorer delta and bind the prior metadata-only NPZ diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[4]
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
V6_CORE = CAMPAIGN / "preparation/first_cell_evaluation_score_v6/first_cell_score_core.py"
V7_CORE = HERE / "first_cell_score_core.py"
DIAGNOSTIC = (
    CAMPAIGN / "records/first_cell_evaluation_v1/score_v3/npz_index_diagnostic_v2/STDOUT.raw.json"
)
DIAGNOSTIC_REVIEW = (
    CAMPAIGN / "records/first_cell_evaluation_v1/score_v3/npz_index_diagnostic_v2/"
    "ACTUAL_NPZ_INDEX_DIAGNOSTIC_METADATA_REVIEW_V2.json"
)
FIXTURE = HERE / "test_score_v7_npz_index_preflight.py"
OUTPUT = HERE / "SCORE_V7_NPZ_INDEX_PREFLIGHT_PROOF.json"


def digest(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(REPOSITORY)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def span(text: str, start: str, end: str) -> str:
    first = text.index(start)
    last = text.index(end, first)
    return text[first:last]


def main() -> None:
    v6_bytes = V6_CORE.read_bytes()
    v7_bytes = V7_CORE.read_bytes()
    v6_text = v6_bytes.decode("utf-8")
    v7_text = v7_bytes.decode("utf-8")

    v6_block = span(v6_text, "def _preflight_npz_members(", "def observe_registered_source(")
    v7_block = span(v7_text, "def _preflight_npz_members(", "def observe_registered_source(")
    restored = v7_text.replace("import ast\n", "", 1)
    restored = restored.replace("MAX_NPY_HEADER_BYTES = 8_192\n", "", 1)
    restored = restored.replace(
        """            _preflight_npz_members(
                source_path,
                {"row_group", "seq_ix"},
                MAX_MEMBERSHIP_ARRAY_BYTES,
                selected_only=True,
            )""",
        '            _preflight_npz_members(source_path, {"row_group", "seq_ix"}, 2)',
        1,
    )
    restored = restored.replace(v7_block, v6_block, 1)
    if restored != v6_text:
        raise SystemExit("V7 inverse does not restore the exact V6 core bytes")

    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic_review = json.loads(DIAGNOSTIC_REVIEW.read_text(encoding="utf-8"))
    diagnostic_pin = digest(DIAGNOSTIC)
    diagnostic_review_pin = digest(DIAGNOSTIC_REVIEW)
    if diagnostic_pin != {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/score_v3/npz_index_diagnostic_v2/STDOUT.raw.json",
        "bytes": 5363,
        "sha256": "8bc92bcf227fc0a8e35f52b15c2092244a5077fb4c5a8f1706804b70c5380e81",
    }:
        raise SystemExit("bounded NPZ diagnostic raw differs from its pinned capture")
    if diagnostic_review_pin != {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/score_v3/npz_index_diagnostic_v2/ACTUAL_NPZ_INDEX_DIAGNOSTIC_METADATA_REVIEW_V2.json",
        "bytes": 7548,
        "sha256": "3c326b532be2f7077007d4b77e14435ecee15b2273c6641fb1d13af990880810",
    }:
        raise SystemExit("independent NPZ diagnostic review differs from its exact pin")
    if diagnostic["array_payloads_decoded"] is not False:
        raise SystemExit("diagnostic reports decoded array payloads")
    if diagnostic["nonselected_members_opened"] != 0 or diagnostic["unit_queries"] != 0:
        raise SystemExit("diagnostic scope differs from its bounded metadata contract")
    if diagnostic_review["status"] != "PASS_METADATA_ONLY_BOUNDED_INDEX_DIAGNOSTIC_NO_UNIT_QUERY":
        raise SystemExit("diagnostic metadata review is not the expected independent pass")
    expected_shapes = {
        "cache/train_targets.npz": (10_606, 84_976),
        "cache/valid_targets.npz": (1_873, 15_112),
    }
    selected_metadata: dict[str, list[dict[str, object]]] = {}
    for archive in diagnostic["archives"]:
        relative = archive["relative"]
        if relative not in expected_shapes:
            continue
        shape_value, member_bytes = expected_shapes[relative]
        rows = []
        for member in archive["members"]:
            if member["name"] not in {"row_group.npy", "seq_ix.npy"}:
                continue
            header = member["selected_npy_header"]
            if (
                header["version"] != [1, 0]
                or header["fields"]
                != {"descr": "<i8", "fortran_order": False, "shape": [shape_value]}
                or member["expanded_bytes"] != member_bytes
                or member["selected_npy_payload_decoded"] is not False
            ):
                raise SystemExit(
                    f"unexpected selected index metadata for {relative}/{member['name']}"
                )
            rows.append(
                {
                    "name": member["name"],
                    "expanded_bytes": member["expanded_bytes"],
                    "header_version": header["version"],
                    "dtype": header["fields"]["descr"],
                    "fortran_order": header["fields"]["fortran_order"],
                    "shape": header["fields"]["shape"],
                    "payload_decoded": False,
                }
            )
        if {row["name"] for row in rows} != {"row_group.npy", "seq_ix.npy"}:
            raise SystemExit(f"missing selected index metadata for {relative}")
        selected_metadata[relative] = rows

    unchanged_modules = all(
        (HERE / name).read_bytes()
        == (HERE.parent / "first_cell_evaluation_score_v6" / name).read_bytes()
        for name in (
            "first_cell_score_contract.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
        )
    )
    if not unchanged_modules:
        raise SystemExit("a scorer module outside the declared core changed")

    proof = {
        "schema": "fg19.first-cell-score-v7-npz-index-preflight-proof.v1",
        "status": "PASS_EXACT_V6_INVERSE_AND_BOUNDED_SELECTED_INDEX_METADATA_V7",
        "source_delta": {
            "base_core": digest(V6_CORE),
            "candidate_core": digest(V7_CORE),
            "exact_inverse_restores_base_bytes": True,
            "only_changed_production_module": "first_cell_score_core.py",
            "other_scorer_modules_byte_identical": unchanged_modules,
            "behavior": "Target/index NPZs cap and inspect only exact row_group.npy and seq_ix.npy members with bounded headers. Existing chains/folds archives still apply the unchanged all-member expanded-size cap.",
        },
        "actual_metadata_diagnostic": {
            "raw": diagnostic_pin,
            "independent_metadata_review": diagnostic_review_pin,
            "review_status": diagnostic_review["status"],
            "nonselected_members_opened": diagnostic["nonselected_members_opened"],
            "array_payloads_decoded": diagnostic["array_payloads_decoded"],
            "registered_stat_join": diagnostic_review["registered_stat_join"],
            "archive_stat_identities": [
                {"relative": item["relative"], **item["identity"]}
                for item in diagnostic["archives"]
            ],
            "selected_index_headers": selected_metadata,
            "large_unselected_members": {
                "train_y_bytes": 1_696_960_128,
                "train_mask_bytes": 212_120_128,
                "valid_y_bytes": 299_680_128,
                "valid_mask_bytes": 37_460_128,
            },
        },
        "focused_synthetic_fixture": digest(FIXTURE),
        "scope": {
            "actual_npz_archives_opened": False,
            "actual_array_or_model_payloads_read": False,
            "b_access_or_action": False,
            "failed_score_v2_unit_queried_or_retried": False,
            "only_new_synthetic_fixture_and_scoped_ruff_are_run": True,
        },
    }
    OUTPUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
