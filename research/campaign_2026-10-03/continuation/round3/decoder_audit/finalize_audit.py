from __future__ import annotations

import json
import logging
import re
import time

from audit_common import FROZEN, OWN, ROOT, digest, phase, save_json

LOGGER = logging.getLogger(__name__)


def timed_cost(name: str) -> dict[str, object]:
    """Extract measured process cost from the owned GNU time log."""
    text = (OWN / name).read_text()
    result = {}
    for field, pattern in {
        "user_seconds": r"User time \(seconds\): ([0-9.]+)",
        "system_seconds": r"System time \(seconds\): ([0-9.]+)",
        "peak_rss_kib": r"Maximum resident set size \(kbytes\): ([0-9]+)",
        "exit_status": r"Exit status: ([0-9]+)",
    }.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Missing resource field {field} in {name}")
        result[field] = float(match.group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    if not elapsed:
        raise ValueError(f"Missing elapsed cost in {name}")
    parts = [float(part) for part in elapsed.group(1).split(":")]
    result["wall_seconds"] = sum(
        value * 60**index for index, value in enumerate(reversed(parts))
    )
    result["cpu_seconds"] = result["user_seconds"] + result["system_seconds"]
    return result


def main() -> None:
    """Bind measurements, sample identities, claims and final artifact hashes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = json.loads((OWN / "raw_replay_results.json").read_text())
    synthetic = json.loads((OWN / "synthetic_results.json").read_text())
    records = json.loads((OWN / "all_origins_raw_replay.json").read_text())
    selection = json.loads((FROZEN / "selection.json").read_text())
    table = json.loads((FROZEN / "decoder_tables.json").read_text())
    fit = json.loads((FROZEN / "fit_results.json").read_text())
    provenance = json.loads((OWN / "input_provenance_checks.json").read_text())
    links = {
        "decoder_selection_sha_matches": table["selection_sha256"]
        == digest(FROZEN / "selection.json"),
        "fit_result_table_sha_matches": fit["table_sha256"]
        == digest(FROZEN / "decoder_tables.json"),
        "supplied_fit_test_components_disjoint": not bool(
            set(selection["fit_components"]) & set(selection["test_components"])
        ),
        "supplied_fit_test_groups_disjoint": not bool(
            set(selection["fit_groups"]) & set(selection["test_groups"])
        ),
        "all_table_classes_valid_positive_quantum_indices": all(
            row[1] in range(4) for row in table["spreads"]
        ),
        "all_table_spreads_positive_integer": all(
            isinstance(row[2], int) and row[2] >= 1 for row in table["spreads"]
        ),
        "1393_unique_origin_group_step_keys": len(
            {(row["group"], row["step"]) for row in records}
        )
        == len(records)
        == 1393,
        "prereg_falsifier_hash_unchanged": digest(OWN / "PREREG.json")
        == synthetic["prereg_sha256"]
        == provenance["own_prereg_sha256"],
        "all_origin_saved_fields_verified": all(
            value == 0 for value in raw["recorded_field_mismatches"].values()
        ),
        "all_raw_group_identity_checks_verified": all(
            all(
                value
                for key, value in check.items()
                if key
                in [
                    "all_seq_ids_equal",
                    "raw_step_order_ok",
                    "metadata_component_ok",
                    "metadata_origin_seq_ids_ok",
                    "recorded_array_hash_matches",
                    "sample_selection_replays",
                    "endpoint_count_matches",
                ]
            )
            for check in raw["group_checks"]
        ),
    }
    save_json("final_link_checks.json", links)
    if not all(links.values()):
        raise ValueError("A final frozen linkage/identity check failed")
    cost = {
        "raw_replay_process": timed_cost("raw_replay_resource.log"),
        "synthetic_process": timed_cost("synthetic_resource.log"),
        "raw_internal": raw["cost"],
        "synthetic_internal": synthetic["cost"],
        "resource_profile": "run_cpu.sh round3-hft;1core2GiB900s;CUDA unavailable",
        "gpu_seconds": 0,
        "cost_limit": "Exact process cost is saved for decisive probes; shell/source inspection, summary rendering, formatting and finalization are not included in their CPU sum.",
        "pass_elapsed_seconds_since_prereg_file": time.time()
        - (OWN / "PREREG.json").stat().st_mtime,
    }
    save_json("COST.json", cost)
    claims = {
        "supported": [
            "Independent raw replay reproduces417/417 mapped outside bound violations;28/445 outside unknown; all1393 origins retained.",
            "Entry-inclusive extrema returns in the frozen positive-scale spread coordinate cannot match those417 losers at any defined stopping horizon.",
            "Changing reference has252/252 exact supervised differences and0/252 physical moves; fixed-reference control252/252 both. Spread exact in both.",
            "All12 recorded raw-array hashes and13 permitted frozen artifact hashes match; per-origin target/class/q/sample fields have zero mismatches.",
        ],
        "contradicted": [
            "Interior-difference exactness by itself identifies physical quote movements.",
            "The bound unconditionally rules out entry-excluding extrema, execution-price or hybrid laws at any horizon.",
            "All445 outside origins have known spread or clipping always preserves a raw contradiction.",
        ],
        "unknown": [
            "Physical entry-spread identification and actual supplied-data future reference/target generator.",
            "The28 missing-map outside origins; actual future stops beyond censored decoded paths; hybrid law transport.",
            "Exact fit-time source identity: decoder_tables embeds a different source hash from the final verified decoder.py.",
            "Independent rediscovery of paired components or campaign-wide sample exposure; neither is claimed.",
        ],
        "hybrid_implication": "Losing execution returns can escape the lower bound, so hybrids remain logically admissible; physical interpretation and exact raw pair transport need separate evidence. No hybrid outputs read.",
    }
    save_json("CLAIMS.json", claims)
    commands = [
        line
        for line in (OWN / "COMMANDS.sh").read_text().splitlines()
        if line.startswith("research/")
    ]
    phase(
        "complete_report_frozen",
        commands=commands,
        pending=[],
        last_failure=None,
        source_read_boundary="Independent formula commitment preceded implementation inspection; no actor code imported/executed; no hybrid/usage/live source read.",
        figures_inspected=["audit_coverage_and_gauge.png"],
        output_origin_count=1393,
        mapped_outside_violations=417,
        outside_unknown=28,
        table_dependence="Frozen round2 TRAIN-supervised spread/class table; no refitting or physical identification.",
    )
    own_files = [
        path
        for path in OWN.iterdir()
        if path.is_file() and path.name not in ["PROVENANCE.json", "finalize.log"]
    ]
    report_provenance = {
        "own_artifacts": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(own_files)
        },
        "frozen_inputs": provenance,
        "data_metadata_only": {
            "path": "datasets/train.parquet",
            "bytes": (ROOT / "datasets/train.parquet").stat().st_size,
        },
        "claims_depend_on_supervised_table": True,
        "code_changes_after_execution": "Formatting only; independent formulas, sample selection, constants and synthetic procedure unchanged.",
        "prior_artifacts_preserved": True,
        "active_log_excluded": "finalize.log is written after inventory creation and is intentionally not hashed.",
    }
    save_json("PROVENANCE.json", report_provenance)
    LOGGER.info("FINAL_LINKS %s", json.dumps(links))
    LOGGER.info("COST %s", json.dumps(cost))


if __name__ == "__main__":
    main()
