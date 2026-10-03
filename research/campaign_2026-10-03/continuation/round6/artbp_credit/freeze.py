from __future__ import annotations

import json
from pathlib import Path

from run_probe import HERE, save, sha


def main() -> None:
    """Bind final completed logs and preserve all older index/month artifacts."""
    bindings = json.loads((HERE / "pre_experiment_bindings.json").read_text())
    if any(sha(Path(name)) != expected for name, expected in bindings.items()):
        raise ValueError("preregistered source/input changed")
    old_hashes = HERE.parent / "HASHES.json"
    old_manifest = json.loads(old_hashes.read_text())
    for name, record in old_manifest["outputs"].items():
        if sha(Path(name)) != record["sha256"]:
            raise ValueError("protected landed output changed: " + name)
    month_hashes = HERE.parent / "month_OPTIONS_HASHES.json"
    month_manifest = json.loads(month_hashes.read_text())
    for name, record in month_manifest["outputs"].items():
        if sha(Path(name)) != record["sha256"]:
            raise ValueError("protected month output changed: " + name)
    state = json.loads((HERE / "STATE.json").read_text())
    state.update(
        {
            "initial_lint_failure": "unused Fraction import; retained lint_attempt1.log and original source copy; corrected BEFORE experiment",
            "scientific_experiment_executions": 1,
            "fixture_probability_tolerance_changes": False,
            "protected_old_index_month_hashes_verified": True,
            "final_completed_log_sha256": sha(HERE / "run.log"),
        }
    )
    save("STATE.json", state)
    report = (HERE / "REPORT.md").read_text()
    report += (
        "\nFinalization note: the first lint command stopped before any experiment due "
        "to an unused Fraction import. lint_attempt1.log and run_probe_lint_attempt1.py.txt "
        "preserve it; only the unused import was removed. Prereg/fixtures/probabilities/"
        "thresholds were unchanged. Exactly one scientific enumeration ran. Final "
        "hashes include the fully completed run.log (after runner stdout), and prior "
        "landed index/report and month option output hashes are verified unchanged.\n"
    )
    (HERE / "REPORT.md").write_text(report)
    protected = [old_hashes, month_hashes]
    protected += [Path(name) for name in old_manifest["outputs"]]
    protected += [Path(name) for name in month_manifest["outputs"]]
    save(
        "HASHES.json",
        {
            "algorithm": "sha256",
            "manifest_self_excluded": True,
            "artifacts": {
                path.name: sha(path)
                for path in HERE.iterdir()
                if path.is_file() and path.name != "HASHES.json"
            },
            "external_input_bindings": {
                name: value
                for name, value in bindings.items()
                if not Path(name).is_relative_to(HERE)
            },
            "protected_old_metadata": {str(path): sha(path) for path in protected},
            "all_originals_unmodified": True,
            "scientific_executions": 1,
        },
    )
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "final_logs_bound": True,
                "prior_index_month_preserved": True,
                "actual_data_reads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
