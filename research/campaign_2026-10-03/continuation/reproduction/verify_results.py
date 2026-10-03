from __future__ import annotations

import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from strong_falsifier import INPUTS, OUT, ROOT, features, save, state


def main() -> None:
    """Verify saved results and quantify stacking without fitting or scoring."""
    state(
        "final_verification_and_stacking_quantification",
        pending_work=["quantify stacking", "correction causality", "source checks"],
    )
    sample = json.loads((OUT / "sampling.json").read_text())
    results = json.loads((OUT / "results.json").read_text())
    provenance = json.loads((OUT / "provenance_evaluation.json").read_text())
    counts = {record["row_group"]: record["selected_rows"] for record in provenance}
    overlap = {}
    eval_groups = {g for _, g, _ in sample["evaluation"]["sequences"]}
    eval_components = {c for _, _, c in sample["evaluation"]["sequences"]}
    for fold in [2, 3]:
        manifest = json.loads(
            (ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json").read_text()
        )
        fit = set(manifest["fit"]["valid"])
        groups = sorted(eval_groups & fit)
        components = {c for _, g, c in sample["evaluation"]["sequences"] if g in fit}
        overlap[str(fold)] = {
            "evaluation_sequences_in_parent_fit": len(groups),
            "evaluation_sequences_total": len(eval_groups),
            "evaluation_components_in_parent_fit": len(components),
            "evaluation_components_total": len(eval_components),
            "evaluation_scored_rows_with_targets_in_parent_fit": sum(counts[g] for g in groups),
            "fraction_evaluation_sequences_in_parent_fit": len(groups) / len(eval_groups),
            "evaluation_row_groups_in_parent_fit": groups,
        }
        assert set(groups) == eval_groups
    overlap["interpretation"] = (
        "Calibrator target components are disjoint, but each calibration parent fitted 100% of evaluation sequence targets. This meta-procedure is not nested training independent. Negative can kill candidate; positive requires nested or training-only calibration before adoption."
    )
    save("stacking_limitation.json", overlap)
    corrector = np.load(OUT / "corrector.npz")
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    prefix_results = {}
    for f, g, _ in [sample["evaluation"]["sequences"][0], sample["calibration"]["sequences"][0]]:
        # No target, mask or target-derived class is read for these tests.
        table = parquet.read_row_group(g, columns=INPUTS, use_threads=False)
        x = np.column_stack([table[c].to_numpy() for c in INPUTS])
        targets_index = np.load(ROOT / "cache/valid_targets.npz")["row_group"]
        index = int(np.flatnonzero(targets_index == g)[0])
        parent = np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r")[
            index
        ]
        full_states = features(x).astype(np.float64)
        full_design = np.column_stack([np.ones(len(x)), parent[:, 1], full_states])
        full = (full_design @ corrector["offset_raw_state"]).astype(np.float32)
        assert np.isfinite(full).all()
        parity = (
            full_design @ corrector["offset_raw_state"]
            - np.column_stack(
                [
                    np.ones(len(x)),
                    parent[:, 1],
                    (full_states - corrector["mean"]) / corrector["std"],
                ]
            )
            @ corrector["offset_standardized"]
        )
        assert np.max(np.abs(parity)) < 1e-12
        errors = {}
        for n in [1, 99, 100, 512, 1000, 19999]:
            prefix_design = np.column_stack([np.ones(n), parent[:n, 1], features(x[:n])])
            error = float(
                np.max(
                    np.abs(
                        (prefix_design @ corrector["offset_raw_state"]).astype(np.float32)
                        - full[:n]
                    )
                )
            )
            assert error == 0
            errors[str(n)] = error
        prefix_results[str(g)] = {
            "fold": f,
            "offset_prefix_max_abs_error": errors,
            "normalized_vs_raw_coefficient_parity": float(np.max(np.abs(parity))),
            "full_sequence_corrected_t1_finite": True,
        }
    save("correction_causality.json", prefix_results)
    moments = np.load(OUT / "sequence_moments.npz")
    assert np.array_equal(moments["moments"][:, 0, 0], moments["moments"][:, 1, 0])
    assert np.array_equal(moments["moments"][:, 0, 0], moments["moments"][:, 2, 0])
    assert results["gate_passed"] is False
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.py")}
    original = json.loads((OUT / "identities.json").read_text())["sources"]
    assert (
        source_hashes["executed_source.py"]
        == original[str((OUT / "strong_falsifier.py").relative_to(ROOT))]
    )
    save(
        "final_verification.json",
        {
            "source_hashes": source_hashes,
            "executed_source_identity_preserved": True,
            "t0_saved_moments_exact": True,
            "correction_input_only_prefix_tests": True,
            "commands_exit_codes": {"first_alignment_run": 1, "corrected_fixed_experiment": 0},
            "initial_alignment_failure": "Overstrict model.pt==model_ep3.pt hash assertion; corrected to exact sidecar-declared epoch3 binding before scores. No design changes.",
        },
    )
    with (OUT / "REPORT.txt").open("a") as stream:
        stream.write("\nEXPLICIT STACKING QUANTIFICATION\n")
        for f in ["2", "3"]:
            v = overlap[f]
            stream.write(
                f"Fold{f} calibration parent fitted targets from {v['evaluation_sequences_in_parent_fit']}/{v['evaluation_sequences_total']} evaluation sequences (100%), {v['evaluation_components_in_parent_fit']}/{v['evaluation_components_total']} components, and {v['evaluation_scored_rows_with_targets_in_parent_fit']} evaluation scored rows.\n"
            )
        stream.write(overlap["interpretation"] + "\n")
        stream.write(
            "No independence claim follows from the independent reproducer role. Frozen negative screen rejects this offset candidate; it does not prove all causal state conditioning is useless.\n"
        )
        stream.write(
            "Post-fit input-only correction prefixes passed exactly on one calibration and one evaluation sequence; raw-state and standardized-state coefficients agree within1e-12. Full corrected t1 finite on all20000 rows.\n"
        )
        stream.write(
            "Corrected and raw t0 moments are exactly equal. executed_source.py preserves the exact experiment source hash; strong_falsifier.py is its formatted runnable version.\n"
        )
        stream.write(
            "Initial run exited1 before scoring due to an overstrict checkpoint-byte equality assertion; failure.log retained. Corrected fixed experiment exited0. Both model.pt and model_ep3.pt hashes are saved; predictions bind specifically to model_ep3.pt.\n"
        )
        stream.write(
            "Additional verification command: research/campaign_2026-10-03/continuation/run_cpu.sh scientific-reproducer env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/reproduction/verify_results.py\n"
        )
    state(
        "completed",
        pending_work=[],
        last_failure=None,
        stacking_overlap_fraction=1.0,
        artifacts=sorted(p.name for p in OUT.iterdir() if p.is_file()),
    )


if __name__ == "__main__":
    main()
