# E486 CPU counterfactual scorer source note

This is source and synthetic-test readiness only. The E486 result directory does not exist; no E486 test scoring, GPU work, or fitting was run.

The scorer uses the frozen independent E483 v3 model, reset-after recurrence, official scorer arithmetic, fit-only normalization, and component bootstrap multiplicities. Before E483 export access and test-row loading, it validates the E483 fit spec/attempt, all 18 checkpoints and traversal/initial-state receipts, the official E483 grade spec/attempt/artifacts, source/runtime identities, and the root arithmetic reproduction receipt. A proof-only invocation completed with status e483_proofs_validated_test_unread, 18 verified arms, test_rows_loaded=false, and test_rows_scored=false under one CPU and 2 GiB RAM.

The score CLI requires an E486 root-sealed CPU spec before it can read test rows. That seal must bind the scorer, tests, design, blind critique, addendum, frozen E483 scorer, source/input hashes, one-CPU/six-GiB/2,400-second limits, exact output paths, and output artifact list.

run_crossed_matrix computes all 36 cells and first gates all 18 matched-shape diagonals against the E483 raw selected predictions by exact float32 equality. It then requires target/global scores, per-arm bootstrap values, and all nine D−A paired intervals to reproduce E483 before creating any of the 54 contrasts. One numeric-sorted 2,000-draw multiplicity matrix is shared across all four cells and all six contrasts in each seed/momentum pair. Exports keep selected raw predictions/targets, component and row identities, complete masks, bootstrap draws, and off-diagonal state/prediction extrema.

Synthetic checks ran with one CPU and two GiB RAM using:

    cd research/campaign_2026-10-05/activation_counterfactual
    WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 scripts/limited .venv/bin/python -m unittest -v test_score_counterfactual

Five tests passed. They check baseline/sine forward and carried-state parity with the frozen scorer, the factorial contrast formulas, rejection of one altered diagonal raw prediction, an assembled synthetic 36-cell/54-contrast export, and fail-closed behavior when real scoring is requested without a root seal. Ruff, py_compile, and the E483 proof-only run also passed. The sealed E486 spec and independent source approval do not exist yet; real exposed-panel scoring remains gated.
