# Neighboring Kalman screens

These measurements motivate a different hypothesis; they do **not** measure the innovation-gain GRU in the parent packet.

| Local screen | Observation | Evidence scope |
|---|---|---|
| Fixed output local-level Kalman filters | Parent WP `0.6812265`; fixed Q/R filter `0.6755207` (Δ `−0.005706`); Q/R estimated from the first99 rows `0.6647620` (Δ `−0.016465`). | One repeatedly inspected F0 panel. This rejects those output-filter recipes, not learned state updates. |
| Groupwise fast/slow innovation-energy inputs, seed3 | Active `0.67986413`; equal-width zero control `0.67981250`; Δ `+0.00005163`; component interval `[−0.00013143,+0.00027018]`. | One exposed F0 panel, same incremental parent, 32 fit components, 210 updates. |
| Same groupwise inputs, seed4 | Active `0.68037958`; zero control `0.68039346`; Δ `−0.00001387`; interval `[−0.00017548,+0.00012404]`. | Same F0 panel, fit components and parent, new optimizer seed; effect changes sign. Descriptive mean delta across two seeds is `+0.00001888`, not a fit-variance estimate. |
| Fixed local-trend input replay | First selected-panel Δ `+0.000338`; expanded 141-group panel Δ `+0.00011632`, conditional component interval `[−0.00007495,+0.00028789]`. | Frozen-GRU feature replacement, not a trained observer or student-transfer result; exposed F0. |

The local seed-4 result note and execution receipt live in the main worktree at `research/campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_SEED4_R1.md` and `execution_receipt_seed4_r1.json`. They are preserved there with hashes, masks, commands, resource peaks and the grader repair. The source working tree that produced those append-feature fits has not been committed or copied into this B300 draft. Treat the numbers as provisional neighboring evidence, not a reproducible source release or a clean-CV claim. No quality conclusion here is upgraded by the present handoff.
