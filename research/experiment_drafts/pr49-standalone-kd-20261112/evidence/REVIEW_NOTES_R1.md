# Standalone KD source and CPU review

The source implementation is committed as `33542502a1404a2843359b3999fa34f3a157bf3f`, based on `390c846cbdacd8e0e06e6079dd3b04312fa6bcc5`; the implementation commit is `1e4306b9bf7b32b54606a2f949aa47fab4803051`, followed by formatting and import-style-only commits. The original source worktree remains untouched. The standalone-KD panel's earlier R1 overlay fingerprint is historical; it predates the final compatibility fixes and must not be used as the final source identity. `CPU_CHECKS_R1.json` records the final commit, commands, exit codes and exact config hashes.

## CPU validation

The new focused test file passes **17/17** tests. The bounded regression group passes **71 tests** and has one failure in `test_legacy_config_identities_match_completed_producer_runs`. I reproduced the identical failure in a separate clean detached worktree at the exact base commit `390c846cbdacd8e0e06e6079dd3b04312fa6bcc5`: the loaded `labels_s101101.json` hashes to `c5f98c986292570b3b29494555946021522fc9ee54fe58f729af57c6f16cff54`, while the fixture expects `d3ef8df541ec5ce9ac4f68b816e5ac61a59e7c1d0a181954f0fb6bdd4f8409ef`. It is a base-reproduced failure.

`check_training_structure.py` exits 1 with two errors in unchanged `sequence_config.py` (module length and `load_config` length). The same check at base commit reports those two plus `sequence_loop.py` exceeding 300 lines; the extension moves the shared window-denominator reducer out of that module. Ruff check and format check both pass at final source commit `33542502a1404a2843359b3999fa34f3a157bf3f`. The final combined pytest run preceded the formatting/import-only commits; the focused KD suite had passed immediately before them.

The pure-label monitor endpoint, error messages, and admission error wording were restored to their pre-extension contracts. The existing loop-level `_window_denominators` monkeypatch hook was restored, with the new shared reducer accepting that hook. Existing standalone-label, since-weighting, blend, monitor, sampling, and reference-cache regressions are included in the 71 passing tests.

## Four-cell config packet

The real K16/2×256, six-epoch L4 configs were production-loaded and passed `validate_fit_reference_inputs` using an explicit teacher-cache path for every cell, including the lambda-zero controls. Each matched seed pair differs only in `kd_weight` (0 versus 0.2). The shared numerator weight is `abs(clamp(y,-2,2))`, the denominator is its sum over both targets across the complete actual optimizer window, and lambda scales only the KD numerator. The teacher target is clipped to `[-2,2]`; monitor output is raw student z. The cache must have exactly the 10,325 official fitting members, with validation absent. Parent/alias provenance is recorded in the panel manifest; it proves normalized logical checkpoint-field equality, not prediction equality.

The panel is **not profile- or fit-qualified**. Native teacher-cache membership preflight, a 30-update profile, reload validation, and the fixed six-epoch fits remain the operator's gates. No native/cache-consumer execution was performed by this actor.
