# Point 5 L4: optimizer updates and runtime context

Status: implementation and bounded CPU checks complete. No fits, GPU runs,
hardware profiles, throughput measurements, SSH, or authentication were done.

## Update measurements

`optimizer_update_diagnostics_every` is opt-in and absent from historical
defaults. Its one-based sample is update 1 and then every Nth update. Sampled
records include actual pre-update weights, the resulting parameter deltas,
pre- and post-clipping group gradient norms, the learning rates used for the
step, and available AdamW `exp_avg` or Muon momentum state norms and counts.
The `delta_weight_over_pre_weight_norm` denominator floor is recorded as
`1e-12`, along with a zero-denominator count. The probe snapshots weights only
on sampled steps; it does not call the model, draw random values, modify
gradients, or alter optimizer and scheduler state.

The L4 runner stores the nested sample in rank JSONL. The aggregate monitor
flattens it under `optimizer_update/...` and merges it into the existing
`diagnostics` event when the activation probe also fires. This retains update 1
even when the regular `train_step` stream is sampled less often. Event-budget
preflight counts unique merged hook and optimizer sample steps; resource samples
remain tied to the activation-probe cadence.

New learning-rate controls are also opt-in. AdamW parameter groups split by
`head.*` and body names only when a head/body factor is supplied. Factors multiply
the base group rate before `LambdaLR` construction; the scheduler then advances
those real group rates. When the controls are absent, the historical flat AdamW
group construction remains in place. The L4 preset explicitly sets the AdamW,
head, and body factors to 1.0, so its rates match the pinned schedule.

## Runtime context and candidate identity

The future L4 recipe declares `gradient_horizon_rows: 1000`; loading rejects a
declaration that differs from `chunk_rows`. Its identity-bound runtime contract
records 20,000 rows per sequence, persistent GRU hidden state, `chunk_rows=1000`
as the actual truncated-BPTT horizon, `accumulate_chunks=1`, detached state
carried across chunks and optimizer updates within a sequence minibatch, and
reset at the next minibatch. It records no attention window for the GRU; its
`context_rows=1` setting is not a recurrent receptive-field or horizon value.
Transformer, local-GRU, GALAM, Mamba, LSTM, and EMA state descriptions are
architecture-specific so that an attention window is not attached to
recurrent or state-space models.

The P5 candidate is still based on the completed L4 raw112 recipe. It retains
the same seed `101101`, six epochs, batch size 32, accumulation 1, chunk length
1000, peak LR `0.0003`, `warmup_cosine_v2`, AdamW betas, weight decay, model
shape, objective, fit mask, and normalization scope. P5 adds only the explicit
runtime horizon, optimizer-probe interval, and neutral LR factors on top of the
Point 4 prospective objective and reporting fields. The normalized candidate
config SHA-256 is
`830b9536c90ce4b951b5aec0c781d794f3159f4caf6d85acd2b7408058af516c`.
The earlier Point 4 receipt's SHA describes that point's pre-P5 config.

## Verification

Python: `C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
All checks used one CPU thread with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and explicit `PYTHONPATH=src;.`.

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONPATH='src;.'; C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest tests/test_sequence_l4_update_telemetry.py tests/test_sequence_runtime_contract.py tests/test_sequence_objective_mode.py tests/test_sequence_launch_monitoring_policy.py tests/test_sequence_monitoring.py tests/test_sequence_muon.py tests/test_sequence_fit_monitor_setup.py -q
# 99 passed

$ruff='C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe'; $files=@('src/wnn/training/sequence_config.py','src/wnn/training/sequence_config_finalize.py','src/wnn/training/sequence_config_validation.py','src/wnn/training/sequence_fit_admission.py','src/wnn/training/sequence_runtime_contract.py','src/wnn/training/sequence_fit_utils.py','src/wnn/training/sequence_optimizer_factory.py','src/wnn/training/sequence_update.py','src/wnn/training/sequence_loop.py','src/wnn/training/sequence_monitor.py','src/wnn/training/sequence_monitor_diagnostics_events.py','src/wnn/training/sequence_fit.py','src/wnn/training/sequence_fit_monitor_setup.py','src/wnn/training/sequence_monitor_budget.py','src/wnn/training/sequence_optimizer_controls.py','src/wnn/training/sequence_update_telemetry.py','tests/test_sequence_runtime_contract.py','tests/test_sequence_l4_update_telemetry.py','tests/test_sequence_launch_monitoring_policy.py','tests/test_sequence_objective_mode.py','tests/test_sequence_fit_monitor_setup.py'); & $ruff format --check $files
# 21 files already formatted
& $ruff check $files
# All checks passed

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe scripts/check_training_structure.py --root .
# 162 files, 0 errors; module limit 300 lines, function limit 200 lines

git diff --check
# clean (Git reported existing LF-to-CRLF working-copy notices for root-owned files)
```

The L4 finisher parity case runs two CPU optimizer updates with a nonzero rate
after warmup, comparing probe enabled and disabled. It checks RNG state,
pre-update gradients, resulting parameters, optimizer state, and scheduler
state. Payload oracles compare the logged update norms and ratios with cloned
weights, check clipping and exp-average counts/norms, and confirm the reported
rate is captured before the scheduler advances. Additional tests check neutral
and nontrivial head/body rates through a scheduler step and optimizer/scheduler
save-load, preserved betas/weight decay/parameter ownership, and rejection of
unknown roles. Runtime tests cover the pinned GRU contract, an LSTM without a
false attention window, and a Transformer window distinct from TBPTT. These are
software checks only, not fits or measured hardware behavior.

## Next-run reporting

The root-owned Torch-free view adapter is shared by the live emitter and native
B300 mirror. It retains sampled update 1 despite ordinary train-step thinning,
keeps applied and next-step LR separate, and preserves zero values. Relative
delta is omitted where the recorded denominator floor was used; absolute delta
and the floor flag remain visible. An absent optimizer state is not rendered as
a zero norm. Group role and head/body partition accompany the series.

Root integration checks passed: 10 tests across the optimizer view, dashboard
wrapper and RPC mirror; the JavaScript invariant executes 48 metric groups and
checks 29 sentinel keys for visibility and unit separation. Ruff check/format
and JavaScript syntax checks passed. Seven optimizer plot families were rendered
as 14 PNG/SVG files from the actual helper's small CPU synthetic fixture and
visually inspected. That fixture uses no competition data and is not a quality
experiment. The independent review is recorded under
`Documents/Codex/artifacts/training-diagnosis-20261108/repair-independent-review.md`.

Keep the six configured epochs and final-epoch `model.pt` as the fixed endpoint.
Report all-epoch full official-valid pooled WP and target-specific t0/t1 WP on
the 1,873 official valid sequences and 3,528,273 `is_scored` rows. The 128-member
hash-ranked fitting diagnostic remains a separate dense `all_after_warmup`
population and cannot calibrate a train/valid generalization gap.

For optimizer diagnostics, report sampled update step, actual applied group LR,
pre/post-clipping gradient norm, pre-weight norm/RMS, absolute delta norm/RMS,
the delta/weight ratio when its denominator is nonzero, denominator floor and
zero-denominator flag, and available optimizer-state norm/count. Keep group
names, role, and head/body partition alongside each series. Continue to report
the raw objective components and official target-weighted WMSE; optimizer
telemetry is diagnostic and does not establish model-quality improvement.

`context_rows` and TBPTT must remain separate in future profiles. The L4 GRU
uses the pinned 1,000-row chunk horizon with state carried detached through its
20,000-row sequences. No throughput or memory threshold was inferred here;
any profile or fit still needs separate resource admission, and validation is
exposed rather than an independent untouched holdout.
