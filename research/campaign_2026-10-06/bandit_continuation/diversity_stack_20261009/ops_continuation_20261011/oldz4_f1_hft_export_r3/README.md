# F1 HFT fixed-CP0 export R3

This packet exports the already-completed `oldz4-r3-f1-s3-hft-fit-resource`
attempt's fixed epoch-0 checkpoint. It does not crop, fit, score, or authorize
an export by itself. The checkpoint is an own-model HFT initialized from the
audited own F1 core crop. It fit 1,416 fold-1 groups and excluded the 457
outer-evaluation groups. Those rows were previously inspected; the prediction
is robustness evidence, not a fresh holdout or public-score claim.

`HFT_EXPORT_CONTRACT_R3.json` identifies the exact R2/R3 producer contracts,
source snapshots, actual fit artifacts, and output schema. R3 is an
export-only successor because the preserved R2 exporter has three concrete
consumer mismatches for this completed run: it dereferences
`trainer_argv` from the export contract although that argv belongs to the fit
contract; it expects sidecar `checkpoint_epoch` although the actual c337
predictor writes `epoch`; and its collision check omits the predictor's
`.partial` NPY and stale `seqsums_<run>.npz` deletion target. R3 imports the
unchanged R2 source/membership and c337 predictor, reads the trainer argv from
the fit contract, and checks the actual sidecar key. Its no-overwrite check
covers the final NPY, `.partial`, sidecar, and sequence-sums path.

The actual producer preflight is sealed in `CP0_EXPORT_PREFLIGHT_R3.json`.
It loads the actual checkpoint, alias, training state, provenance, history,
core crop receipt, split, target ordering, and predictor CLI. The CLI probe
intercepts `configure_torch`, so it proves argument parsing and import routing
without initializing a device or writing a prediction. The preflight records
the resource-only nature of the fit: 877 updates, 3 protected AMP skips,
`strict_profile_gate_passed=false`, and `profile_valid_for_throughput=false`.

To repeat the read-only preflight under the project support envelope:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn-agent-worktrees/student-export-handoff-20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_hft_export_r3/run_f1_hft_export_r3.py \
--stage preflight
```

One prediction export is possible only when separate root terminal and
single-export admission receipts exist in this packet and their exact SHA-256
values are supplied. R3 checks those receipts against the current checkpoint,
provenance, state, fit/source bindings, preflight receipt, output paths, and
bounded resource environment before calling the unchanged c337 predictor.
The final result receipt path is fixed by the contract and collision-checked
before the predictor starts. An earlier producer preflight, made before that
receipt destination was added, remains preserved as
`CP0_EXPORT_PREFLIGHT_PRE_RESULT_BIND_R3.json`; it binds the preceding contract
SHA and cannot authorize this final contract. The preflight before adding the
runner self-hash is also preserved as
`CP0_EXPORT_PREFLIGHT_PRE_RUNNER_BIND_R3.json`; only the final preflight binds
the final runner SHA.
There is no fit authorization in this packet. After export, it validates the
full-valid NPY header, exact byte length, finite values, sidecar checkpoint
binding, and canonical row-group order. Any failed/partial output remains a
collision and must be reviewed before another attempt.
