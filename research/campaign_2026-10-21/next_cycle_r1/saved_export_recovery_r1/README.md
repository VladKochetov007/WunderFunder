# Serial local saved-model export plans

These plans prepare two exports from existing, already-derived checkpoints after
the active i1 panel releases the local GPU. They add no fit, checkpoint
derivation, coefficient search, source mutation, or provider work. The root
operator owns the later fresh resource admission and execution. The plans have
not been launched and make no quality claim.

The combined recovery stage is capped at 7,200 seconds from the first export
through fixed grading: at most 900 seconds for the projected-mean full-valid
export, at most 3,600 seconds for both B16 exports together, 2,500 seconds for
fixed CPU acceptance/scoring, and 200 seconds for sequence transitions and
handoff overhead. These are proposed hard limits, not measured durations. Run
the three sequences serially, once each, with no queue or retry. Stop when the
combined wall cap is reached.
The earlier [B16 local fallback](../../kaggle_h512_b16_offline_recovery_r1/LOCAL_FALLBACK_R1.md)
remains the source of its exact recovery recipe; this packet narrows its older
standalone 7,200-second cap to 3,600 seconds for the two exports within this
combined stage.

Each GPU export uses one CPU core, 3,000,000,000 bytes of task RAM, no task
swap, one admitted visible GPU, a 4,000,000,000-byte allocator limit, and an
external stop at 5,000,000,000 observed VRAM bytes. The root must confirm the
device and idle GPU immediately before launch; both B16 exports remain serial
on that same device. Use the existing root
`.venv/bin/python`, `WNN_LOADER_WORKERS=0`, one CPU math thread, and physical
scratch at `/home/vlad/development/wunder/research_tmp`. No serving or frozen
grade code changes are part of this packet.

## Projected-mean export

The exact input is the already-derived serving-projected F0 checkpoint at
`/home/vlad/development/wunder/research_tmp/overnight-nll-projected-mean-f0-r1/artifacts/runs/overnight_nll_f0_s3_wmse_projected_mean_r1/model_ep1.pt`,
SHA256
`fc5e432eff521ad4829a6d1e0aefa6325a3d706b4dc02e17f602d3b33cf26a55`. Its
[derivation receipt](../WEIGHT_MEAN_PROJECTED_DERIVATION_R1.json) and frozen
[plan](../weight_mean_projected_r1/plan.json) bind the CP5/CP0/CP1 source
checkpoint hashes, provenance sidecars, split and fit membership, normalization,
teacher cache, and all eleven serving/training source pins. The plan SHA256 is
`3b0e0e7b072be59d6125e518b3f84443f011d6bb2713e7f6463940ce4d2115f1`; the
derivation receipt also binds its actual clean NLL worktree at commit
`d3055423a185d2ecfe7baf1c20206abeb743ad8b`. The predictor file SHA256 is
`904ddc9a2cafd194dc8d6ab9bdf1524120d6a089b83e33e2eb731e67abed6754`.
The existing derivation-receipt SHA256 is
`a0049df0aeea57e5dc267d9ee498162dc489560eb9cad277b057799fe6b19ac3`; its
derived run provenance is pinned to
`faf01caebd4cb29b2d69c8e571af43ab22000796a380db7499edb630563834d3`.

The [export sequence](PROJECTED_MEAN_LOCAL_EXPORT_SEQUENCE_R1.json) runs the
[standard-library preflight/export adapter](preflight_and_export_projected_mean_r1.py).
It verifies the complete existing source and
checkpoint pin set, the derived checkpoint and F0 split identities, then
refuses any existing alias, prediction, `.partial`, sidecar, or sequence-sum
path. It creates a temporary symlink at
`$PROJECT/artifacts/runs/overnight_nll_f0_s3_wmse_projected_mean_r1` to the
physical-scratch derived run; the helper removes that exact symlink in a
`finally` block after the frozen predictor returns.

The call uses the unchanged d305 NLL `scripts/predict_run.py` at epoch 1 with
`--campaign-full-valid-feature-only`, FP32, CUDA, and zero loader workers. It
exports all 1,873 valid groups while reading features only; the frozen F0
screen later reads targets. The predictor writes its canonical cache output
through a `.partial` file and removes a stale `seqsums_<run>.npz`, so both paths
are explicit no-overwrite checks. The full-valid selector reads row-group IDs
from the valid-target cache but does not load label or mask values.
`--create-only` is incompatible with this full-valid selector and is not added
or monkey-patched.

## B16 F1 exports

The [B16 sequence](B16_LOCAL_EXPORT_SEQUENCE_R1.json) reruns the exact clean
source gate and reuses the staged offline-recovery root
`/home/vlad/development/wunder/research_tmp/h512-f1-b16-offline-recovery-r1`
and exact clean inference worktree
`/home/vlad/development/wunder/research_tmp/b16-inference-source-68ff` at
commit `68ff682977b822f38351aac0f621146f5f13b440`. Its first step reruns the
existing `verify-source` gate. It then exports CP5 and SWA345 serially over the
same frozen 457-group F1 evaluation membership with the canonical predictor,
`--campaign-evaluation-only`, `--epoch 5`, FP32, CUDA, and zero loader workers.
Both serial calls must use the same freshly admitted physical GPU. Raw input
identity remains checked by the existing portable registry during predictor
preflight.

The unchanged CP5 checkpoint SHA256 is
`82cfa6662d7e8f2b61700253070f38e9f7ff0272aa26380049334306567840a9`; the
existing equal-parameter CP3/CP4/CP5 SWA checkpoint SHA256 is
`cc604820981478cea0eaa4a59f300c3d33f7acda23c1c1e37937c18fa8324cc6`. The
original provenance limitation is retained: the pinned trainer did not emit
`initial_parameter_sha256`, so the initial tensor digest remains unavailable.
The offline recovery acceptance checks the cold-start arguments, no-parent/no-
resume state, membership, normalization, update count, and checkpoint hashes
without reconstructing that missing digest.

Before running the sequence, follow the exact output and partial-file
nonexistence loop in
[LOCAL_FALLBACK_R1.md](../../kaggle_h512_b16_offline_recovery_r1/LOCAL_FALLBACK_R1.md).
That existing procedure preserves the recovery checkpoint and missing
initialization-digest caveat. The evaluation-only predictor cannot use its
`--create-only` mode; if any guarded path exists, stop without overwriting it or
retrying. The original fixed acceptance and five-endpoint grade run later under
the [CPU sequence](SAVED_EXPORT_CPU_ACCEPTANCE_SEQUENCE_R1.json); no grader or
score contract is modified.

## Ordered commands

Run from the project root after the i1 panel is terminal and fresh local GPU,
CPU, RAM, and task-budget admission passes. Use a new unit name if the named
unit has already existed; never restart an earlier unit. `CUDA_VISIBLE_DEVICES=0`
below represents the one device confirmed by the operator; change that ordinal
only to the freshly admitted device.

```sh
set -euo pipefail
PROJECT=/home/vlad/development/wunder/wnn_connectome_starterpack
PYTHON="$PROJECT/.venv/bin/python"
PACKET="$PROJECT/research/campaign_2026-10-21/next_cycle_r1/saved_export_recovery_r1"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES=0 systemd-run --user --wait --collect \
  --unit=wnn-projected-mean-export-r1 --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=3000000000 -p MemorySwapMax=0 -p RuntimeMaxSec=900 \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=WNN_GPU_MEMORY_BYTES=4000000000 \
  --setenv=TMPDIR=/home/vlad/development/wunder/research_tmp \
  -p WorkingDirectory="$PROJECT" \
  "$PYTHON" "$PROJECT/scripts/kaggle_sequence.py" \
  --plan "$PACKET/PROJECTED_MEAN_LOCAL_EXPORT_SEQUENCE_R1.json"

# Before this call, run the exact output-path guards in LOCAL_FALLBACK_R1.md.
CUDA_VISIBLE_DEVICES=0 systemd-run --user --wait --collect \
  --unit=wnn-b16-local-export-r1 --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=3000000000 -p MemorySwapMax=0 -p RuntimeMaxSec=3600 \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=WNN_GPU_MEMORY_BYTES=4000000000 \
  --setenv=TMPDIR=/home/vlad/development/wunder/research_tmp \
  -p WorkingDirectory="$PROJECT" \
  "$PYTHON" "$PROJECT/scripts/kaggle_sequence.py" \
  --plan "$PACKET/B16_LOCAL_EXPORT_SEQUENCE_R1.json"

# Run only if both export reports are complete and acceptance/grade paths remain new.
systemd-run --user --wait --collect \
  --unit=wnn-saved-export-cpu-acceptance-r1 --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=3000000000 -p MemorySwapMax=0 -p RuntimeMaxSec=2500 \
  -p WorkingDirectory="$PROJECT" \
  "$PYTHON" "$PROJECT/scripts/kaggle_sequence.py" \
  --plan "$PACKET/SAVED_EXPORT_CPU_ACCEPTANCE_SEQUENCE_R1.json"
```

The CPU acceptance sequence has a 2,500-second ceiling and uses one CPU core, a
3,000,000,000-byte RAM limit, and no swap. Keep transitions and handoff within
the 200-second reserve. The combined measured elapsed time, including
preflight and transitions, must remain at or below 7,200 seconds. Stop at the first failed gate, nonzero exit,
partial output, memory/VRAM stop, or exhausted time allowance; preserve the
report and artifacts for diagnosis.

## Outputs and navigation

- Projected-mean export: root `cache/pred_overnight_nll_f0_s3_wmse_projected_mean_r1_valid.npy` and its JSON sidecar; no target values are read until the fixed F0 screen.
- B16 exports: recovery-root CP5 and SWA345 valid prediction arrays and sidecars; then the existing acceptance receipt and fixed-grade receipt.
- Sequence reports are create-only at the paths named inside each plan. A report left `running` or `failed` is historical evidence and blocks an automatic retry.
- No exported predictions, acceptance, grade, or new scientific result exist from this packet at preparation time.

The detailed command and budget handoff is indexed in the
[next-cycle packet](../README.md) and the unified
[operations index](../../../unified/operations/README.md).
