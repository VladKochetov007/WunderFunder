# B03 on the GPU receiver

Use Bash on the assigned Linux L4 host with Git, GNU coreutils, util-linux (flock), a provisioned native Python environment (3.11–3.13 with PyTorch/CUDA, PyArrow and the team training dependencies), the official data and the original fitting-only PR49 teacher cache. A cell uses one L4 and keeps the same physical device for its profile and fit. Reserve the time, host memory, VRAM and persistent artifact directory with the receiving owner before starting. The commands below are for this eight-cell experiment only.

## 1. Pin the two sources

Run from an existing authenticated **WunderFunder** clone. The following creates two detached worktrees, without switching the team's working checkout. Set B03_WORK to a new absolute directory on the receiver. If the existing clone uses sparse checkout, enable the B03 packet directory in the new PR92 worktree.

```bash
set -euo pipefail
: "${B03_WORK:?absolute directory for the two new worktrees}"
repo=$(git rev-parse --show-toplevel)
mkdir -p -- "$B03_WORK"
git -C "$repo" fetch origin refs/pull/92/head
packet_commit=$(git -C "$repo" rev-parse FETCH_HEAD)
git -C "$repo" worktree add --detach "$B03_WORK/pr92" "$packet_commit"
git -C "$repo" fetch origin refs/pull/96/head
base_commit=33542502a1404a2843359b3999fa34f3a157bf3f
git -C "$repo" worktree add --detach "$B03_WORK/native" "$base_commit"
export WF_BASE="$B03_WORK/native"
export PACKET="$B03_WORK/pr92/research/experiment_drafts/l4-impact-batch2-vlad-20261112/b03"
printf 'PR92 revision %s\nNative source %s\n' "$packet_commit" "$base_commit"
```

## 2. Qualify the data, cache and loss

Set the following variables to the values for this allocation. WF_DATA contains `datasets/train.parquet`, `datasets/valid.parquet` and the original `cache/` with the chronological chain graph, indices and source audit recorded in the split. Native fitting reuses or creates its source-bound moments under `cache/sequence_normalization/`. WF_CACHE is the directory containing the original teacher `manifest.json` and all of its shards. WF_RUNS must be a **fresh persistent directory outside the two worktrees**. It will hold the private derived runtime and every attempt's artifacts. The Python interpreter and all paths must be absolute.

```bash
: "${WF_OWNER:?receiver responsible for this allocation}"
: "${WF_DATA:?original dataset root}" "${WF_SPLIT:?absolute path to chronological_primary.json}"
: "${WF_CACHE:?PR49 fitting cache directory}" "${WF_PYTHON:?absolute path to the native Python}"
: "${WF_RUNS:?fresh persistent output directory}" "${CUDA_VISIBLE_DEVICES:?one assigned physical L4}"
export WF_OWNER WF_DATA WF_SPLIT WF_CACHE WF_PYTHON WF_RUNS CUDA_VISIBLE_DEVICES
bash "$PACKET/run.sh" check
bash "$PACKET/run.sh" probe
```

The first command runs the native checks for the pinned source, chronological split, recorded data-file identities, training membership and teacher-cache shard hashes. The second checks the four objective values and gradients, masking, teacher detachment and the causal clock across a chunk boundary. Inspect `$WF_RUNS/qualification.json`, `probe.json` and the stage logs before profiling. Both checks are shared across all eight cells.

The [PR96 controller](https://github.com/VladKochetov007/WunderFunder/blob/7926582ac6042f3ca01eedeebf4433ce6d06b3e3/research/experiment_drafts/pr49-standalone-kd-20261112/controllers/l4_standalone_kd_wave_r1.py) records the previous receiver locations. If the same mount is allocated, WF_DATA was `/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project`, WF_SPLIT was `$WF_DATA/chronological_primary.json`, WF_CACHE was `/data2/lc-epanov_s/tabdata-final-quality-20261112-r1/teacher-cache`, and the interpreter was `/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python`. Use a new WF_RUNS for B03.

## 3. Profile each cell, then approve from measured resources

These commands give a serial, one-L4 schedule. Keep the `cells` array in the same Bash session. For parallel allocations, assign disjoint cells explicitly to different workers using the same qualified output store and revision. Each cell's profile and fit must use the same physical GPU and software environment.

```bash
cells=(labels_s203301 since32_s203301 kd_s203301 both_s203301
       labels_s203302 since32_s203302 kd_s203302 both_s203302)
for cell in "${cells[@]}"; do
  bash "$PACKET/run.sh" profile "$cell"
done
```

Each profile takes 30 optimizer updates and exercises checkpoint reload. Review each `$WF_RUNS/<cell>/profile/profile_receipt.json` for errors, peak GPU reservation, host RSS and throughput. Check the observed rate against the actual allocation for 38,760 updates, six full validation passes and export. Set the two limits to the allocation in GiB. Record what the receiver checked in the review note, and approve only the reviewed cells. The loop below applies when all eight passed that review.

```bash
: "${WF_MAX_HOST_GIB:?allocated host GiB}" "${WF_MAX_VRAM_GIB:?allocated GPU GiB}"
: "${WF_REVIEW_NOTE:?record the profile and wall-clock budget review}"
export WF_MAX_HOST_GIB WF_MAX_VRAM_GIB WF_REVIEW_NOTE
for cell in "${cells[@]}"; do
  bash "$PACKET/run.sh" approve "$cell"
done
```

## 4. Fit, export and score the fixed sixth epoch

```bash
for cell in "${cells[@]}"; do
  bash "$PACKET/run.sh" fit "$cell"
  bash "$PACKET/run.sh" predict "$cell"
  bash "$PACKET/run.sh" score "$cell"
done
```

Fit retains all six epoch snapshots. Prediction uses the fixed final `fit/model.pt` and the full official evaluation membership. Scoring checks the exported arrays using the canonical pooled WP metric. Stage deadline caps are 60 minutes for qualification, 20 for the numerical probe, 30 per profile, 10 per approval, 7 hours per fit and 2 hours each for prediction and scoring. Set `WF_TIMEOUT_SECONDS` on a **single command** to shorten a stage. The caps are limits and do not forecast runtime.

## 5. Compare and keep the evidence

Use the same worktrees, WF_BASE and environment. This CPU step has a 30-minute deadline and reads the score receipts and component moments after all eight exports have been scored.

```bash
bash "$PACKET/run.sh" readout
```

`$WF_RUNS/comparison.json` reports each arm on each seed, gains over either single intervention, the factorial interaction and component-bootstrap intervals on the reused official valid. Retain this file, the qualification and probe, original profiles and approvals, terminal status, checkpoints, per-epoch monitor, export manifests/shards, independent scores and `stages/*/attempt.*/` logs. An incomplete panel yields no comparison. Keep failures, timeouts and negative results with their original run IDs. After an interruption inspect the terminal status and artifacts before assigning another attempt. No automatic retries or submissions are part of these commands.
