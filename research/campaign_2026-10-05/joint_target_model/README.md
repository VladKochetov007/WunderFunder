# Joint target distribution plus residual: isolated falsifier

This is an isolated, fit-only frozen-state experiment. It uses the existing
paired-state export path and does not change `train_rnn.py`, E464, or any root
trainer. Its first input is the existing fold-0 pair export from the two
fixed-epoch GRUs used in the earlier branch screen. That makes the CPU/GPU
falsifier runnable without another backbone export; it is not evidence for the
current strongest Sol model or for the public score.

The shared arm predicts one softmax over 32 paired atoms. Atoms are farthest-
point initialized and Lloyd-refined on at most 65,536 deterministically
subsampled, same-origin fit targets after per-coordinate clipping to `[-2,2]`.
The two independent controls each fit 16 marginal atoms from those same fit
rows. The shuffled negative control cyclically deranges the learned `t1` atom
pairing while preserving both atom marginals. All rows still train the heads;
the cap applies only to prototype fitting. Nearest-atom unweighted CE is the
primary likelihood objective for this quantized approximation, and every arm
also receives the same pooled clipped absolute-target-weighted MSE surrogate. This
WMSE is a fit objective, not the official score: `global_wp_clip2_v1` averages
two separate weighted Pearson correlations. Only the evaluation scorer reports
that official metric. The
unweighted CE is proper for the assigned quantized labels, not for the
unquantized continuous target law. Nearest-atom labels are assigned after
clipping to `[-2,2]`, matching both the atom fit and official metric target.
On the historical fit pool, 6.35% of rows have at least one raw coordinate
outside this clip.

`joint_tilted` decodes margin `j` as `sum_k q_k |c_kj| c_kj / sum_k q_k
|c_kj|`; each margin has its own denominator and falls back to the clipped
mean parent prediction when that denominator is zero. `joint_ordinary` uses
the ordinary codebook mean as a readout control. `regression_anchor` has the
same trainable parameter count and same auxiliary joint CE as the candidate,
but predicts only a continuous correction to the parent. All five arms use
the same trunk, two-output correction, initialization, optimizer, data order,
batch size, one epoch, and fixed checkpoint. The metric gate and final
residual are zero-initialized, so all arms begin exactly at the mean of the
two frozen members (clipped to the official prediction range).

This is an approximation to a paired conditional law: clipping, a finite
codebook, nearest-atom labels, and a finite one-epoch optimizer can all hurt.
The weighted-Bayes readout is exactly the metric-relevant conditional mean for
the represented discrete law, but this is not a theorem that the finite model
beats direct regression. The metric-optimal readout alone adds no advantage
over a perfect weighted regression head. The testable hypothesis is that a
train-fitted paired likelihood and its shared support improve finite-sample
inductive bias under a matched procedure. E421 is an important negative
result: its fixed categorical joint CE scored `.6303` versus unchanged
`.6821`, and tilted joint WMSE `.5807`; those losses are retained as prior
evidence and this retest does not erase them.

## CPU invariants

Run the small synthetic unit suite and forward/backward smoke under the
campaign one-core/2-GiB cap:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
WNN_SLICE=wnn-campaign.slice scripts/limited \
  python -m pytest -q research/campaign_2026-10-05/joint_target_model/test_model.py

WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
WNN_SLICE=wnn-campaign.slice scripts/limited \
  python research/campaign_2026-10-05/joint_target_model/fit.py --smoke --device cpu

WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
WNN_SLICE=wnn-campaign.slice PYTHONPATH=research/campaign_2026-10-05/joint_target_model \
scripts/limited python research/campaign_2026-10-05/joint_target_model/e2e_fit.py \
  --fit-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_fit \
  --member-index 0 --device cpu --preflight-only
```

The smoke uses generated data only and checks exact parent anchoring plus finite
forward/backward for all arms. It is not a training or score result.
The E2E preflight additionally checks real fit-export/checkpoint hashes,
component membership, and the real GRU's exact two-output parent parity on a
fit-sequence prefix; it does not update weights or open evaluation sequences.

After a fixed head fit exists, validate the paired evaluation export on CPU.
This reads the existing all-row export read-only, checks frozen component
membership/order, and does not calculate arm scores:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
WNN_SLICE=wnn-campaign.slice PYTHONPATH=research/campaign_2026-10-05/joint_target_model \
scripts/limited python research/campaign_2026-10-05/joint_target_model/score.py \
  --fit-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_fit \
  --eval-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_eval \
  --device cpu --preflight-only
```

## GPU launch commands (documented, not run by this reviewer)

A small accelerator smoke can be run on generated data without touching a
provider queue:

```bash
python research/campaign_2026-10-05/joint_target_model/fit.py --smoke --device cuda
```

The fit-only falsifier reuses the already exported states and consumes no
evaluation arrays. It writes checkpoints to a new directory and refuses to
overwrite one:

```bash
python research/campaign_2026-10-05/joint_target_model/fit.py \
  --fit-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_fit \
  --output-dir research/campaign_2026-10-05/joint_target_model/runs/fold0_pair_falsifier \
  --device cuda --epochs 1 --batch 8192 --lr 0.001 --seed 271828
```

After that head-only falsifier is independently reviewed, this isolated
end-to-end option continues the first checkpoint that generated the same state
export. It is deliberately tiny: the first eight fit groups from the frozen
fold manifest, one sequence at a time, one TBPTT pass with 1,024-row chunks,
and the same five matched arms/optimizer updates. It validates the checkpoint
SHA, split/component manifest, official scored mask plus 99-row warm-up, and
exact two-output parent parity before updates. Each sequence resets the causal
GRU state; no evaluation group is loaded.

```bash
python research/campaign_2026-10-05/joint_target_model/e2e_fit.py \
  --fit-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_fit \
  --member-index 0 \
  --output-dir research/campaign_2026-10-05/joint_target_model/runs/fold0_e2e_8seq \
  --device cuda --max-sequences 8 --batch-sequences 1 --chunk 1024 \
  --lr 0.0001 --seed 271828
```

Do not use fit diagnostics for arm selection. The package scorer streams every
fixed arm over `need_prediction AND is_scored`, reports `global_wp_clip2_v1`
and paired component bootstrap against the parent-pair average, and performs no
arm selection. It intentionally does not compare this older fold with the
current public `.666` anchor.

```bash
python research/campaign_2026-10-05/joint_target_model/score.py \
  --fit-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_fit \
  --eval-prefix research/campaign_2026-09-29/artifacts/pair_states/c20260929_pair_f0_eval \
  --head-dir research/campaign_2026-10-05/joint_target_model/runs/fold0_pair_falsifier \
  --output-dir research/campaign_2026-10-05/joint_target_model/runs/fold0_pair_falsifier_eval \
  --device cuda --batch 8192 --boot 2000 --seed 271828
```

A scientifically current frozen-state replication must first export fit and evaluation states
from the selected E464 parent pair, pin the two checkpoint SHAs, and freeze a
new clean split. The E2E wrapper preserves the two-value model callback, but
this historical 2026-09-29 parent pair is a mechanics test, not the current
strongest Sol. Any current-parent continuation or evaluation needs a new
registered export and independent code/provenance review. The E421 negative
result remains binding prior evidence; this experiment is a retest on a
different, larger fit procedure and cannot retroactively make E421 positive.
