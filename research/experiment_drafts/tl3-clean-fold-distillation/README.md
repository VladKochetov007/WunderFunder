# TL3 student: labels versus clean-fold distillation

Implementation handoff; **zero fits, GPU profiles, executed tests or inference runs in this delivery**. Published as [draft PR #39](https://github.com/VladKochetov007/WunderFunder/pull/39) targeting `research-team`. The user's 23 October request authorizes code preparation/publication and explicitly forbids local tests and training. It does not launch the older local/Kaggle campaign or allocate paid compute.

## What this experiment measures

Does pointwise distillation improve a student with the **TL3 211-input, H256×2, linear-projection, GELU-head architecture**, including its contribution to a fixed three-model endpoint? Both students start cold with the same seed, exact initial tensors, fitting-only normalization, sample order and optimizer/update budget. The only objective change is alpha: 0 versus 0.5. Both read the same validated teacher payloads to keep input shapes, I/O and loss code paths equal; alpha 0 has no teacher contribution to its loss.

This is a **new direct-teacher ablation**, not reproduction of historical TL3 training. The original 211 student had a six-source mixture, family heads and a warm initializer. The two tied-lead members in the public ZIP had their own upstream distillation lineage. Those production artifacts are not certified clean controls. No old model, initializer, propensity model or teacher cache is admitted here.

For each outer fold, train two new label-only tied-lead H256×2 teachers. Their raw112 + event99 + causal lead52 features and shared book encoder follow the recovered architecture. Carrying the previous book across TBPTT chunks explicitly repairs the archived encoder's boundary reset; historical numerical parity is not claimed. This experiment therefore asks about this new clean teacher pair, not whether the original W18 recipe explains the public result.

## Frozen panel and cost

| Cell | Outer fold | Teacher seeds | Paired student seed | New fits |
|---|---:|---|---:|---:|
| F0 | 0 | 2001, 2002 | 1001 | 2 teachers + 2 students |
| F2 | 2 | 2003, 2004 | 1002 | 2 teachers + 2 students |

**Eight total scientific fits:** four clean teacher prerequisites, two labels-only students, two KD students; zero reused fits. This explicit prerequisite amendment exceeds the usual 4–6-fit initial screen because retaining two teachers on two folds otherwise requires unverifiable ancestors. Six bounded engineering profiles (teacher architecture, labels student, KD student in each cell) do not produce scientific endpoints. There is no search, coefficient tuning, checkpoint selection or automatic panel expansion.

Teachers: four full-data epochs, fixed epoch 3. Students: two epochs, fixed epoch 1. All eligible train and valid components outside the outer fold are used. Batch 32, complete 20,000-row sequences, chunk/update/detach 2,000, deterministic seeded batch order and canonical matching drop-tail. AdamW, LR 0.001, weight decay 0.01, OneCycle, clip 1. Strict FP32 for new computation, autocast and TF32 off, no quantization, no auxiliary heads, no future-label bootstrap, no external learned features. The complete command builder is [commands.py](commands.py); [recipe.json](recipe.json) freezes the estimand and settings.

The canonical objective uses `yc = clip(y, -2, 2)`, `tc = clip((teacher_a + teacher_b)/2, -2, 2)`, and `w = abs(yc) * valid`. Within each 2,000-row TBPTT window, it minimizes `sum(w * (prediction - ((1-alpha)*yc + alpha*tc))²) / sum(w)`. Clipping occurs **after averaging teachers**. At alpha 0 this is the label-only objective with the same normalization. Scoring always uses the unchanged official scored/prediction mask and metric, independently of this fitting objective.

## Clean data and ancestry

The pinned F0/F2 component manifests are the repository's `research/campaign_2026-09-29/artifacts/splits/fold0.json` and `fold2.json`. Expected hashes are in the recipe. Use the matching supplied train/valid Parquet, target caches, feature-only chain graph and chain-fold cache. `load_campaign_split` recomputes graph components and checks full fitting/evaluation membership. Complete sequences and linked components stay together across train and valid.

New teachers and students have empty initializers. Statistics fit only on the eligible fitting pool; the canonical two-batch normalization estimate is identical within the student pair and is fingerprinted before updates. No outer-fold loader/score is used during fitting or checkpoint selection. Teacher target exports read features only from fitting groups. The adapter scatters those outputs into the legacy full cache index and explicitly zeros excluded groups; it checks all finite values, exact coverage, checkpoint/source hashes and excluded zeros before use. Prediction on evaluation groups is a separate feature-only phase after fitting.

These folds were previously examined. “Clean” means excluded from all fitted ancestry, not pristine development exposure or a proven chronological holdout. Anonymous data and component grouping do not establish future-regime representativeness. Two cells confound fold and fitting seed.

## Receiver setup and commands

Run from the published branch on Linux with the mounted supplied-data workspace and an allocated B300. Keep Python source and scratch on physical disk. Install the existing locked environment **on the GPU receiver**, not locally:

```bash
uv sync --frozen --group research --group dev
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"
packet=research/experiment_drafts/tl3-clean-fold-distillation
```

Copy `receiver-assets.example.json` outside the source checkout and fill absolute paths. For relocated source bytes, supply the **source-sealed** portable registry and its known SHA256; do not invent a registry from unverified receiver bytes. Set `portable_registry` to null only when original manifest identities remain valid. Raw inputs are content-hashed at binding; subsequent phases reject changed size/mtime/ctime and revalidate the canonical split/portable content identities. Keep these input mounts immutable.

```bash
python "$packet/receiver.py" bind --assets /physical/scratch/tl3-inputs.json \
  --output /physical/scratch/tl3-bound.json
python "$packet/receiver.py" preflight --assets /physical/scratch/tl3-bound.json
```

Before scientific fits, run `python -m pytest tests/test_tiedlead_training.py` on the receiver for targeted causal/reset/chunk acceptance. They were not run locally. Then run one cell per assigned GPU; matching student arms must stay on the same device and software environment:

```bash
export WNN_TL3_ASSETS=/physical/scratch/tl3-bound.json
CUDA_VISIBLE_DEVICES=0 bash "$packet/run_cell.sh" F0
# On a separate assigned GPU, this cell may run concurrently:
CUDA_VISIBLE_DEVICES=1 bash "$packet/run_cell.sh" F2
python "$packet/receiver.py" score --assets "$WNN_TL3_ASSETS"
python "$packet/receiver.py" bundle --assets "$WNN_TL3_ASSETS"
```

The shell script performs explicit stages: preflight → profile-teachers → fit-teachers → cache → profile-students → fit-students → predict. Each stage is also callable individually through `receiver.py ... --cell F0`. One process/GPU; no DDP or queue manager. Use two aggregate CPU threads per cell and zero loader workers. Every profile measures actual full-sequence batches and finite gradients/updates, checks B300 identity and reserves 10% device-memory headroom. No B300 throughput or completion time has been measured. A failed profile stops scientific admission; it must not silently alter batch/precision/epochs.

## Recovery and output

Outputs live under `<project_root>/research_tmp/tl3-clean-kd-r1`, `<project_root>/artifacts/runs/tl3kd_r1_*`, and uniquely named prediction caches. Existing attempts are never overwritten. Per-cell advisory locks prevent two receiver stages from writing the same cell concurrently. A partial attempt is an execution failure, not a negative scientific result.

The canonical trainer saves `training_state.pt` at every epoch boundary, including optimizer, RNG, sampler and scheduler state. The receiver never auto-resumes or overwrites an attempt. Repeating a stage skips only already audited fits; use `receiver.py fit-teachers --assets "$WNN_TL3_ASSETS" --cell F0 --resume` (or `fit-students`) for explicit epoch-boundary continuation. It binds the full state hash and original initialization/source/assets/runtime, retains the failure log, writes a distinct resume log/receipt, and delegates optimizer/RNG/scheduler restoration to the canonical trainer. Uncheckpointed partial-epoch updates are discarded and repeated; this wasted compute is retained in logs. A failure before the first epoch checkpoint requires a separately reviewed restart; no automatic retry or weights-only continuation. Completed profiles/caches can be revalidated without recomputation.

After both cells finish, grading requires all eight completed audits and complete prediction coverage. Return separate `bundles/evidence.tar.gz` (logs, provenance, held-out predictions, grade and hashes) and `bundles/weights.tar.gz` (checkpoints and resume states), with `MANIFEST.json`. Supplied datasets are excluded. Keep the sealed assets file and exact source commit alongside the private archive. No model ZIP or portal upload is produced by this experiment.

## Frozen decision

Primary: equal-cell mean of official `global_wp_clip2_v1` differences between
`0.34*KD + 0.33*T_a + 0.33*T_b` and `0.34*labels + 0.33*T_a + 0.33*T_b`.
Secondary: standalone students, teacher-only average, and each complete mixture versus teacher-only, including t0/t1. The grader streams sufficient statistics and uses paired whole-component resampling (10,000 replicates, seed 231023). Intervals condition on fixed fits; they do not estimate training-seed uncertainty, erase exposure or establish public transfer.

Both cell deltas nonnegative and mean ≥ +0.0003 support separately budgeted confirmation. Mixed signs or a zero-crossing interval are uncertain, not automatic rejection. Negative complete-mixture results narrow this exact teacher/alpha/budget recipe. A teacher may be weak or the new cold student may underfit; report these alternatives instead of declaring all distillation ineffective. No local delta is added to the public score, and no superiority or first-place claim is made without measured results.

## Files and trace

- `common.py`: bounded metadata, immutable data/source bindings and canonical split admission.
- `commands.py`, `recipe.json`, `receiver-assets.example.json`: frozen model panel and receiver interface.
- `worker.py`: canonical trainer adapter, actual-process B300/FP32 identity and paired initial-state checks.
- `receiver.py`, `run_cell.sh`: explicit bounded stages and profile admission.
- `audit.py`: terminal epoch/update/ancestry/tensor checks and paired normalization/order checks.
- `cache_targets.py`: feature-only cache export and explicit excluded-row padding.
- `grade.py` and grading helpers: independent complete-endpoint evaluation and conditional uncertainty.
- `bundle.py`: separate scientific evidence/weights archives with hashes.
- `PR.md`, `VALIDATION.json`: publication rationale and honest validation status.

Historical architecture sources: `research/campaign_2026-10-18/quality_gap_r1/tl3_source_closure_r1/source_snapshot/` at commit `999f6af0`; its source-time limitations are recorded in `SOURCE_CLOSURE.json`. Plain student shape is independently visible in the hash-bound portal ZIP. Scientific lineage is the student/teacher connection, with canonical implementation and new fold-specific ancestry. H-TL3-CLEAN-KD-20261023 → E-TL3-CLEAN-KD-20261023-R1; published in [draft PR #39](https://github.com/VladKochetov007/WunderFunder/pull/39).

The mechanism reference is [Hinton, Vinyals and Dean, 2015](https://arxiv.org/abs/1503.02531): ensemble-to-student transfer motivates the experiment but supplies no evidence that this regression recipe will improve the competition metric.
