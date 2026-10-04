---
name: wnn-connectome-top1
description: Continue the root Wunder Connectome Challenge 2 research campaign toward a deterministic submission with honestly expected public score >=0.6660. Use when auditing, experimenting, validating, packaging, or planning this project's root Challenge 2 work. Exclude the separate wunder/challenge-3 tree and do not claim the target from proxy-only evidence.
---

# WNN Connectome Top-1 Campaign

## Current evidence overrides the historical contract below

As of the October 4 portal results, the fixed seed-4 combo i16 ZIP scored
**0.6660** (`2e5j6bnt`, 37m07s /626 MB), reaching the numeric target at displayed
precision. The research goal is explicitly **paused**: do not start fits,
kernel pushes, timing trials or research iterations without user resumption.
Read `research/campaign_2026-10-04/CURRENT_STATUS.md` and the latest public-result
reflection first. The old incumbent/8-hour budget/metric ambiguity in the frozen
sections below are historical, not current instructions. The authoritative
metric is `global_wp_clip2_v1`; the later user-approved research allowance was
80 GPU-hours. Original E462 q8 ZIPs are withdrawn after a libpython3.13 prerun
failure; repaired ZIPs have compatibility proof but no runtime approval yet.

## Mission

Produce a deterministic, rules-compliant submission ZIP for the root Connectome project whose
evidence supports an honest expected public metric of at least **0.6660**. The current verified
public incumbent is **0.6622**. The supplied leaderboard snapshot has a leader near **0.6677**.

This is an empirical research campaign, not a packaging exercise. Existing blend evidence does not
support an expectation of 0.6660. Do not rename a locally positive candidate as the deliverable.

## Scope

- Work in the repository root containing `README.md`, `src/`, `scripts/`, `research/`,
  `artifacts/`, `cache/`, and `submissions/`.
- Exclude the nested `wunder/challenge-3` tree.
- The nested `wunder/challenge-2` documentation describes a different 1,000-step data format and
  clip-6 metric. It is not authoritative for the root 20,000-step project.
- Preserve all existing user work. The worktree is dirty and contains concurrent research files.
- Never reset, clean, or revert unrelated changes.
- Do not upload. Uploading is a human gate.

## Frozen Campaign Contract

1. **Target:** estimated public score `>= 0.6660`, not merely a point estimate above 0.6622.
2. **Incumbent:** `submissions/final_since2_gclp/final_since2_gclp.zip`, reported public 0.6622,
   submission `1bmxngta`, runtime 36m27s, memory 619 MB.
3. **Known archive identity:** SHA-256
   `7990bafd721b09bde2670bc9281fdb0913456a7e2caf046d894fed31faae10f7`.
4. **Incumbent members:** equal blend of `ab67_final_since_s4_gcorr` and
   `ab67_final_since_s3_gcorr`.
5. **Research budget:** at most 8 GPU-hours for this campaign.
6. **Resource ceiling:** aggregate at most 12 CPU cores, 40 GB RAM, and 5 GB observed VRAM.
7. **Concurrency:** one GPU training job at a time.
8. **Recommended hard RAM cap:** 36 GiB to stay below 40 decimal GB.
9. **Submission limits:** one vCPU, 16 GB RAM, 60 minutes, ZIP <=20 MB, offline Linux/Python 3.11.
10. **Verification tier:** C until the public score is returned. Local validation and calibrated
    leaderboard proxies are imperfect and heavily selected.
11. **Success:** a verified ZIP plus an independent expectation audit supporting `E[public] >=
    0.6660`; later, user-reported public score confirms or falsifies it.
12. **Abort:** leakage, invalid metric, prediction mismatch, adaptive member identity, runtime over
    budget, or an evidence-based expectation below 0.6660.

## Resource Enforcement

For CPU work use:

```bash
WNN_CPU_PCT=75 WNN_MEM_MAX=36G scripts/limited <command>
```

On a 16-logical-CPU host this requests 12 workers and a `CPUQuota=1200%`. `scripts/limited` does
not cap GPU memory. Set `WNN_GPU_FRACTION` conservatively and verify total device usage rather than
trusting the PyTorch allocator limit. Start around `0.70` on a 6 GB card; increase only after a
measured batch-32 smoke test remains below 5 GB total usage.

At the end of the prior session, `nvidia-smi` failed to communicate with the NVIDIA driver. Recheck
before launching GPU work. No remote target was configured. Do not silently switch the historical
batch size if batch 32 does not fit; first establish a matched gradient-accumulation control or
record that exact recipe reproduction is blocked.

Long runs go through the durable job mechanism, not background shell processes. Seed every run,
cap thread pools, declare artifacts, preserve checkpoints, and log the exact command and exit code.

## Evidence Hierarchy

Use this order when claims disagree:

1. Uploaded archive bytes and embedded manifest.
2. Raw checkpoint, prediction array, split manifest, and training log.
3. Script implementation at the recorded revision.
4. Machine-readable experiment output.
5. Experiment ledger entry.
6. Research memo or agent summary.
7. User-reported portal evidence.

The public score-to-archive association is user-reported portal evidence. The archive identity and
payload are directly inspectable.

Never cite this skill as primary evidence. Reopen the named file or artifact.

## Critical Metric Contradiction

Resolve this before freezing the evaluator.

The root published files define **equal mean of per-sequence weighted Pearson scores**:

- `README.md:23-34`
- `METRIC.md:72-82`
- `src/wnn/metric.py:63-76`
- `docs/faq.md:9-13`

Later research treats **globally pooled weighted Pearson across rows** as the live competition
metric:

- `src/wnn/gmetric.py`
- `scripts/decide.py`
- `scripts/verify_submission.py`
- `research/game_changers.md:10-15`
- `research/contract-2026-09-28-new-best.md:11-14`

The two metrics materially differ. The nested `wunder/challenge-2` docs cannot resolve this because
they describe a different challenge format. The prior web search produced no usable result. Obtain
a dated official text or API response from the live root competition before declaring one metric
authoritative. Until then:

- Report both metrics for every serious candidate.
- Do not optimize one and describe it as definitively official.
- Preserve identical row set, mask, folds, and aggregation when comparing candidates.
- Treat public-score calibration as evidence that pooled WP is useful, not proof of authority.

## Current Evidence Frontier

### Verified incumbent

`final_since2_gclp.zip` is the only audited public incumbent. Do not confuse it with
`final_since2_gclp_shared.zip`, which collapses to one member on v2 and has no public score.

Verify at session start:

```bash
sha256sum submissions/final_since2_gclp/final_since2_gclp.zip
unzip -tqq submissions/final_since2_gclp/final_since2_gclp.zip
unzip -p submissions/final_since2_gclp/final_since2_gclp.zip members.json
```

### Existing blends do not meet the target expectation

The selected leaderboard-calibration artifact is
`research/artifacts/lb_calibration.json`. Its strongest listed point prediction is about 0.66382,
with leave-one-out RMSE near 0.0010. This is below 0.6660 and was selected after exploring many
candidates and proxy variants.

Important existing hypotheses:

- Fixed incumbent plus raw chunk-500, equal thirds: compatible since/base analog delta around
  `+0.00099 +/- 0.00031` pooled WP; repeatedly inspected holdout about `+0.00038 +/- 0.00037`.
- Fixed incumbent plus raw online-normalized checkpoint, equal thirds: compatible analog delta
  around `+0.00073 +/- 0.00029`; holdout about `+0.00032 +/- 0.00038`.
- Four-member raw chunk-500 plus raw online-norm: compatible analog around
  `+0.00115 +/- 0.00038`; runtime is less attractive.
- Existing GCLP `keep_k4`: `research/artifacts/ensemble_scan_keep.json` reports pooled exact delta
  `+0.001009 +/- 0.000361`, both available folds positive, all four classes positive. Its holdout
  gain is approximately zero and the package has a runtime governor.
- EMA6 candidates looked strong in selected local scans but public submissions scored 0.6612 and
  0.6616, below the incumbent.
- Merged-model additions were negative against final incumbent predictions at every tested weight,
  but that comparator is not symmetric OOF because the final incumbent was trained on those
  development rows.

These can be baselines or controls. None honestly supports expected public 0.6660.

## Validation Truths

### There is no exact fold-matched incumbent analog

The clean available OOF anchor is a **since/base** pair, not the public **since/since** pair. Do not
call it the incumbent.

Available:

- `ab82_ft_since4s3_f0`, initialized from `ab62_since_f0@ep3`.
- `ab82_ft_since4s3_f1`, initialized from `ab62_since_f1@ep3`.

Missing for an exact four-fold since/since panel:

- Seed-4 four-epoch since parents and two-stage GCLP fine-tuning on folds 0 and 1.
- Seed-3 and seed-4 equivalents on folds 2 and 3.
- Total: six parent fits and twelve fine-tuning stages.

The public seed-4 and seed-3 parents are both fine-tuned with FT seed 3. Preserve that detail.

### Folds 2-3 are not untouched

They were used in architecture, weighting, blending, and calibration research. They may be used for
less-exposed confirmation under a newly frozen contract, but never described as untouched.

### The historical holdout is not chain-independent

The 368-sequence holdout has substantial chain overlap with training/development data and has been
repeatedly inspected. Use it as a sanity check, not an independent hidden-period estimate.

### Ordinary OOF still has overlap risks

- `prop_v1` was trained/selected using validation masks.
- Validation chains can link to train chains.
- Standard fold training excludes held-out validation rows but not necessarily train row groups
  sharing their chain.
- `train_rnn.py` can select `model.pt` on evaluation score; fixed epoch 3 avoids this one channel.

A genuinely cleaner panel requires a fold-specific propensity model and exclusion of train row
groups linked to the evaluation chain. The inspected trainer has no ready flag for the latter.

## First Execution Phase: Freeze Measurement

Do not start broad model search until these are complete.

1. Resolve or explicitly dual-report the official metric.
2. Recheck incumbent archive hash and manifest.
3. Inventory exact run/checkpoint/prediction availability.
4. Implement or verify chain-component split metadata that excludes linked train row groups.
5. Train fold-specific propensity models without the evaluation fold and with fixed checkpoint
   choice.
6. Build exact since/since fold-matched controls.
7. Freeze candidate weights and model hyperparameters before looking at confirmation results.

Historical parent recipe, reconstructed from `artifacts/logs/q23e.sh`:

```bash
export WNN_CPU_PCT=75 WNN_MEM_MAX=36G
export WNN_GPU_FRACTION=0.70 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4

EMA="--train-source train+valid --n-valid-folds 4 --eval-source vfold \
--full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 \
--propensity prop_v1 --prop-power 2 --prop-eps 0.03 \
--valid-true-mask 0.03 --hidden 128 --epochs 4 \
--ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast \
--since-weight 32 --since-floor 0.25"

scripts/limited .venv/bin/python scripts/train_rnn.py \
  --name exact_since_s${SEED}_f${FOLD} \
  --valid-fold "$FOLD" --seed "$SEED" $EMA

scripts/ft_gclp.sh \
  "exact_since_s${SEED}_f${FOLD}@ep3" \
  "exact_ft_since4s${SEED}_f${FOLD}" \
  "$FOLD" 3
```

This command reproduces the historical overlap-prone recipe. Replace `prop_v1` and add linked-chain
exclusion for the clean panel rather than pretending the command alone solves leakage.

Estimated cost for the six missing parents plus FT is 1.5-2 sequential GPU-hours. Measure the first
fold before committing the full estimate.

## Research Strategy

Allocate the 8 GPU-hours approximately as follows:

| Phase | GPU budget | Purpose |
|---|---:|---|
| Measurement controls | 2.0 h | Exact since/since references and parity checks |
| Cheap mechanism screens | 2.0 h | Frozen-backbone or one-fold falsifiers |
| Confirmation | 3.0 h | Second folds/seeds for survivors |
| Final fit and package reserve | 1.0 h | Only after an expectation gate passes |

Reserve 10-20% of the experiment count for a structurally novel arm. Do not spend the campaign on
blend-weight grids, post-hoc routing, calibration, or generic width/depth changes already shown to
be noisy or negative.

Use the creative loop:

`reframe -> propose -> attack -> cheap probe -> implement -> grade -> reproduce -> update`

Separate proposer, critic, grader, and reproducer roles. A candidate author does not approve its own
result. Preserve failed branches in the ledger.

## Priority Mechanisms

### 1. Cross-member interaction head

**Representation change:** jointly read the two incumbent hidden states instead of averaging two
independent scalar predictions.

Candidate form:

```text
p = 0.5 * (p_seed3 + p_seed4)
    + W_out((A h_seed3) * (B h_seed4))
```

Start with low rank 8 and zero-initialize the residual correction.

**Mechanism:** the incumbent discards interactions between its two hidden representations. A small
bilinear residual can condition one member's evidence on the other's state without adding another
full GRU.

**Cheap falsifier:** freeze matched OOF backbones and compare, under equal parameter/update budget:

- bilinear cross-member head;
- unary/additive per-member nonlinear head;
- output-only calibration control;
- unchanged average.

An AUC improvement without exact metric improvement fails. Require the interaction head to beat the
unary control, not merely the original average.

**Entry points:** `src/wnn/nn.py`, `scripts/train_rnn.py`, `native/src/rnn_fast.rs`, and
`native/src/python.rs`.

**Risk:** hidden coordinates are seed-specific and highly redundant; the head can memorize local
sequence calibration.

### 2. Content-addressed quote-event memory

**Representation change:** keep a small causal ring buffer, for example the last 32 i0 price-event
embeddings, and retrieve relevant events using a query from the current state.

**Mechanism:** a GRU compresses all history into one state. Rare, long error episodes may require
direct access to specific past events rather than another recurrent compression.

**Cheap falsifier:** on frozen causal embeddings compare:

- query-key content retrieval;
- parameter/FLOP-matched position-only event kernel;
- last-event-only input.

Stop if content retrieval does not beat the position-only control. Never place residuals, labels,
future events, or future mask information in memory.

**Entry points:** event flags in `src/wnn/features.py`; state and model in `src/wnn/nn.py`; chunk
state in `scripts/train_rnn.py`; streaming parity in `native/src/stream.rs` and
`native/src/python.rs`.

**Risk:** earlier event-clock recurrence, timing corrections, seasonal readouts, and future-quote
auxiliaries failed. The error may depend on unknowable future information rather than memory.

### 3. Prefix-conditioned low-rank input adapter

**Representation change:** use causal prefix context to modulate cross-feature interpretation, not
just normalize each feature.

Candidate form:

```text
z_t = GELU(W x_t + b + U(g(c_{t-1}) * (V x_t)))
```

Initialize the residual path to zero and shrink context toward the global prior early in a sequence.

**Mechanism:** strong source separability and covariance-subspace differences suggest that identical
feature codes may have context-dependent meaning. A low-rank adapter can rotate interactions while
retaining raw levels.

**Cheap falsifier:** compare matched-budget conditional low-rank, unconditional low-rank, and
diagonal-only modulation. Gains that occur only on source-like chains fail.

**Entry points:** `src/wnn/nn.py:RNNModel.encode`, `scripts/train_rnn.py:build_inputs`, forward chunk
and detach-state paths, then batch/native parity.

**Risk:** domain separability may reflect genuine `P(y|x)` change rather than nuisance encoding.
Earlier online z-score and additive warm-up corrections were flat or negative.

## Lower-Priority Existing Candidate

If GPU work remains blocked, inspect the fixed raw-chunk500 K3 as a reproducible fallback, not as the
0.666 deliverable:

```bash
sha256sum submissions/final_k3_ch500/submission.zip
unzip -tqq submissions/final_k3_ch500/submission.zip
unzip -p submissions/final_k3_ch500/submission.zip members.json
unzip -l submissions/final_k3_ch500/submission.zip
```

Use `submission/solution_multi.py` for fixed membership. Do not use the adaptive governor in
`submission/solution_ensemble.py`, which can drop trailing members according to wall time and thus
change the prediction identity across hosts.

## Promotion Gates

### Screen gate

- Same rows, mask, metric, fold provenance, and compute budget as control.
- Exact paired delta at least `+0.0005` on the first screen.
- No implementation mismatch or leakage finding.
- Mechanism-specific control is beaten, not only the incumbent analog.

### Confirmation gate

- Predeclare candidate, checkpoint epoch, weights, and all transformations.
- Positive delta on each confirmation fold.
- Pooled paired delta exceeds one chain-bootstrap SD.
- Chain-bootstrap 95% interval reported; target lower bound above zero for final promotion.
- No market-class delta below `-0.0005`.
- Direction robust under exact mask and softened-mask sensitivity, while softened masks remain
  diagnostics rather than hidden-mask estimates.
- Positive under both pooled and block-average metrics until the official metric is resolved.

### Expectation gate

An expected score of 0.6660 requires a delta of roughly +0.0038 over the reported 0.6622 incumbent.
Do not add gains from correlated experiments. Require a calibrated posterior or forecast that:

- accounts for proxy selection and winner's curse;
- uses leave-one-submission-out calibration;
- reports predictive uncertainty, not only fit RMSE;
- discounts local effects that failed public transfer, especially EMA6 and softened-mask gains;
- places the posterior mean at or above 0.6660;
- is independently audited before packaging is called successful.

The present calibration does not pass this gate.

### Packaging gate

- Fixed member count and weights for the whole run.
- No wall-clock governor that changes the ensemble.
- ZIP <=20 MB and valid archive.
- `solution.py` at archive root.
- Correct warm-up behavior and state reset on every sequence change.
- Finite predictions on every required row.
- Deterministic repeated output.
- Batch, Python streaming, AVX2, v2, and v3 parity within the predeclared tolerance.
- One-vCPU projected full-test runtime <=52 minutes, leaving margin below 60.
- Offline Python 3.11 scorer-image smoke test.
- Check the actual ZIP's ELF dependencies with `scripts/check_submission_abi.py`:
  reject any `DT_NEEDED` libpython dependency. `abi3-py311` or an `.abi3.so`
  filename alone is not portability evidence. Build Python extensions with the
  `extension-module` feature in the scorer's base image.
- Bind offline Python 3.11 import, constructor and streaming checks to the exact
  delivered ZIP hash and every shipped CPU build. Use
  `scripts/check_submission_python311.py` in a network-disabled container.
  Local Python 3.13 parity cannot replace this gate. See the E462 packaging
  incident in `research/campaign_2026-10-04/E462_PACKAGING_INCIDENT_20261004.md`.
- Both pooled and block-average scores recorded on the same complete validation subset.

## Runtime and Parity Notes

- Historical public incumbent runtime: 36m27s.
- Historical K2 local timing: about 96.8 microseconds/row on a loaded host.
- Historical K4 v3: about 190 microseconds/row; v2 about 262 microseconds/row.
- K3 was projected around 38 minutes, but rerun under one-vCPU scorer conditions.
- Ledger E-255 reports AVX2 versus v3 exact parity and v2 versus v3 maximum absolute difference
  around `4.77e-7`.
- `scripts/check_package.py` checks prediction correlation and maximum difference.
- `scripts/verify_submission.py` labels pooled WP official but also reports block WP. Its saved NPZ
  reportedly stores only the final row-group prediction array, so do not use that NPZ for
  multi-sequence parity without inspecting/fixing the script or restricting to one sequence.

## Failed or Deprioritized Families

Do not repeat these unchanged:

- Wider h192 replacement.
- Static MLP replacement.
- LSTM or extra GRU depth.
- Generic post-hoc calibration and per-segment weighting.
- Future quote-path auxiliary.
- Mask auxiliary and weak class-tier corrections.
- Spread/history correction scans.
- Online recalibration as already implemented.
- More EMA6-only blending; related public submissions regressed.
- Existing merged GRU as an additive member; no deployable final model exists and stored exact-pair
  comparison was negative, despite comparator asymmetry.
- Runtime-dependent ensemble membership.
- `gclpall` local evaluation as independent evidence; those models fit all validation rows.

Study the negative ledger before proposing a nearby idea:

- `research/experiments.jsonl`
- `research/hypotheses.jsonl`
- `research/game_changers.md`
- `research/runs/INDEX.md`
- `research/merge_2026-09-29/`

## Where to Look

### Contracts and metric

- `README.md`
- `METRIC.md`
- `docs/faq.md`
- `src/wnn/metric.py`
- `src/wnn/gmetric.py`
- `scripts/decide.py`
- `scripts/verify_submission.py`
- `research/contract-2026-09-28-new-best.md`

### Model and training

- `src/wnn/nn.py`
- `src/wnn/features.py`
- `scripts/train_rnn.py`
- `scripts/ft_gclp.sh`
- `scripts/predict_run.py`
- `scripts/train_propensity.py`
- `artifacts/logs/q23e.sh`
- `artifacts/logs/q27.sh`
- `artifacts/logs/q51_ft_folds.sh`

### Validation and ensemble evidence

- `scripts/ensemble_scan.py`
- `scripts/holdout_compare.py`
- `scripts/lb_calibration.py`
- `research/artifacts/ensemble_scan_keep.json`
- `research/artifacts/lb_calibration.json`
- `research/artifacts/lb_calibration_grid.json`
- `research/artifacts/lb_calibration_k4grid.json`
- `research/artifacts/lb_calibration_headonly.json`

### Incumbent and packaging

- `submissions/final_since2_gclp/`
- `submissions/final_k3_ch500/`
- `submission/solution_multi.py`
- `submission/solution_ensemble.py`
- `scripts/package_submission.py`
- `scripts/check_package.py`
- `scripts/build_native.sh`
- `native/src/`

### Diagnostic evidence

- `research/artifacts/domain_geometry.json`
- `scripts/audit_domain_geometry.py`
- `research/artifacts/brainteaser_aux_sign_readout.json`
- `research/artifacts/brainteaser_metrics.json`
- `scripts/audit_class3_law.py`
- `scripts/probe_aux_sign.py`
- `scripts/probe_regime_robustness.py`
- `notebooks/learning_dynamics.py`

## Marimo Workflow

The existing notebook is `notebooks/learning_dynamics.py`. At the end of the prior session no
marimo server was discoverable. Always discover first. If none is running, start this notebook in
the background without `--headless`, preferably with `--no-token`, then use the `marimo-pair` skill.

When a session is live:

- Read and edit cells through marimo code mode, not direct file writes.
- Do not create temporary-file dependencies in cells.
- Use the notebook for inspecting measured data and candidate deltas, not as the sole record of a
  result.
- Persist material computations as scripts and machine-readable artifacts with exact commands.

## Autoresearch Operation

Use the platform study/experiment tracker only after the evaluator is frozen and a runnable script
logs one fixed metric. Proposed initial queue:

1. Baseline exact since/since control, priority 1000.
2. Cross-member interaction head with matched unary control.
3. Content-addressed event memory with position-only control.
4. Prefix-conditioned low-rank adapter with unconditional control.

One run per idea. Record every ended run, including killed or negative runs. Keep at least three
mechanistically distinct ideas queued while budget remains. Every sixth run or after four failures,
reframe rather than tune another scalar.

Do not launch an autoresearch study while the local GPU is unavailable unless a configured remote
GPU target exists and obeys the 5 GB VRAM ceiling.

## Session Start Checklist

1. Read this skill and the current git status.
2. Do not discard dirty-worktree changes.
3. Verify incumbent hash, ZIP integrity, and manifest.
4. Check GPU driver and actual VRAM availability.
5. Confirm `scripts/limited` CPU/RAM behavior.
6. Resolve or dual-report the metric contradiction.
7. Inspect the latest ledger tail; do not assume this handoff is current.
8. Inventory exact fold checkpoints and caches before retraining.
9. Write a concise experiment contract and hypothesis cards before code changes.
10. Run the cheapest falsifier first.

## Final Deliverables

Do not declare completion until all applicable items exist and are checked:

- Deterministic submission ZIP.
- Full SHA-256 and embedded member manifest.
- Machine-readable training and evaluation provenance.
- Exact candidate-versus-incumbent comparisons on compatible rows.
- Both metric definitions reported until resolved.
- Chain-bootstrap effect, interval, class slices, and fold deltas.
- Public-score expectation with uncertainty and selection correction.
- One-vCPU runtime and memory measurement.
- Streaming/batch/native parity report.
- Independent leakage/rules audit.
- Reproduction commands and exit codes.

If the evidence does not support expected public 0.6660 after the budget is spent, deliver the best
verified candidate separately from a clear statement that the requested expectation was not met.
Never convert an unmet target into a success claim.
