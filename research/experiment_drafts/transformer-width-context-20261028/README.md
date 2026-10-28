# Transformer quality: width versus context

**[Draft PR #48](https://github.com/VladKochetov007/WunderFunder/pull/48); no fits or receiver profiles have run.**
Implementation commit: `06ef176d0aed42fdfac1c46395451a58aae73374`. This study prioritizes the
standalone Transformer. Six fresh Transformer fits, zero GRU/student fits.
H-TRANSFORMER-WIDTH-CONTEXT-20261028 / E-TRANSFORMER-WIDTH-CONTEXT-20261028-R1.

This branch builds on [draft #47](https://github.com/VladKochetov007/WunderFunder/pull/47)
at `f0d8da40fb545a630e5f042bfcc3b3360784177e` and targets `research-team`. It reuses
the source-holdout producer and receiver runtime from #47. Merge that dependency
first to narrow this PR's diff, or review the incremental comparison from that
commit. No merge or launch is performed by publication. #47's distillation panel
remains a separate future allocation, not part of this six-fit study.

## Frozen panel

| Role | Hidden × layers | Heads | Attention window per layer | Seeds |
|---|---|---|---|---|
| baseline | 768 × 16 | 12 | 1024 | 1001, 1002 |
| wide | 1024 × 16 | 16 | 1024 | 1001, 1002 |
| context | 768 × 16 | 12 | 4096 | 1001, 1002 |

Width retains head dimension64 and tests model capacity. Context changes only
the sliding attention window, testing access to more past representations.
Both use a common fresh baseline; no joint width+context arm yet. Six total new
scientific fits, no reused completed controls. #47's teachers are unrun; they
cannot be counted as available controls. A later audited adoption of this study's
baseline into #47 could avoid redundant teacher fitting, but that adapter is not
implemented or implicitly authorized by these commands.

Shared settings: raw112, five complete epochs, FINAL5 evaluation only, Muon +
AdamW peakLR1e-4, warmup5%, cosine floor0.1, weight decay0.01, dropout0, clip1,
W2, batch128/rank, chunk512, accumulation8, efficient attention, compiled blocks,
BF16/BF16x9 with FP32 masters/gradients/loss/state. `final_only` retention keeps
safe resumable state and one final model, not every epoch's weights. Exact
configs are in [configs](configs). These are proposed batches, not measured fits.

Equal epochs and batches match the data/update budget, **not FLOPs or wall time**.
Five epochs cannot settle the asymptotic value of a larger model or its optimum
LR. #43 peaked at epoch4 under a different split; FINAL5 is a predeclared bounded
endpoint, not a claim that epoch5 is best. A negative result narrows this recipe.

## What longer context actually means

The window is per layer. A16-layer stack has a theoretical input receptive field
up to `1 + 16 × (window − 1)`, subject to the available sequence prefix. KV state
is detached after every512-row training chunk. Window4096 therefore does not
provide4096-step differentiable backpropagation or full-sequence gradients.
State can also carry representations produced before an optimizer update.

Persistent FP32 KV alone is analytically about12/16/48GiB per rank for the three
arms at batch128. This excludes activations, casts/copies, optimizer, compilation
and NCCL. Actual admission needs fresh profiles for every cell. Do not inherit
#43's memory approval or silently reduce a failing batch. See [review](REVIEW.md).

## Validation and relationship to team work

The [shared source protocol](../source-holdout-transformer-kd-20261028/README.md#data-and-interpretation)
fits eligible train-source rows only and evaluates every valid-source sequence.
Any train component linked by the feature-only graph to valid is excluded.
Preparation covers every physical group, including train528; malformed data fail
instead of being silently omitted. Normalization is fit-only, initialization is
fresh, inputs are causal and states reset per sequence.

One source split and two seeds are not independent folds. Full valid is already
exposed to team development, chronology is unverified and the linkage graph is
approximate. No best-epoch search is performed on full valid. Use canonical pooled
clip2 WP, official `is_scored` mask after99 warmup rows, both targets and all
six fixed-final endpoints. Paired component intervals condition on fixed fitted
models; they do not include training uncertainty or correct for the broader search.

[PR #44](https://github.com/VladKochetov007/WunderFunder/pull/44) already prepares
LR/dropout/decay variants and was stopped with zero scientific fits. This study
does not restart it. The existing1.54B R4 plan includes a context ladder; this is
a compact executable source-separated comparison of width and context, not a claim
to invent context scaling. #43 inner-valid scores are motivation, not matching
controls or comparable full-valid grades.

For each candidate, mean paired WP gain≥0.001 and nonnegative gains at both seeds
signals follow-up. Smaller/mixed gains remain uncertain; negative results do not
close width/context directions. Record target tradeoffs and resource cost too.
Select the next scale/horizon study from this evidence; no automatic1.5B launch,
distillation, submission or quantization occurs here. Compression is a later
serving question and need not choose GRU as its student architecture.

## B300 commands

Use a clean checkout of the reviewed full Git commit and an already provisioned
Linux B300 Python environment. Source an edited copy of [receiver.env.example](receiver.env.example)
outside the checkout. Choose two free GPUs and a fresh run root, separate from data
and the checkout. All commands are prepared for the receiver; none ran locally.

```bash
cd /srv/WunderFunder
source /srv/experiments/transformer-quality.env
PACKET=research/experiment_drafts/transformer-width-context-20261028
COMMON=(--root "$WNN_TRANSFORMER_ROOT" --data-root "$WNN_DATA_ROOT" --expected-commit "$WNN_TRANSFORMER_COMMIT")
bash "$PACKET/run.sh" prepare "${COMMON[@]}"
bash "$PACKET/run.sh" accept "${COMMON[@]}"
```

These commands prepare complete source caches/split and run explicit correctness
acceptance on the receiver. The shared runtime binds study ID, config, data,
source, helper code and receiver tests. KD and quality roots cannot substitute
for each other. Old roots remain tied to their original commit; do not switch an
existing run to this revised runtime.

For **each explicit cell** (`baseline_s1001`, `wide_s1001`, `context_s1001`, then
the corresponding three `s1002` cells), run a fresh12-update profile and inspect it:

```bash
CELL=baseline_s1001
bash "$PACKET/run.sh" profile "${COMMON[@]}" --cell "$CELL"
REVIEWER=receiver-owner
HOST_RAM_GIB=128  # Replace with actual RAM allocated to this job.
bash "$PACKET/run.sh" admit "${COMMON[@]}" --cell "$CELL" --reviewer "$REVIEWER" --host-ram-gib "$HOST_RAM_GIB" --max-vram-fraction 0.8
bash "$PACKET/run.sh" fit "${COMMON[@]}" --cell "$CELL"
bash "$PACKET/run.sh" evaluate "${COMMON[@]}" --cell "$CELL"
```

No loop/queue advances automatically. Fresh profile approval and real RAM/VRAM/
scratch headroom are mandatory. Cache generation is rejected for this study.
For interrupted safe-state fits use `resume`; for completed exports use `score`,
and `seal` can finish a missing evaluation receipt without repeated inference.
Existing scientific evidence is never overwritten. See shared runtime for time
limits, cleanup and exact resume identity checks.

After all six endpoints complete:

```bash
"$PYTHON_EXECUTABLE" "$PACKET/compare.py" "${COMMON[@]}" --output "$WNN_TRANSFORMER_ROOT/comparison.json"
```

The report contains all scores, four candidate-minus-baseline seed deltas,
conditional intervals, model parameter counts and memory/checkpoint-size evidence.
Profile duration includes startup; it is not a full-fit or serving benchmark.
Return immutable config/source/data receipts, tests/profiles, complete run logs,
terminal states, predictions, grades and comparison, including failures. Archive
small evidence separately from weights/arrays with SHA256 and storage locations.

Files: [PR.md](PR.md) answers the eleven experiment questions; [recipe.json](recipe.json)
freezes allocation; [run.sh](run.sh) selects the teacher-only study;
[compare.py](compare.py) reports the full panel; [REVIEW.md](REVIEW.md) and
[VALIDATION.json](VALIDATION.json) distinguish static preparation from unrun runtime
acceptance. Shared receiver helpers live in the #47 packet and remain source-bound.
