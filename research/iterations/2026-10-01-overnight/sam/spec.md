# SAM-PARENT-01 implementation contract

Status: prepared, no empirical data read/fit/WP query by this actor. Independent review and root allocation precede any jobs. SAM rho=.05 is the only new scientific configuration. The initial blind commitment is `../architecture-scout.md`.

Train the standard h128x2, linear encoder, chunk2000 parent from scratch for4 epochs, fixed EMA checkpoint ep3. Control and candidate differ only in rho0/.05. All loss expressions, masks, auxiliary weights, data order, precision, normalization, AdamW/OneCycle settings, EMA update frequency and recurrent resets match. Then train both heads for2 ordinary gcorr epochs, frozen `inp,gru`, fixed ep1. Head SAM is disabled. Existing ref parents are not reused: their complete training provenance has not been independently established here.

Both passes start from the same detached incoming state. The first unperturbed forward alone supplies outgoing carried state. Compute teacher once; preserve it across both losses. First gradients come from `autograd.grad` on the scaled loss, followed by division by GradScaler's current scale. The perturbation is global L2 radius.05 over every trainable parameter. Compute its norm with float64 reductions to prevent finite-gradient norm overflow. Release the first graph, then evaluate/backpropagate the unchanged full loss at perturbed weights. Exact saved weights are restored in a finally block before the original single unscale/clip/AdamW/scaler/scheduler/EMA sequence. There is no added step, teacher update or carried perturbed state.

First-gradient nonfinites with enabled GradScaler cause no perturbation/second pass. Assign those same scaled gradients to parameters and delegate skip/backoff to the original outer scaler sequence. This matches the baseline's overflow path, including its subsequent schedule/EMA advance. With a disabled scaler, reject nonfinites. A second-pass overflow always restores weights and is skipped by the ordinary scaler. Exceptions restore weights and abort; never continue an unknown partially updated state.

The complete original chunk loss is a verbatim indented closure; strict argument guards exclude stateful/undeclared alternatives. `trainer-source.json` declares exact source replacements. `build_trainer.py` deterministically regenerates the isolated copy; it does not modify the original trainer. `rho=0` production takes its normal single backward path. Synthetic `force_two_pass=True` tests the rho0 equivalence of the SAM path.

No validation iteration or scoring runs during training. Consume exactly the single CPU base-seed draw of the original sequential, map-style, nonpersistent validation DataLoader after every epoch. This preserves the following epoch's shuffle. Save all fixed epoch files and a final-only `model.pt`; no metric values or best-epoch choice exist. Synthetic tests compare four epochs' train orders with actual synthetic evaluation loader iteration, including2 workers.

Plans explicitly list every command. Seed3 costs8 fits: control/SAM x folds0/1 x parent/head. Seed4 repeats8 fits only after the frozen seed3 gate. Two strided `run_queue.py` lanes own one fold each, preventing parent/head dependency races. A promoted final candidate costs2 fits and **must use1 lane**. Each parent stage's `predict_argv` performs checkpoint integrity only; head stage predicts fixed ep1 once with no metric calculation. The copied prediction script is identical to `../training/predict_fixed.py` at preparation. Artifacts are fresh `on_sam_*` namespaces; directories must not already exist.

The primary paired serving comparison uses the three fixed V1-like anchors plus either the new control or SAM in the seed-matched h128 slot, all weights.25. Parent controls are rerun, so the paired baseline is not silently identified with the incumbent. Additionally compare the fixed seed3 candidate against the unmodified V1-like anchor. Primary screen: seed3 pooled lift >=.0003 with both folds nonnegative. Promotion: both seed pooled lifts positive, every seed/fold sign nonnegative, mean pooled seed lift >=.0003, and seed3 candidate versus original anchor nonnegative in both folds. The deployment slot remains seed3; no best seed or blend search.

Source/data checks and whole-chain evaluator remain the controller/grader's responsibility. All folds are exposed. Bootstrap intervals condition on selected fitted models; do not equate them with training-seed or hidden-test uncertainty. Do not inspect intermediate-parent WP to select continuation, epochs or rho.

Resource contract mirrors root runner: at most2 GPU jobs, .34 memory fraction (<=5.5GiB on16GiB A4000),4 Torch threads and2 loader workers per lane,600% CPU quota,20GiB process-scope RAM, no swap,1800s stage timeout. If observed peak GPU memory exceeds5.5GiB, stop; do not silently change batch size. Root must enforce long-parent launch cutoff06:00UTC and final cutoff07:00UTC. SAM uses no new serving state or operations; normal export parity and full ZIP runtime verification still apply.

Preparation commands (synthetic only):

```sh
PYTHONPATH=src .venv/bin/python research/iterations/2026-10-01-overnight/sam/selftest.py --device cpu
```

Conditional execution, only after root records review/allocation and checks input hashes/GPU capacity:

```sh
.venv/bin/python research/iterations/2026-10-01-overnight/run_queue.py --plan research/iterations/2026-10-01-overnight/sam/execution-plan.seed3.v1.json --gpu 0 --lane 0 --lanes 2 --timeout 1800
.venv/bin/python research/iterations/2026-10-01-overnight/run_queue.py --plan research/iterations/2026-10-01-overnight/sam/execution-plan.seed3.v1.json --gpu 0 --lane 1 --lanes 2 --timeout 1800
```

The CUDA synthetic test is available via `--device cuda`; it is not a data fit, but still needs a free slot. This actor initially runs CPU synthetic tests only. Existing plan versions are immutable; any review fix needs a recorded new version before execution, preserving this attempt.
