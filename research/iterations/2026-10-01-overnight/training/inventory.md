# Two-epoch anchor inventory and training branch

Read-only checkpoint inventory was taken from `/home/grinchenkoeg/WunderFunder/artifacts/runs`. The original files were not modified. [Machine inventory](checkpoint-inventory.json) records 24 checkpoint sizes, SHA256, full args, tensor shapes and historical histories. Historical histories are provenance, not new candidate readouts. Root copied the eight OOF parent files to the isolated creative workcopy with SHA verification.

| Slot | OOF parent | Existing one-epoch head | Final V1 two-epoch head | Shape and schedule |
|---|---|---|---|---|
| Ordinary seed3 | `ref_f0`, `ref_f1` | `ft128_f0h`, `ft128_f1h` | `fin128_s3h2` | h128, chunk2000 |
| Ordinary seed4 | `refs4_f0`, `refs4_f1` | `ft128s4_f0h`, `ft128s4_f1h` | `fin128_s4h2` | h128, chunk2000 |
| Distilled seed3 | `d256_f0`, `d256_f1` | `ft256d_f0h`, `ft256d_f1h` | `find256_s3h2` | h256, chunk2000 |
| Short chunk seed3 | `c500_f0`, `c500_f1` | `ftc500_f0h`, `ftc500_f1h` | `finc500b_s3h2` | h128, chunk500 |

Every OOF parent is read from `model_ep3.pt`; every final V1 head has two epochs and starts anew from its corresponding parent ep3. No two-epoch OOF heads exist in the inspected run inventory. A one-epoch head plus one epoch is not equivalent to a two-epoch restart because OneCycle's total steps change. All these models use 211 inputs, two GRU layers, linear input projection, no dropout and 23 output channels (two targets plus the existing structural and next-event auxiliaries). Distillation belongs to the d256 parent training; its head FT has no distillation teacher.

The full train-source parent recipe uses train+other validation folds, propensity weighting, EMA4, since weighting. Head-only FT uses validation sequences outside the held-out chain fold, exact scored mask after warmup, batch16, lr5e-4, AdamW wd.01, fp16 fast path, no EMA, no since weighting and no propensity. Structure/next auxiliary constructor coefficients remain .3/.3. The unchanged `gcorr` loss is pooled within each minibatch×chunk; the scoring metric is separately pooled globally. Full 20k causal state is replayed per sequence, even though gradients truncate by chunk. Only pure held-out chains enter the root's primary scorer because parents include train sequences, some of which have validation neighbors.

## Executable plan

[plan.json](plan.json) supplies argument vectors for eight two-epoch anchors and four aux-off treatments. Run names match the root scorer: `on_ctl128s3_f{0,1}`, `on_ctl128s4_f{0,1}`, `on_ctld256_f{0,1}`, `on_ctlc500_f{0,1}`, `on_aux128s3_f{0,1}`, `on_aux128s4_f{0,1}`. The trainer writes only new `artifacts/runs` directories and refuses nonempty destinations. It preserves all auxiliary output tensors and graph computations at zero scale. Controls and candidates have identical initial parameters, parameter groups, masks and schedule. Suppressed validation advances the same CPU RNG draw as the original sequential DataLoader, so it does not alter the second-epoch sampler order. Its numeric zero is explicitly tagged as a placeholder, not a score; only fixed `model_ep1.pt` is graded. `model.pt` is also saved from the last epoch, never selected on a held-out score.

[predict_fixed.py](predict_fixed.py) writes only inside the campaign's own cache. It fills the complete validation row-index space with NaN first and predicts the requested fold, allowing the scorer to reject accidental cross-fold reads. Each `.inputs.json` records checkpoint and output SHA256, exact row indices/groups, source-order hash, chain-fold hash, epoch, elapsed time, and zero metrics computed. It inherits the base causal prediction preprocessing and fp16 batched inference. Runtime, stream and export parity remain separate required checks before submission; the synthetic test alone does not establish those.

## Controls and risks

The initial [independent hypothesis](hypothesis.md) was written before receiving the other actor's implementation. The current gradient evidence gives no generic conflict justification; this is a cheap auxiliary-removal intervention. Expected score direction is a small positive serving increment if auxiliaries constrain useful head adaptation, or negative if they regularize it. Root's fixed gate is +.0003 pooled with neither fold negative, with every seed and slot effect retained. Existing negative gcorr+wMSE, full-trunk FT and running-normalization recipes are not repeated.

[selftest-aux-result.json](selftest-aux-result.json) passed eight CPU-only checks, including exact default/main-only gradient identities, zero-versus-None private auxiliary gradients, unchanged frozen trunk after an AdamW step, strict checkpoint loading, unchanged two-epoch sampler order, and byte-exact declared source transformation. [trainer-source.json](trainer-source.json) records base hash `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f` and frozen implementation hash `f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f`.

Historical GPU logs imply roughly 15–25 GPU-minutes for 12 head fits plus prediction/startup/I/O; shared-host load can raise this. No hidden-state caches are needed; predictions total about 3.6GB. Resource limits and launches belong to root. In-sample V1/final-head histories are not held-out evidence. All local folds were exposed previously; paired chain intervals condition on fitted models and do not resolve optimizer variation, selection or hidden-test shift.
