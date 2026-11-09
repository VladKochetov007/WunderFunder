# Transformer optimizer/horizon screen: execution contract R1

## Authority and ownership

The user explicitly authorized exactly six fresh B300 fits: C0 Muon256/A4,
C1 AdamW256/A4, C2 Muon512/A2, each at seeds203101 and203102. All fits are
Transformer768x24,4E,global batch256,`standalone_labels_wmse_v2`,BF16/BF16x9
forward with FP32 master weights, gradients, losses and optimizer state. ROOT
alone owns the existing B300 RPC/SSH stream, source/data/profile verification,
hardware admission, eight-hour wall-clock authority, detached process lifecycle,
and final receipts. No new network stream, provider, billing, data download,
foreign process signal or competition submission is authorized.

The cohort clock starts at the first exact native profile: maximum8h total,
7h fit/profile admission plus1h reserved for full-valid evaluation, plotting and
collection. No reset, automatic retry, new seed or extra scientific fit. Each of
the six profiles is exactly50updates, at most900s and one attempt. The optional
world1/B256 engineering profile is outside the six-fit panel and cannot delay
the world2 fit admission after all six exact profiles pass.

Frozen source: commit `5d73bc2b10ae20e607e0d4b4533da85e76e27bca`, source-r2
capsule root `/data/epanov_s/tabdata/runtime/transformer-optimizer-horizon-20261108-r1/source-r2`,
manifest SHA256 `a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937`.
Only the source-capsule root is used by child commands. Historical source-r1,
frozen configs, completed results and the five repair commits remain immutable.

## Frozen panel

| Cell | Optimizer recipe | Batch / accumulation | Chunk / horizon | Seed | Pair | Fits |
|---|---|---:|---:|---:|---|---:|
| C0_muon256_a4_s203101 | Muon256/A4 control | 128/GPU ×2, A4 | 256 | 203101 | 0,1 | 1 |
| C1_adamw256_a4_s203101 | AdamW256/A4 recipe | 128/GPU ×2, A4 | 256 | 203101 | 2,3 | 1 |
| C2_muon512_a2_s203101 | Muon512/A2 horizon | 128/GPU ×2, A2 | 512 | 203101 | 4,5 | 1 |
| C0_muon256_a4_s203102 | Muon256/A4 control | 128/GPU ×2, A4 | 256 | 203102 | 6,7 | 1 |
| C1_adamw256_a4_s203102 | AdamW256/A4 recipe | 128/GPU ×2, A4 | 256 | 203102 | 2,3 | 1 |
| C2_muon512_a2_s203102 | Muon512/A2 horizon | 128/GPU ×2, A2 | 512 | 203102 | 4,5 | 1 |

Every run uses four epochs,820 updates/epoch,3,280 optimizer updates, global
batch256 and the same 1,024-row nominal optimizer window. C2 changes detach
chunk/horizon512 and uses A2. It retains detached Transformer KV state between
chunks and optimizer updates; accumulation does not extend gradient horizon.
The final partial optimizer window contains544 rows. Each config declares
`gradient_horizon_rows == chunk_rows` and telemetry cadence50.

C1's one AdamW role uses LR factor1 and head/body factors1; C0/C2 use neutral
Muon's four role factors plus head/body factors1. LR remains3e-4, global decay.01,
betas(.9,.95),eps1e-8,clip1,5% warmup, cosine schedule and floor ratio.1. The
optimizer contrast includes decay coverage: plain AdamW applies decoupled decay
to every parameter, while Muon roles exclude bias and normalization parameters.
The result cannot be described as isolating the optimizer algorithm alone.

## Data, objective and ancestry

Use the pinned chronological split manifest SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c` and already
qualified local tabdata root
`/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project`.
No new data are downloaded. Fit only on the official earlier train partition:
10,325 fitting sequences; no validation fitting members. Normalization is
`all_fit`. Optimization uses `all_after_warmup`, with causal warm-up rows
processed but excluded from loss. `seq_ix` is never treated as chronology.

The objective is standalone raw-z label WMSE v2. There is no KD, hint objective,
reference cache, training blend or correlation loss. No teacher/initializer or
other fitted parent enters ancestry. Whole sequences and official fit membership
are preserved. Full valid is a later independent partition used for primary
evaluation only, with `is_scored` after warm-up: 1,873 sequences and 3,528,273
rows. Repeated valid evaluation is exposed development feedback, not a pristine
holdout.

Primary metric remains unchanged official pooled weighted Pearson
`global_wp_clip2_v1`; all official rows and both targets are required. Canonical
prediction export and the official scorer bind each fixed E4 model to the full
valid coverage. The monitoring producer reports raw-z WP/WMSE, per-target
statistics and clipping diagnostics on the same forward, with legacy `valid/wp`
and `valid/loss` aliases equal to raw-z. No reference or mixture is fabricated.

Train-eval is128 full fitting members selected by stable source-qualified
`hash_rank_v1`; its mask is `all_after_warmup`. Valid is the complete official
population with `is_scored` after warm-up. Train-eval and valid differ in
population and mask; `generalization_gap_comparable=false`. Hash-ranked sampling
does not claim representativeness. Evaluation reports/checkpoints are fixed and
do not adapt optimization or select epochs.

## Endpoints and evidence

All epoch checkpoints are retained under `all_epochs`; endpoint policy is the
preconfigured final epoch4. Per-epoch metrics and PNG/SVG are diagnostic only.
There is no best-epoch selection, SWA, or reweighting. Every full fit requires
the exact ROOT-approved profile receipt whose source, normalized config, data,
world size, GPU pair and strict model/optimizer/scheduler/RNG round-trip identity
match that exact seed/config. Profiles never initialize or resume fits.

Return every cell's immutable config hash, source/data/split/runtime/ancestry,
four epochs and3,280 updates, epoch history, sampled optimizer-update records,
rank logs, full-valid predictions and official score, per-epoch plots, terminal
status and safe state. Every evidence/weights bundle must carry SHA256s. The
receiver refuses partial valid coverage, mismatched checkpoint/score hashes,
wrong profile/config identity or altered source. Failed/partial outcomes remain
visible; no automatic retry.

Two paired seeds are descriptive only. Report C1−C0 and C2−C0 paired deltas for
each seed and every absolute endpoint. Do not claim statistical significance,
public score gain, or a reliable training-variance estimate. C1's decay-coverage
difference is part of the named AdamW recipe contrast.
