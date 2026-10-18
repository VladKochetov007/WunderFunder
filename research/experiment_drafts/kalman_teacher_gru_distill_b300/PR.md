# [Experiment] E-KALMAN-LOCAL-TREND-B300-20261018: causal smoother GRU teachers and distillation

## 1. Status, lineage and owner

This is an implementation and profile-preparation draft, not a completed experiment. It adds a differentiable local-trend feature module, matched EMA and zero-feature raw arms, a strict-FP32 adapter to the canonical modular trainer, adapter-aware prediction reconstruction, and adapter/provenance checks in the existing distillation-cache builder. The source base is `research-team`; the branch and final commit are recorded by the PR. Hypothesis `H-KALMAN-LOCAL-TREND-20261018`; experiment `E-KALMAN-LOCAL-TREND-B300-20261018`. No B300 profile, PyTorch smoke, task fit, student cache, task grade or performance claim has been produced. A separate NumPy synthetic receipt is preserved and labeled as such.

The local inference-only motivation is now independently graded: on 16 selected exposed-F0 groups, replacing four raw inputs with local-trend posterior levels changed pooled WP by +0.000338 versus raw (12-component conditional 95% interval [−0.000019,+0.000716]); the random-walk change was +0.000119. This is one frozen checkpoint, not a trained smoother or student result. The independent scorer revalidated alignment and reproduces the grade. See the separately indexed [local probe packet](https://github.com/VladKochetov007/WunderFunder/blob/main/research/campaign_2026-10-18/online_features_r1/kalman_filter_probe/RESULTS_R2.md) in the main research checkout.

Checkpoint loading is strict for both paths: the Kalman adapter is reconstructed
from checkpoint metadata and its state dict, while standard family teachers
load their saved state dict into the canonical model before prediction export.
This is required for valid teacher caches; shape and finiteness checks alone
cannot detect random initialized predictions.

For adapted checkpoints, provenance binds the hashes of both
`src/wnn/kalman_local_trend.py` and `src/wnn/nn.py`. Prediction export checks
the current implementation against those training hashes before inference;
the sidecar records current hashes and the prediction-driver hash. Cache
construction independently checks the current files against both checkpoint
provenance and prediction sidecar before it accepts the teacher.

## 2. Why it should work

A local-linear-trend filter can expose a causal estimate of level, slope and innovation uncertainty to the GRU, providing a compact inductive bias for changing latent regimes. The strongest alternative is that ordinary smoothing or the extra input projection explains any change. For both the four-channel local pilot and full 112-channel scope, the primary control is an EMA whose gain is the fixed scalar random-walk Kalman steady-state gain; the fixed-Q/R local-trend arm separates trend-state structure from learned Q/R. A scope-matched raw arm appends zeros so each four-arm set has the same GRU width and projection shape. The learned filter uses positive scalar Q/R parameters tied by state family across selected channels. All arms retain the original raw input.

## 3. Compute advantage and scaling path

The competition payoff is through a better fixed-size GRU student, not a
teacher-only score: a Kalman teacher must improve the matched four-family
student-cache endpoint, then retain a useful quality/runtime/package-size
tradeoff in the final served pipeline.

The filter adds four features per selected raw channel and a small number of learned scalars, but its eager Python time loop may erase that nominal arithmetic advantage. The 112-channel case adds 448 inputs to the H1024 projection and may be activation-memory intensive. An actual B300 profile at the full planned shape must precede scientific fitting; no time estimate is supplied. Each arm uses fresh H1024×2 weights in this primary attribution experiment. A clean-parent strong-model continuation is a separate future factor: compare it against a matched fresh model and preserve identical parent ancestry across feature treatments before claiming a continuation benefit. If the teacher has value, the included path replaces one clean same-fold teacher family in the existing four-family student cache. Larger GRU teachers, teacher-only/student-only/combined feature placement, and distillation at larger context are later scaling experiments. Teacher DDP is not implemented or required for the independent small-GRU cells.

## 4. Frozen comparison

The task target is the original two competition outputs under the canonical WMSE training objective; there is no next-input reconstruction target. Direct teachers use fresh H1024×2 GRUs, batch 16, chunk 2,000, AdamW at 0.001, weight decay 0.01, gradient clipping 1, two epochs, OneCycle over the exact update budget, and fixed checkpoint `model_ep1`. Students use fresh H256×2 GRUs, the existing two-epoch fixed-`model_ep1` cache recipe, and `--distill-alpha 0.5`. Use canonical fit-only global normalization, identical sampler/order semantics within paired cells, and sequence state reset at each source-qualified sequence while carrying state over chunks. FP32 is required for weights, inputs, recurrent/filter states, loss, gradients and optimizer; AMP, GradScaler and CUDA TF32 are disabled. The official metric may use FP64 accumulators. No per-sequence or future-row transform is added. The code and exact commands are in [recipe.json](https://github.com/VladKochetov007/WunderFunder/blob/experiment/kalman-teacher-gru-distill-b300/research/experiment_drafts/kalman_teacher_gru_distill_b300/recipe.json).

## 5. Thirty paired runs

Every arm uses the same 30 exposed fold/seed cells: F0/F1 with seeds 1001–1008 and F2/F3 with seeds 1001–1007, for 8/8/7/7 cells. The direct teacher matrix is raw-zero-pad, steady-state EMA, fixed local trend and learned local trend at `pilot4` and `all112`; this is eight arms. The student matrix has a matched existing-cache control plus pilot4-Kalman and all112-Kalman source-replacement arms; this is three more arms. Total: 11 arms, 330 scientific fits, scheduled as 42 waves of up to eight independent fits. Launch one fit per GPU; no DDP command is supported. The same seed identities pair arms within each fold. All four folds have prior exposure; this is robustness evidence, not a pristine holdout. `pilot4` is the local screening scope, while `all112` is the expanded channel-scope comparison. Profile the all112 learned arm before fit admission; if the profile passes, run both predeclared matrices in full. Pilot quality does not gate all112 inclusion.

## 6. Validation and significance

For seed variability, use NumPy PCG64 seed `20261018` for 10,000
percentile-bootstrap replicates: draw eight seed identities with replacement
from 1001–1008, apply the same multiplicities in every fold, renormalize over
each fold's available seeds, and redraw if any fold has zero available
multiplicity. For evaluation-unit variability, use separate PCG64 seed
`20261019` for 10,000 paired source-component bootstrap replicates within each
fold, sharing component multiplicities across arms and seed cells. Report the
two marginal 95% intervals separately; both condition on the exposed folds and
fixed fitted endpoints.

Grade with unchanged `global_wp_clip2_v1`, fixed scored rows and its original clipping and weighting. Primary contrasts are learned versus EMA and learned versus fixed local trend separately for both scopes (four tests). Secondary contrasts are learned versus same-scope raw-zero-pad control and each Kalman-family student versus the shared matched-cache student (four tests). If all are reported, apply Holm correction across all eight. Use the equal-fold mean of paired seed means, a 10,000-replicate crossed seed-identity bootstrap with shared seed multiplicities across folds, and a separate 10,000-replicate paired source-component bootstrap within each fold. Report fold and seed variation separately; intervals condition on exposed folds and fitted endpoints. Profiles, synthetic checks and PyTorch parity checks are engineering evidence, not quality comparisons.

## 7. Leakage and coverage

The canonical raw/event input width is 211; the adapter rejects alternative feature assembly. The four pilot channels `[0,3,52,55]` are anonymous indices and carry no physical interpretation; pilot-only findings do not extrapolate to 112 channels. The `all112` scope is explicitly separate. All features are causal and state resets by sequence. Canonical input normalization is fitted from the exact training-view loader and reused from checkpoint statistics at inference. No targets enter feature construction. Before cache building, freeze the four teacher family run IDs, epochs, split hashes, coefficients and checkpoint hashes per cell. The existing builder checks clean-fold fit membership, prediction sidecars, row ordering and checkpoint bytes. For treatment caches only `samefold_baseline` is replaced, at its unchanged coefficient. Shared ancestors are not independent replications. Previously exposed folds remain exposed.

## 8. Decision-changing result

A mean gain around `+0.0005` weighted Pearson is a provisional reference for prioritizing a follow-up, not a kill threshold. A useful result also needs compatible behavior across folds and acceptable measured resource cost. A zero-crossing interval is inconclusive. Negative results apply to the tested state family, channel scope, schedule and teacher/student path, not all Kalman or smoothing mechanisms. If the local pilot differs from full112, report the interaction as channel-scope dependence rather than treating the smaller result as confirmation. Continue with a clean-parent warm-start/fresh factorial, larger teacher, or student feature placement only when the profile and evidence justify it.

## 9. How to run

Use the exact branch source with the compatible project environment. `WNN_PROJECT_ROOT` must be a writable project namespace containing supplied `train.parquet`/`valid.parquet`, target indexes, source-qualified fold manifests, run outputs and caches. Mount the code checkout read-only at the exact source SHA. Place cache, predictions, logs and checkpoints on physical worker storage, not `/tmp` or `/dev/shm`. Run the full112 H1024 profile gate on the actual eight-B300 host and record driver/runtime versions, host/GPU peak memory, finite optimizer-update latency, filter/GRU forward-backward throughput and input overhead. If resized, freeze the new dimensions/batch and reprofile every direct arm at both scopes before admission. The packet has a synthetic smoke command, teacher training command, prediction export commands, cache builder command and standard student command in [README](https://github.com/VladKochetov007/WunderFunder/blob/experiment/kalman-teacher-gru-distill-b300/research/experiment_drafts/kalman_teacher_gru_distill_b300/README.md) and [recipe](https://github.com/VladKochetov007/WunderFunder/blob/experiment/kalman-teacher-gru-distill-b300/research/experiment_drafts/kalman_teacher_gru_distill_b300/recipe.json). The smoke and every fit remain unrun. Use one independent process per GPU; no torchrun/DDP route is implemented. Student fits are blocked pending a clean per-cell four-family registry.

## 10. Evidence to return

For each optimizer update preserve attempted/successful steps, progress, loss, LR, preclip gradient norm and clipping/nonfinite counts. Each cell must bind branch/source/config hashes, software/hardware identity, split and membership hashes, teacher ancestry, optimizer-update budget, checkpoint/prediction hashes, official score, paired deltas, phase times and peak host/GPU memory. Archive evidence and model weights separately, including file and archive SHA256 and private storage location in the run manifest. Retain every valid cell and negative/inconclusive result. The local NumPy synthetic receipt and source are included in `evidence/`; the receipt reports 841,376 KiB peak RSS, so requested support-resource verification is unconfirmed. It contains no task data, labels or task model result.

## 11. Results and trace

No task cell has been admitted or run from this branch; teacher/student quality and B300 runtime are unknown. A source-only independent review initially found that the adapted-source sidecar echoed checkpoint metadata rather than validating current filter code and omitted the imported GRU core. The follow-up fix binds both source files through training provenance, prediction export and cache construction; the independent source-only review cleared that path. The [review summary](https://github.com/VladKochetov007/WunderFunder/blob/main/research/unified/reviews/KALMAN_B300_DRAFT_REVIEW_20261018_R1.md) is indexed in the main checkout. The local fixed-GRU screen is separately recorded in the [Kalman results note](https://github.com/VladKochetov007/WunderFunder/blob/main/research/campaign_2026-10-18/online_features_r1/kalman_filter_probe/RESULTS_R2.md) and does not establish trained-feature value.

The three source records are [Backprop KF](https://arxiv.org/abs/1605.07148), [KalmanNet](https://arxiv.org/abs/2107.10043) and the [AI-Aided Kalman Filters survey/tutorial](https://arxiv.org/abs/2410.12289). These motivate differentiable and hybrid filtering as a direction; they do not establish a Challenge2 gain. The full112 actual B300 profile and per-cell clean teacher registry are still required before fitting. This PR is an implementation handoff, not fit authorization; no public score, leaderboard transfer, speedup or competitive result is claimed.
