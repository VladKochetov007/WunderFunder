# Review of the next experiment families

Reviewed 19 October 2026, read-only. I inspected the archived PR17 source tree and recipe, the local PR13 contract and outcome notes, the fresh PR9 and PR24 contracts, and the current R01 checkpoint-output contract. I did not connect to the host, import Torch, run a profile, or start a fit.

## Decision

None of the fresh proposals is ready to add as another training family while the current queue is full. PR17 has a real canonical-trainer adapter but still requires a Torch smoke, an expensive full-shape profile, and a complete fold-clean teacher registry; its proposed full experiment is 330 fits. PR9 is a helper without a wired trainer/cache path. PR24 has no current-base training command and would require porting and profiling several architecture families for 120 fits.

The local `AGENTS.md` records the selected density-four plan as one wide and one deep R01 fit plus two PR15 fits per B300 (32 scientific CUDA actors total at the cited snapshot). PR17's H1024×2 profile is a much larger shape than these lanes; this static review provides no evidence to admit it as a fifth process or substitute it into the measured plan.

The only low-marginal-training-cost candidate is a separately named R01 trajectory-average successor based on existing per-cell checkpoints. It would create no new fit jobs. Treat it as postprocessing, not a way to increase training throughput: it still requires a pinned checkpoint constructor and as many as 60 additional prediction exports for the two R01 arms. Preserve R01's fixed epoch-5 endpoints and compare each average against its already graded endpoint.

## PR17: implemented mechanism, unready scale

The local source archive `backlog_review_v2/source_packages/pr17_kalman_588b93f9c2ad.tar.xz` is pinned to commit `588b93f9c2ad7e53aac7b1e2a39f30c32047ea22` and SHA256 `236ed4277bac52266d0e5c3eb8a842d77d59a1950f36f13e419d61f58a3dc3ce`. Its recipe is in `parallel_expansion_v1/source_contracts/pr17.json`. This review read the archived `scripts/train_kalman_teacher.py`, `src/wnn/kalman_local_trend.py`, the canonical training loop, and the recipe directly from the archive; no files were extracted or executed.

The implementation does more than the PR9 feature helper: `train_kalman_teacher.py` calls the shared `run_training` with a model factory and enforces FP32; the feature adapter retains raw inputs, resets/carries filter state with the recurrent model, and provides raw-zero, EMA, fixed-trend and learned-trend arms. Prediction reconstruction and distillation-cache checks are also present. The all112 mechanism is causal in the inspected recurrence: each output uses the current and preceding observations, and state is passed across chunks.

Dependency fit is plausible but not closed: the archived `pyproject.toml` requires Python 3.11–3.13 and PyTorch `>=2.14,<2.15` from the explicit CUDA 13.0 wheel index; the operator reported PyTorch 2.14.0+cu130, while the host's original system Python was 3.9. Verify the exact active virtual-environment interpreter before launch. The version range match is not evidence that the archived environment was installed or that this model runs on the current host. No environment install or import was attempted here.

The unresolved issue is cost and readiness. The local-trend/EMA code loops in Python over every timestep inside each chunk. With `chunk=2000`, batch 16 and 112 selected channels, that is a 2,000-step eager/autograd loop per model call; all112 appends 448 inputs and raises the GRU input width to 659. The recipe itself identifies this as a runtime risk, requires a real full-length B300 profile, and records that no PyTorch smoke or B300 profile has run. Its full comparison is 11 arms × 30 cells = 330 fits / 42 eight-card waves. It also leaves the per-cell four-family teacher registry null, so its proposed student-cache comparisons cannot start simply by fitting the Kalman teacher.

Smallest defensible next gate, if this family is selected later: run the existing synthetic Torch smoke for all four feature modes on CPU; verify prefix invariance, chunk carry/reset, finite outputs and learned-Q/R gradients. Then measure the full `all112_learned`, H1024×2, batch-16, chunk-2000, 20,000-row profile from the recipe on an available B300, recording peak memory and per-update wall time. Do not estimate fit duration from the NumPy prototype or short synthetic smoke. Freeze a smaller, separately named comparison only after the measured profile and an explicit teacher registry; a pilot4 result cannot stand in for all112.

## PR13 successor: cheapest additional endpoint, with a missing constructor in this checkout

The new R01 source contract records six epoch snapshots (`model_ep0.pt` through `model_ep5.pt`) and fixes `model_ep5.pt` as the primary endpoint in `deployed_review_v4/source/code/command_builder.json`. That makes an inference-only average of epochs 3, 4 and 5 technically possible per already trained R01 cell, without touching the fit recipe or requesting new training slots.

There is relevant but weak prior evidence. `campaign_2026-10-13/strategy_execution_r1/trajectory_average_f0_r1/README.md` records two prechosen FP64 parameter averages of epochs 3/4/5, each formed within one seed trajectory with its own normalization. They were both graded against the same exposed F0 baseline: deltas were +0.000043510 and +0.000057455 WP (mean +0.000050483), with the same teacher cohort and no fresh-fold evidence. This supports a cheap replication attempt, not a quality claim or a reason to expect a larger gain on R01.

The PR13 contract is not directly runnable from the current checkout: it says it adds no code and reuses `scripts/average_trajectory_checkpoints.py`, but that script is absent from the local worktree. Do not substitute native `--swa-frac`; that is an online training average with different semantics. Before any successor export, recover and pin the PR13 constructor (or a reviewed exact implementation) and test it against the actual R01 checkpoint schema. For each R01 cell separately, require matching architecture, source/recipe/split/seed, tensor keys/shapes, and bitwise-equal normalization/nonfloating buffers across epochs 3/4/5; average only the intended floating model tensors in FP64, cast back, preserve epoch-5 metadata and parent hashes, and never mix cells, folds, seeds, wide/deep arms, or optimizer states.

There is a second integration blocker: the frozen R01 exporter explicitly requires `model_ep5.pt`, validates its hash against the original fit receipt, and the official export receipt calls that endpoint epoch 5. Replacing that file in the original run or editing its fit receipt would destroy provenance. A successor therefore needs its own versioned derived-checkpoint export adapter/workspace that keeps the original R01 fit evidence read-only, binds all three parent checkpoint hashes plus the derived weight hash, and uses the pinned predictor/grader without pretending the average was fitted. The original exporter and grader contract must remain unchanged.

If those gates pass, freeze a distinct postprocessing-only endpoint contract before reading average scores. Use the existing fixed-epoch-5 predictions as paired controls, then export/grade one derived average per cell through the new adapter and official grader. Full coverage of both R01 arms means at most 60 extra exports. The prior PR13 full-valid prediction payload was 299,680,000 bytes, but R01's fold-specific output sizes are unknown; measure actual files and disk headroom before admission rather than applying that size as a forecast. Generate and grade sequentially with physical-storage checks. This creates no scientific fit cells and should not be represented as accelerating the active fit queue. Existing folds remain exposed.

## Other fresh contracts

| Contract | Readiness for the active B300 queue | Concrete blocker |
|---|---|---|
| PR9 causal smoothing | Not ready | Source annotation says helper-only: the raw-plus-EWMA/residual path is not wired into trainer/cache production. `--online-norm` has different semantics; same-width controls and coefficient fitting are unresolved. |
| PR24 GRU / Transformer / Mamba-2 | Not ready | Contract states `launch_ready=false`, `training_authorized=false`, and no current-base fit command. It calls for a current-trainer port, strict TF32-off proof and pinned Mamba build before its 120-fit comparison. The historical one-epoch screen does not satisfy the six-epoch CP30 recipe. |
| PR4 / PR14 successors | Parent is already preparing these | Keep their new source/recipe bindings distinct. This review does not certify their launch readiness or recommend adding concurrent workers. |

## Practical queue recommendation

Do not add PR17 or PR24 processes to the live GPU cohort. Continue the measured R01/PR15 plan unchanged. If a low-training-cost follow-up is desired, prepare the PR13-derived R01 average as a separate post-fit endpoint contract; first pin the missing constructor and validate representative completed checkpoints on CPU. Admit exports only after they fit within the existing postprocessing and disk headroom, and preserve every original epoch-5 output and receipt. This follow-up remains limited to exposed development folds, with no fresh-holdout or public-transfer claim.

## Evidence reviewed

- PR17 archive manifest and recipe: `backlog_review_v2/source_packages/pr17_kalman_manifest.json`, `pr17_kalman_588b93f9c2ad.tar.xz`, and `parallel_expansion_v1/source_contracts/pr17.json`.
- PR13 contract and prior outcomes: `parallel_expansion_v1/source_contracts/pr13.json`; `campaign_2026-10-13/strategy_execution_r1/trajectory_average_f0_r1/README.md`; `campaign_2026-10-13/strategy_execution_r1/reviews/trajectory_average_f0_r1/README.md`.
- PR9 and PR24 contracts: `parallel_expansion_v1/source_contracts/pr9.json` and `pr24.json`.
- R01 checkpoint and endpoint contract: `deployed_review_v4/source/code/command_builder.json` and `deployed_review_v4/source/PANEL.json`.
- PR family implementation/readiness annotations: `backlog_review_v2/annotations.json`.
