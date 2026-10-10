# External ideation round 3 (risky ideas) - synthesis, 2026-10-10

Providers: Grok grok-4.7 (plan mode), Freebuff DeepSeek V4.1 Flash (no Kimi in the picker; GLM was used in round 2), OpenCode nemotron-3-ultra-free. Task: `workspace/TASK3.md`. Raw: `grok_round3.md`, `FREEBUFF_ROUND3.md`, `opencode_nemotron_round3.md`. Quality: Grok and DeepSeek sharp; Nemotron generic (it assumed unlabeled test sequences are provided - false).

## Learning-curve finding mined from our own logs (scored today, fold 1, `learning_curve_*_f1.jsonl`)
- Default recipe (4 epochs, onecycle): WP 0.644 (ep0 p123) -> 0.654 (ep0) -> 0.661 (ep1) -> 0.666 (ep2) -> 0.6656 (ep3 end). Plateau reached by epoch 2-3; mid-epoch snapshots are as good as epoch ends (noise +-0.001).
- 6-epoch run: peaks at 0.6671 (end of epoch 3), then falls: 0.662 (ep4), 0.649 (ep5). So the cliff is overfitting by memorisation (repeated sequences), not under-training. Data economy, not optimisation, limits the teacher. This supports: chunk-length gain (more, noisier steps), augmentation/regularisation that is label-preserving, and the sample-starved class 3.

## Convergent ideas
1. Cross-sequence structure (scenario library / duplicates / one long tape): Grok, DeepSeek, partly Nemotron. Checked against our notes: sequences are consecutive 20k-row chunks of 3,288 continuous streams (E-095); train and valid are cut from the same streams (85 chains span both). Serving sees one sequence at a time with reset state, so predecessor carry-in is not usable at serving; a retrieval of the train predecessor would only help the first rows. Not queued; low value because the 99 unscored warm-up rows already absorb the state. Kept as a CPU probe only (neighbour label-path correlation within class for class-3 age>16), cheap.
2. Redundant-clock / lattice (Grok 2, 3): lattice already probed (`feature_lattice_probe.py`, no discrete lattice). The fast->slow increment regression (R^2) is new and cheap (CPU).
3. Training-only moves that change what the recurrence must carry: lag-split target (Grok 4), age-split finetune midpoint (Grok 6), clock-phase auxiliary (DeepSeek 7), gate-bias evolution (Grok 8), hypernetwork h0 (all).
4. Servable weight-space merging of seeds (DeepSeek 4): the one idea that would convert the (non-servable) ensemble gain +0.0015-0.002 into a single net.

## Ranked queue (risky, behind the running set; each has a numeric falsifier)
| id | idea | cost | go / no-go (fold 1, 3 ep) |
|---|---|---|---|
| R3-1 | lag-split target: on rows 0-3 after a label change supervise the previous level (needs label-change mask in loss) | code + 1 fit | WP >= 0.676 or class-3 >= 0.453; no-go both gains < 0.002 |
| R3-2 | weight-space merge of 4 seeds (alignment by activation matching), single net | align script + 4 existing seeds (vt_f1 scratch s3-s5 same recipe) | merged beats best seed by >= 0.006; barrier < 5% |
| R3-3 | age-split finetunes (age<8, age>64) then parameter midpoint | 2 short finetunes | midpoint >= 0.674, or class-3 >= 0.452 with pooled >= 0.666 |
| R3-4 | fast->slow increment regression (CPU) | CPU only | median R^2 >= 0.25 go, <= 0.05 no-go |
| R3-5 | label-preserving sequence-level augmentation to attack memorisation (run permutation inside long class-3 runs, column-block dropout at the block level) | small trainer change | class-3 sign acc at age>256 +2 pts with pooled >= 0.667 |
| R3-6 | h0 hypernet from rows 0-32 (linear first) | small | sign +2 pts and pooled >= 0.674 |
Dropped: pseudo-labels on unlabeled test (no such data), diffusion simulator (synthetic labels already hurt), learned optimiser (plateau is data-limited, not optimiser-limited; consistent with the curve).

## Reprioritisation
Important: (1) overfitting/data economy (curve), so chunk/step-count and regularisation levers, class-3 coverage; (2) servable ensemble transfer (merge, R3-2); (3) boundary estimation, not smoothing. Less important: width/epoch/optimizer hyper-parameters (plateau), distillation chains, post-hoc inference tricks (all closed).
