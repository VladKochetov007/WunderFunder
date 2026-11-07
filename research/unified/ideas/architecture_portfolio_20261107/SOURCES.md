# Sources, transfers and existing work

Source capture is read-only; [raw-response hashes](SOURCE_RECEIPTS.json) identify
the local observations. The GitHub connector returned404 for the private repo;
the already-authorized native API access succeeded as SemyonEpanov. No message,
PR or release was written. User profile data and the raw literature search output
remain local/untracked. No training data was accessed.

## Current task and evaluator

- [Official data overview](https://wundernn.io/connectome/docs/data_overview):
 112 raw features,20k-step independent streams, shuffled sequence IDs, current
 and past rows only. This agrees with `src/wnn/data.py`.112+99 derived causal
 channels form the repository's event211 view. Train has no `is_scored`.
- [Submission guide](https://wundernn.io/connectome/docs/submission_guide):
 CPU-only1vCPU,16GB RAM,60minutes,20MB ZIP; process every row/reset per sequence.
- [Current FAQ](https://wundernn.io/docs/faq): external data prohibited; public
 open-source pretrained models require no external market data. This does not
 establish eligibility of any particular foundation model's training corpus.
- [ADR0044](../../../../specs/adr/0044-chronological-primary-validation.md): our
 stricter official train/full-valid primary policy is binding. The previous
 Predictorium statement is not independent Connectome timestamp evidence.
- `src/wnn/gmetric.py`: separate two-target clipped global WP on all eligible
 pooled rows, then mean across targets. No per-sequence averaging or mask input.

## Current competition/project evidence

- [PR61 terminal status](https://github.com/VladKochetov007/WunderFunder/pull/61#issuecomment-6639856413),
 updated4 November23:47UTC:10/12 completed fits, two completed Transformer
 teachers, two unstarted TF students and a profile-identity failure. It is an
 operator report, not a newly independently recomputed score.
- [PR67 source audit](https://github.com/VladKochetov007/WunderFunder/blob/f0f48926c0dd19457ab4a58bf2ebd7d751f3b085/research/experiment_drafts/tl3-transformer-complement-20261106/SUBMISSION_AUDIT.md):
 dated public TL3/W18 observations and clean validation anchors. The exact fetched
 file is retained with Git/blob/content identities. These are6 November portal
 observations, not a live7 November leaderboard claim.
- [PR67](https://github.com/VladKochetov007/WunderFunder/pull/67): implemented
 label-versus-fixed-blend objective, official train-only manifest, zero fits.
- [PR65](https://github.com/VladKochetov007/WunderFunder/pull/65) /
 [PR66](https://github.com/VladKochetov007/WunderFunder/pull/66): separately
 proposed SDPA gate/SwiGLU; zero reported profiles/fits. Their F0/F2 recipes need
 prospective validation-policy adaptation, not blind execution.
- [Published historical Mamba-2 adapter](https://github.com/VladKochetov007/WunderFunder/blob/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb/src/wnn/mamba2_regressor.py):
 inspected source, pinned mamba-ssm2.2.4, explicit chunk/convolution/SSM state.
 The adapter explicitly rejects CPU tensors and has no admitted target-device
 evidence here. Reuse/port it rather than claiming a CPU package already exists.

## Primary literature: facts and proposed transfer

| Source | Source-supported component | Proposed transfer / limitation |
|---|---|---|
| [Griffin](https://arxiv.org/abs/2402.19427) | Gated linear recurrence combined with local attention | Whole-prefix state plus exact recent retrieval; task-specific benefit unknown |
| [Mamba-2/SSD](https://arxiv.org/abs/2405.21060) | Selective SSD recurrence and hybrid attention experiments | Baseline long-memory operator; language results are not WP/CPU evidence |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | Revised discretization, complex state update and MIMO | Separate higher-risk state-tracking teacher; already an older repository proposal |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Gated erasure plus targeted delta writes; local-attention hybrids | Alternative memory edits if the SSD baseline is informative |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | Distinct erase/write gates in a technical report | Later operator ablation; no new kernel/CPU admission was done here |
| [PatchTST](https://arxiv.org/abs/2211.14730) | Temporal patch tokens; paper also uses channel independence | Transfer completed patches only; audit per-step causality and interactions |
| [TLOB](https://arxiv.org/abs/2502.15757) | Feature/time attention, fixed-window classification and bilinear normalization | Group/feature stem idea; normalization/head require causal streaming adaptation |
| [Gated attention](https://arxiv.org/abs/2505.06708) | Query/head-dependent sigmoid after SDPA in tested LLMs | PR65's exact controlled intervention; no local quality evidence |
| [Numerical embeddings](https://arxiv.org/abs/2203.05556) | Piecewise-linear and periodic scalar embeddings | Fit-only encoder possibility, already explored in earlier repo forms |
| [TabM](https://arxiv.org/abs/2410.24210) | Parameter-efficient MLP ensembling on tabular benchmarks | Causal-statistics regressor/head with a different bias; temporal WP untested |
| [Event History Over Scale](https://arxiv.org/abs/2610.02917) | Compact L3-event encoder with a slow-context extension | Transfer multi-rate context, not unavailable L3 messages, their CPU numbers or F1 gains |

The last paper's inputs include individual order submissions/cancellations,
which our snapshot features do not supply. A difference feature is not a recovered
L3 message. Generic DualClock is also not new: `TwoClockModel` and
`DualClockModel` exist in `src/wnn/nn.py`, with negative historical evidence in
the prior reviews. Do not use hard output holding or treat old event heuristics
as exact target-change laws.

## Recorded random-source reframe

[SOURCE_SELECTION_R1.json](SOURCE_SELECTION_R1.json) freezes a six-paper discovered
primary-source pool and a single stdlib Random draw with seed20261107. It selects
PatchTST. This is context diversification, not representative literature sampling.

The inspected passage is section3.1, Model Structure, specifically the patching
and instance-normalization paragraphs in
[the original paper](https://arxiv.org/html/2211.14730v2#S3.SS1).
Transfer: coarsen the memory clock with completed subseries tokens while retaining
an immediate per-row branch. Reject whole-sequence normalization and the default
channel-independent forecasting interface as unqualified replacements for our
stateful two-output regressor. The independent multiscale note defines the exact
publication schedule and a parameter-matched control. Both the repository's old
PatchTST proposal and this new causal adaptation remain distinguished.

All proposed dimensions, ranks, compute allocation and predicted observables are
our design choices. The cited papers' reported gains do not forecast our metric.
