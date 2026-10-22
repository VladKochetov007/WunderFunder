# Public solutions and the next submission candidates

This is source research and experiment preparation. No local tests, model imports,
inference, fitting, paid compute or portal uploads ran for this task. The current
user request authorizes preparing and publishing hypotheses; historical compute
allocations in other campaigns are not restarted. Empirical selection is presently
exploratory; a proposed comparison is not a measured result.

## What is actually available

| Source | Identity and usable evidence | Limit |
|---|---|---|
| [MarkBrezina/WunderConnectome](https://github.com/MarkBrezina/WunderConnectome/tree/5d086fc90ea0c11a39391dde702e2a4f0a47a869) | Exact current task; streaming code, `model.npz`585,768bytes and `submission.zip`546,071bytes in the published tree | Author's 300-sequence screen reports0.646531 WP and +0.000802 for scored-row head fine-tuning. Neither is a verified leaderboard result. MODEL.md and README describe different results; package-to-best-result identity is unresolved. No license surfaced. No code or weights imported. |
| [Predictorium HF model](https://huggingface.co/msrishav/wunder-fund-lob-predictorium) | MIT; ONNX recurrent weights, callback and reports | Earlier 32-input task, different clipping and evaluation population. Its local0.2846 is not a Connectome score or a usable112-input model. |
| [Earlier Wunder RNN HF model](https://huggingface.co/msrishav/wunder-rnn-gru-ensemble) | Public model card/code/weights | Next-state32-feature/R² task, not this two-target WP competition. |
| [TradeFM paper](https://arxiv.org/abs/2602.23784v1) | Source for typed numeric representation and event-model objectives | Physical trades/order-action/time inputs are not recoverable assumptions for our anonymous rows. [HF paper page](https://huggingface.co/papers/2602.23784) shows zero linked model repositories; this is not proof no weights exist elsewhere. |
| [Numerical embeddings authors' code](https://github.com/yandex-research/rtdl-num-embeddings/tree/a8fc25025c83f2321c63ff127a3bcef83bb1bfb5) | MIT, reference implementation of scalar bases from [NeurIPS2022](https://arxiv.org/abs/2203.05556) | Reusable method reference, no Connectome weights/result. No dependency installed or code executed. |

Searches covered web queries for Wunder/Alfa/Alpha Connectome, GitHub repository
metadata and HF model/dataset searches for `Alfa Connectome`, `connectome` and
`wunder`. The reviewed public results yielded one exact-task repository, not a
proven winning solution. HF exact-name searches yielded no exact checkpoint;
generic `connectome` results include unrelated neuroscience. This is bounded
search evidence, not an exhaustive nonexistence claim. The private team archive
is a separate source, not a public solution. Preserve licensing and competition
provenance before any future third-party reuse; current proposals use ideas only.

The [official data specification](https://wundernn.io/connectome/docs/data_overview)
defines112 inputs, two i0 targets and20,000-row sequences. Column indices do not
define depth order. The callback has no labels or scoring mask. Do not transplant
a physical-book convolution, bid/ask pairing, future-label AR input or a predecessor
task's mask/clipping convention without a separately justified adaptation.

## Submission triage

The [portal](https://wundernn.io/connectome/submit) was read in authenticated
background Chrome on October22. H512 `0equo922`, filename
`b300_h512_cp5_tl3_20261021.zip`, scored **0.66667**,53m47s,672MB. TL3
`1lmehfi3` scored **0.66832**,39m12s,656MB. H512 passed the observed runtime
envelope but lost0.00165 on public transfer. This single result does not refute
width or provide an optimization target for a leaderboard weight sweep.

| Candidate | Evidence | Next concrete action |
|---|---|---|
| Existing TL3 | Scored team endpoint0.66832 | Retain as serving comparator; unknown ancestry prevents relabeling its replay as clean CV. |
| Fixed oldZ4/WMSE75/25 FP32 | Exposed F0 point delta+0.00035747 vs oldZ4; interval crosses zero | Obtain WMSE ep1/provenance with exact hashes, build and grade native ZIP. Local preparation commit69633e47 exists, no ZIP. Linux runtime/one-core acceptance remain open. |
| Strong-parent i1-history follow-up | PR33's small-model fixed90/10 mixture has positive two-cell deltas | Bind clean strong parents and preserve parity before new fits. Do not submit the weaker standalone history model or silently tune mixture weights. |
| Recovered trajectory mean / Kaggle B16 | PR33 exposes recovery and exact endpoints | Retain as diagnostics; no stronger full submission established. Do not spend quota merely because an artifact exists. |

## Concrete hypothesis choices

1. **Numerical basis, teacher representation.** [Typed-bin packet](../../experiment_drafts/typed-bin-feature-encoding/README.md):
   six total fits, uniform versus quantile spacing as the primary matched contrast,
   raw reference as secondary. Child of existing PR8/31, not a new family.
2. **Mask-aligned readout, student objective.** [Draft PR36](https://github.com/VladKochetov007/WunderFunder/pull/36):
   four head-only fits over the same fitting-side valid components, changing only
   mask weighting. Child of H-023/H-030/H-035; direct actual mask is never an input.
   Actual parent checkpoint/ancestry bindings and exact commands remain missing.
   Canonical freeze/mask flags already exist; disable inherited propensity in both
   arms because it overrides the mask branch.
3. **Strong-parent i1 complementarity.** Continue [PR33](https://github.com/VladKochetov007/WunderFunder/pull/33)
   rather than create a duplicate: preserve the fixed blend and compare the full
   endpoint on source-qualified cells. Better small-model mixture quality is not
   a direct public-score forecast or proof of the history mechanism.
4. **Observable-event teacher objective.** Continue [PR31](https://github.com/VladKochetov007/WunderFunder/pull/31)
   with the [PR32 trainer](https://github.com/VladKochetov007/WunderFunder/pull/32):
   future events are training targets only, never inference inputs. Separate its
   fitting budget from the representation screen above and from optional KD.

## Fresh team measurements

At [PR33 head f8f4414](https://github.com/VladKochetov007/WunderFunder/blob/f8f4414fdd476d2799571c350a57d795f43fad12/research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md),
the completed six-fit,10%-component i1 screen gives fixed90/10 history-mixture
deltas **+0.00035953** on F0/1001 and **+0.00027145** on F2/1002. Mean is
**+0.00031549**. History alone loses both; masked-i1 also contributes in one cell.
Cold history activates156 additional randomly initialized columns, so changed
initial function is an alternative explanation. The existing warm-parent follow-up
must demonstrate exact output parity after zero extension before attributing a
new gain to history. Its actual-parent parity, fitting and native callback are pending.
The latest repository snapshot admits a dynamics profile; this is not a live
process-status check and does not authorize any execution from this task.

[PR26 age](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6331586358)
reports mean−0.00021730 across six10%/25% cells.
[PR27 innovation](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6331586689)
reports mean+0.00045846, concentrated in10% fits;25% mean is−0.00002895.
These are exposed-fold, partial-data screens. A four-fit full-membership
confirmation at F0/1001 and F2/1002 belongs in PR27, not another duplicate draft.
Two matched arms and fixed preprocessing isolate innovation versus the existing
moments control; fidelity sensitivity is the question, not an assumed full-data gain.

The submission work should first finish an existing supported endpoint and its
native grade. The new scientific comparisons remain unrun proposals with explicit
dependencies; they do not produce new submission files by themselves.
