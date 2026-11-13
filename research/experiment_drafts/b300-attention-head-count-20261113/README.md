# Six or twelve temporal attention heads at width 768

The screen asks whether a wider query/key per head improves fixed-E4 weighted Pearson correlation on raw competition sequences. Each seed gets a fresh twelve-head control and a six-head candidate. All four use the existing native training, prediction and scoring programs from PR75. No model implementation or shared training pipeline changes are needed.

| Setting | Control | Candidate |
|---|---:|---:|
| Temporal attention heads | 12 | 6 |
| Coordinates per head | 64 | 128 |
| Transformer width / depth | 768 / 24 | 768 / 24 |
| Seeds | 207301, 207302 | 207301, 207302 |

Changing the head dimension also changes the SDPA scale and RoPE frequencies. The result measures their joint effect under the native implementation. The original Transformer study also examined head count at fixed model width in a translation setting. [Vaswani et al., sections 3.2 and 6.2](https://arxiv.org/html/1706.03762)

## Receiver handoff

1. Read [IDEAS.md](IDEAS.md) for the audit through team PR #96, the nearest comparisons and three follow-up ideas.
2. Inspect the four native [configurations](configs) and [frozen comparison](recipe.json). The control is the plain PR75 model. PR91's "linear" model has an additional input branch.
3. Follow the [B300 commands](B300_COMMANDS.md) with the published packet checkout and pinned native checkout. The receiver owns a fresh budget. First qualify the data, run the geometry and streaming probe, then profile each cell before admitting training.
4. After four completed endpoints, run the paired [readout](readout.py). Keep E3 journals, partial results, failures, GPU time and artifact hashes. The [MIDAS handoff](midas/README.md) carries evaluated endpoint evidence back to the team.

The decision is computed on the full, previously inspected official validation split. If both paired E4 deltas favor six heads and the mean gain is at least 0.0005 WP, fund a confirmation on a separately declared source-qualified split or a matched transfer study. Mixed signs or a smaller positive gain leave the effect unresolved. A consistently negative result deprioritizes this recipe. Two seeds estimate neither statistical significance nor public gain.

Four profiles and four fits are planned. No model runs, scores, or receiver acceptance are claimed at publication. The existing repository index and H/E ledgers point here. The execution contract is recorded in [PR.md](PR.md).
