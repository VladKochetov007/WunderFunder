# L4 GRU impact-10: ten heterogeneous screens on the PR79/PR80 contract

**Status: proposal, no fits executed.** Preparation owner: Vlad / Claude Code. Receiving and
launch owner: Semyon (as requested on 2026-11-10: "take the experiment contract from my PRs and
propose experiments that can bring impact; do not optimize the experiments, I will set them up").
No execution engineering, throughput tuning or queue changes are part of this packet.

## Contract inherited from PR79/PR80

Every card keeps the PR80 frozen recipe unless the card names the single field it changes:
GRU H256×2, raw112, causal input stem K16, batch32, chunk1000, accumulation1, strict FP32,
AdamW 3e-4, wd 0.01, clip 1, warmup 5%, `warmup_cosine_v2` floor 0.1, six epochs, fixed E6
endpoint, all-epoch retention, `standalone_labels_wmse_v2` unless stated. Fitting and
normalization use the 10,325 official train sequences (split SHA
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`); full-valid monitoring
scores all 1,873 sequences / 3,528,273 rows every epoch; the primary score is canonical
`global_wp_clip2_v1` at fixed E6.

**Controls are reused:** PR80 `k16_s203301` (0.683019) and `k16_s203302` (0.682629). Each card
adds **two treatment fits** at seeds 203301/203302 on L4: **20 new fits total**, plus two
zero-fit export jobs (cards 02 and 10 need train-set predictions of existing checkpoints).
Decision heuristic as in PR80: mean paired fixed-E6 gain ≥ 0.0005 WP with both seed deltas
positive warrants follow-up; preserve mixed and negative outcomes. Validation is exposed
development data; two seeds are not two folds.

## Why these ten

Our train-only production parent recipe (P, valid 0.68671) beats the PR80 K16 labels-only
control (0.6830) by ≈ +0.0037, and the warm distilled child D1 reaches 0.68997. The parent
differs from K16 in four ingredients that the PR runtime does not have: Muon on recurrent
matrices, auxiliary structure heads, since-event row weighting and per-update EMA. Cards 01,
03, 04 and 06 isolate those ingredients in Semyon's runtime, where they have never been
measured against this control. Card 02 is the largest measured lever (distillation). Cards
05, 07, 08, 09 and 10 are new mechanisms aimed at measured weaknesses: late-sequence memory,
the regime shift between train and valid (adversarial AUC 0.984), and class 3, which carries
48% of the metric weight at WP ≈ 0.44.

Already covered elsewhere and deliberately **not** repeated: K16/K64 stems, width 512,
dropout 0.1 (PR79/80), target specialists (PR77), input-dynamics SSL (PR78), dropout/SWA
(PR71), correlation objective (PR70), Ivan's ten B300 cards (PR81: PLE embeddings, group
tokens, target-private adapters, deep supervision, tied depth, ranking, future-label heads,
state perturbation, NLL/variance head, clip-event heads), loss-family changes, class-3 up-
weighting, i1 lead block, hand microstructure features, TabM/NorMuon/view masks (our
negatives, October 25–29).

| # | Card | Family | Changes | Code needed | Prior evidence |
|---|---|---|---|---|---|
| 01 | [Muon on GRU matrices](CARDS.md#01-muon-on-gru-recurrent-matrices) | optimizer | `optimizer` | port GRU Muon | +0.0038/+0.0043 cold, 3 replications |
| 02 | [Distill a train-only TL3C teacher](CARDS.md#02-distillation-from-a-train-only-tl3c-teacher) | objective | `kd_weight`, teacher cache | teacher train export | D1 +0.0033 over P |
| 03 | [Since-event row weighting](CARDS.md#03-since-event-row-weighting) | loss weighting | row weights | small | E-351 +0.0015/+0.0010 |
| 04 | [Auxiliary structure heads](CARDS.md#04-auxiliary-structure-and-next-change-heads) | multi-task | aux heads | port heads | part of P, unisolated |
| 05 | [Longer gradient horizon](CARDS.md#05-longer-gradient-horizon-at-matched-rows-per-update) | memory | `chunk_rows`, `batch_size`, horizon | none (config only) | late-step gap |
| 06 | [Per-update weight EMA](CARDS.md#06-per-update-weight-ema-endpoint) | averaging | EMA shadow | small | part of P; SWA null (PR71) |
| 07 | [Causal regime channels](CARDS.md#07-causal-per-sequence-regime-channels) | shift robustness | extra inputs | small | adversarial AUC 0.984 |
| 08 | [Per-sequence scale augmentation](CARDS.md#08-per-sequence-input-scale-augmentation) | shift robustness | train-time aug | small | shift drivers = scale |
| 09 | [Class-routed readout](CARDS.md#09-causal-class-routed-readout) | architecture | head | small | class 3 = 48% weight |
| 10 | [Seed-ensemble self-distillation](CARDS.md#10-seed-ensemble-self-distillation-born-again) | objective | `kd_weight`, self teacher | control train export | seed mean +0.0011 |

Config-only card 05 has frozen JSON in [configs/](configs/). The others need the named code
change in the receiving runtime first; each card lists the exact field change and the falsifier.

## Precision note (no fit)

Our October 25 audit found that torch 2.14 cuDNN GRU fits ran TF32 recurrences
(`cudnn.rnn.fp32_precision='tf32'`) even in recipes labelled strict FP32, until
`--strict-fp32-kernels` set the RNN precision explicitly. The PR runtime's `sequence_precision.py`
sets `cudnn.allow_tf32=False` but not the per-op RNN precision; on CPU, torch 2.14 then reports
`cudnn.rnn.fp32_precision='none'`, whose GPU resolution we have not verified here. Please log
`torch.backends.cudnn.rnn.fp32_precision` (and the effective GRU math mode) on the L4 runtime
before admitting new fits. On our side the fix cost no measurable quality (−0.0011/−0.0001), so
this is a fidelity item, not an expected gain.
