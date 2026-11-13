# B03. Does a since32 GRU still benefit from distillation?

This packet implements [B03 of PR92](https://github.com/VladKochetov007/WunderFunder/pull/92) on the K16 GRU. A since32 student gives less weight to training rows deep in a quiet stretch. A distilled student sees a smoother target. Their improvements could overlap. The decision is whether paying for the teacher yields a useful gain once since32 is present.

## Matched controls

| Arm | Since time constant | Teacher fraction | Student's training target |
|---|---:|---:|---|
| labels | off | 0 | clipped labels |
| since32 | 32 rows | 0 | clipped labels |
| kd | off | 0.5 | midpoint of clipped labels and clipped teacher predictions |
| both | 32 rows | 0.5 | same midpoint |

All four arms use cold GRU256×2 students, the same 16-row causal input stem and the same fitting-only teacher cache. The labels-only arms give its predictions zero weight. Teacher and age are used only for training. The score is computed from the raw two-output student. The two seeds are **203301 and 203302**. Two combination fits plus six control fits yield eight new fits. Each fit occupies one L4. No teacher fit, new teacher export or optimizer search is scheduled.

The standalone KD result in PR95 used label loss plus 0.2 times teacher loss on different seeds. It cannot fill a B03 cell. The published since32 endpoints likewise lack B03's common cache and source. PR95 reported +0.002904220 WP for since32 over zero and +0.001845795 for standalone KD over its own control. These numbers motivate the experiment. Their sum cannot predict the combined result. [Team results](https://github.com/VladKochetov007/WunderFunder/pull/95#issuecomment-6850953592) · [PR82 card 02](https://github.com/VladKochetov007/WunderFunder/blob/c0dc2e7b85847d45e088e6090cd1bffad5fdf549/research/experiment_drafts/l4-impact10-vlad-20261111/CARDS.md)

## Training contract

For each selected row and target let `y` and `t` be the label and detached teacher prediction clipped to ±2, and `z` the raw student prediction. Define `m = (1−α)y + αt`. Use `r = 1` on the unweighted arms and `r = max(0.25, 1/(1 + age/32))` on the since32 arms. The native causal i0-price event clock is computed from the full input sequence ahead of chunking, reset at each sequence, and never consumes labels.

The backward objective is **Σ(mask · |y| · r · (z−m)²) / Σ(mask · |y|)** over the actual optimizer window and both targets. The age multiplier applies to the entire supervised target. The label-magnitude denominator is the unaged denominator from the matched since32 recipe. Thus α stays exactly 0.5 throughout quiet stretches. Monitoring and public scoring never receive these row weights. [PR49-style target construction](https://github.com/VladKochetov007/WunderFunder/blob/33542502a1404a2843359b3999fa34f3a157bf3f/src/wnn/training/losses.py#L73) · [Native since weighting](https://github.com/VladKochetov007/WunderFunder/blob/33542502a1404a2843359b3999fa34f3a157bf3f/src/wnn/training/sequence_objectives.py)

The launcher prepares a private copy of the pinned PR96 native runtime in the receiver output directory and applies the packet's objective there. All arms run against those same derived source bytes. Config, data, optimizer, chronological loop, student forward pass and evaluator use the native interfaces. Strict FP32, AdamW at 0.0003, weight decay 0.01, gradient clipping 1.0, batch 32, chunk and gradient horizon 1,000, one chunk per update, 5% warmup followed by cosine decay to a 0.1 LR ratio. The endpoint is **E6 / 38,760 updates**. Earlier epoch scores are diagnostics.

## Data and teacher

Train and normalize on 10,325 official training sequences, excluding 282 linked training sequences. Evaluate all 1,873 official validation sequences and 3,528,273 scored rows with the pooled `global_wp_clip2_v1` metric. This validation set has repeatedly been used for development. The two seeds share this single split. A seed is not a new holdout.

| Required artifact | SHA-256 |
|---|---|
| PR49 chronological split | `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c` |
| Fitting-only cache manifest | `138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d` |
| Cache producer checkpoint | `759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90` |
| Official TL3C FBC1k s5 E3 checkpoint | `840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496` |

The team established logical equality of the two checkpoint payloads. Cache admission uses the original manifest, native file and fitting-membership validation, and its actual producer hash. No access to held-out teacher labels or predictions is needed during fitting. [Producer alias receipt](https://github.com/VladKochetov007/WunderFunder/blob/9540f7da9b4d4abf0326b8dc273974ffc5037f6e/research/experiment_drafts/final-quality-execution-20261112/evidence/PR49_CHECKPOINT_ALIAS_R1.json) · [Original cache admission](https://github.com/VladKochetov007/WunderFunder/blob/7926582ac6042f3ca01eedeebf4433ce6d06b3e3/research/experiment_drafts/pr49-standalone-kd-20261112/evidence/180-newkd-canonical-cache-profiles-r1.json)

## Decision after all eight endpoints

For each seed report the full E6 score, each target, **both − since32**, **both − KD**, the single-factor gains and the interaction **both − since32 − KD + labels**. A negative interaction means the gains overlap on this WP scale. The combination may still beat either singleton. Follow up on the combined recipe if both conditional gains are positive at each seed and each mean conditional gain reaches **+0.0005 WP**. A smaller gain over since32 alone weakens the case for training the teacher. Keep mixed or negative outcomes and the time spent. Two matched seeds and a repeatedly inspected valid cannot establish significance or a leaderboard gain.

The [GPU instructions](GPU_COMMANDS.md) cover qualification, the numerical probe, per-cell profiles, fixed training, independent prediction/score and panel readout. The packet has no measured runtime, WP, public score or server acceptance at publication.
