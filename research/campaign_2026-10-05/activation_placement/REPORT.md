# E474: localized activation changes on a two-layer GRU

All12 fixed fits and independent CPU scoring completed with no new OOM. The score is official global_wp_clip2_v1 on16 previously exposed diagnostic components (26,990 selected rows); this is neither the strong submission baseline nor a public expectation.

Sine in the embedding changes global WP by −0.004506 and +0.002476 across the two seeds. Sine in the head changes it by −0.000570 and −0.000072. Thus the earlier GRU32 sine advantage does not transfer uniformly to this topology/placement/optimizer. Sine versus matched linear slope in the embedding remains positive at both seeds, but that relative comparison does not imply improvement over ordinary GELU.

| Seed | Arm | Global WP | t0 | t1 |
|---|---|---:|---:|---:|
| 271828 | base | 0.4441839 | 0.5122195 | 0.3761483 |
| 271828 | sine_embed | 0.4396783 | 0.5097308 | 0.3696257 |
| 271828 | slope_embed | 0.4383880 | 0.5095991 | 0.3671769 |
| 271828 | sine_head | 0.4436142 | 0.5098945 | 0.3773338 |
| 271828 | slope_head | 0.4441013 | 0.5091221 | 0.3790805 |
| 271828 | sine_both | 0.4375042 | 0.5063030 | 0.3687054 |
| 271829 | base | 0.4940204 | 0.4912452 | 0.4967955 |
| 271829 | sine_embed | 0.4964964 | 0.4915323 | 0.5014605 |
| 271829 | slope_embed | 0.4955801 | 0.4902680 | 0.5008922 |
| 271829 | sine_head | 0.4939483 | 0.4937315 | 0.4941650 |
| 271829 | slope_head | 0.4929584 | 0.4931783 | 0.4927386 |
| 271829 | sine_both | 0.4957077 | 0.4929955 | 0.4984200 |

| Seed | Registered contrast | Global WP delta | Conditional95% interval |
|---|---|---:|---|
| 271828 | sine_embed_minus_base | -0.0045056 | [-0.0069374, -0.0017889] |
| 271828 | slope_embed_minus_base | -0.0057959 | [-0.0083255, -0.0022060] |
| 271828 | sine_embed_minus_slope_embed | +0.0012903 | [+0.0000612, +0.0022749] |
| 271828 | sine_head_minus_base | -0.0005697 | [-0.0033380, +0.0021269] |
| 271828 | slope_head_minus_base | -0.0000826 | [-0.0033572, +0.0031937] |
| 271828 | sine_head_minus_slope_head | -0.0004871 | [-0.0014603, +0.0004549] |
| 271828 | sine_both_minus_sine_embed_minus_sine_head_plus_base | -0.0016043 | [-0.0024352, -0.0003998] |
| 271829 | sine_embed_minus_base | +0.0024761 | [-0.0000560, +0.0049959] |
| 271829 | slope_embed_minus_base | +0.0015598 | [-0.0013598, +0.0046563] |
| 271829 | sine_embed_minus_slope_embed | +0.0009163 | [-0.0003229, +0.0024200] |
| 271829 | sine_head_minus_base | -0.0000721 | [-0.0027409, +0.0049147] |
| 271829 | slope_head_minus_base | -0.0010619 | [-0.0038923, +0.0042494] |
| 271829 | sine_head_minus_slope_head | +0.0009898 | [+0.0004906, +0.0012786] |
| 271829 | sine_both_minus_sine_embed_minus_sine_head_plus_base | -0.0007166 | [-0.0016226, -0.0000507] |

All14 contrasts are retained, including the joint placement interaction. Intervals resample the same components across all arms and condition on fitted models; they do not include seed/search uncertainty or correct multiplicity. No winner was used to alter the registered fits.

The independent grader reconstructs each model and recomputes fit-only normalization from sealed training rows. It checks runner receipts, raw selected targets/masks and finite predictions before official clipping. See scores/twelve_cpu_v1/score_result.json for exact identities and eda/all_placement_arms.png for all arms. Admission-only errors and the earlier probe artifact error are preserved separately; no failed fit was counted as completed.
