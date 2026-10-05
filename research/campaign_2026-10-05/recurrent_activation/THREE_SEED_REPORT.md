# E471/E475: all recurrent activation recipes across three seeds

All18 fixed fits completed with successful teardown and no OOM. The independently written grader validated seals/artifacts, exact raw selected targets/masks, finite predictions and official global/target scores.

| Seed | Recipe | Global WP | t0 | t1 |
|---|---|---:|---:|---:|
| 271828 | m10__adamw | 0.4887901 | 0.5010069 | 0.4765732 |
| 271828 | m10__muon_adamw | 0.4971020 | 0.4991056 | 0.4950985 |
| 271828 | zero__adamw | 0.4822645 | 0.4987989 | 0.4657301 |
| 271828 | zero__muon_adamw | 0.5054955 | 0.5055565 | 0.5054344 |
| 271828 | p10__adamw | 0.4793591 | 0.4945869 | 0.4641314 |
| 271828 | p10__muon_adamw | 0.5116375 | 0.5086565 | 0.5146186 |
| 271829 | m10__adamw | 0.4732534 | 0.5097297 | 0.4367771 |
| 271829 | m10__muon_adamw | 0.4406638 | 0.4987007 | 0.3826269 |
| 271829 | zero__adamw | 0.4836598 | 0.5120437 | 0.4552760 |
| 271829 | zero__muon_adamw | 0.4592916 | 0.5050181 | 0.4135651 |
| 271829 | p10__adamw | 0.4893710 | 0.5131418 | 0.4656003 |
| 271829 | p10__muon_adamw | 0.4745704 | 0.5103992 | 0.4387417 |
| 271830 | m10__adamw | 0.5481459 | 0.5205567 | 0.5757352 |
| 271830 | m10__muon_adamw | 0.5165775 | 0.5023150 | 0.5308400 |
| 271830 | zero__adamw | 0.5489639 | 0.5171897 | 0.5807380 |
| 271830 | zero__muon_adamw | 0.5309727 | 0.5068929 | 0.5550524 |
| 271830 | p10__adamw | 0.5480696 | 0.5125683 | 0.5835709 |
| 271830 | p10__muon_adamw | 0.5425230 | 0.5102879 | 0.5747581 |

| Fixed contrast | WP delta | Conditional95% component interval |
|---|---:|---|
| seed271828:m10__adamw_minus_zero__adamw | +0.0065256 | [+0.0003891, +0.0109914] |
| seed271828:m10__muon_adamw_minus_zero__muon_adamw | -0.0083934 | [-0.0128293, -0.0047516] |
| seed271828:p10__adamw_minus_zero__adamw | -0.0029053 | [-0.0064516, +0.0019704] |
| seed271828:p10__muon_adamw_minus_zero__muon_adamw | +0.0061420 | [+0.0028803, +0.0101905] |
| seed271828:m10__muon_adamw_minus_m10__adamw | +0.0083120 | [-0.0108818, +0.0262744] |
| seed271828:zero__muon_adamw_minus_zero__adamw | +0.0232310 | [+0.0041524, +0.0431232] |
| seed271828:p10__muon_adamw_minus_p10__adamw | +0.0322784 | [+0.0116651, +0.0535106] |
| seed271829:m10__adamw_minus_zero__adamw | -0.0104064 | [-0.0141002, -0.0066349] |
| seed271829:m10__muon_adamw_minus_zero__muon_adamw | -0.0186278 | [-0.0240548, -0.0126230] |
| seed271829:p10__adamw_minus_zero__adamw | +0.0057112 | [+0.0026175, +0.0089594] |
| seed271829:p10__muon_adamw_minus_zero__muon_adamw | +0.0152788 | [+0.0098608, +0.0206993] |
| seed271829:m10__muon_adamw_minus_m10__adamw | -0.0325896 | [-0.0471121, -0.0189171] |
| seed271829:zero__muon_adamw_minus_zero__adamw | -0.0243682 | [-0.0366379, -0.0126966] |
| seed271829:p10__muon_adamw_minus_p10__adamw | -0.0148006 | [-0.0242133, -0.0061145] |
| seed271830:m10__adamw_minus_zero__adamw | -0.0008180 | [-0.0037315, +0.0025782] |
| seed271830:m10__muon_adamw_minus_zero__muon_adamw | -0.0143952 | [-0.0214337, -0.0082133] |
| seed271830:p10__adamw_minus_zero__adamw | -0.0008943 | [-0.0033358, +0.0012505] |
| seed271830:p10__muon_adamw_minus_zero__muon_adamw | +0.0115503 | [+0.0059987, +0.0182778] |
| seed271830:m10__muon_adamw_minus_m10__adamw | -0.0315684 | [-0.0496939, -0.0165007] |
| seed271830:zero__muon_adamw_minus_zero__adamw | -0.0179912 | [-0.0322645, -0.0050629] |
| seed271830:p10__muon_adamw_minus_p10__adamw | -0.0055466 | [-0.0167068, +0.0054278] |

Positive alpha improves Muon at allthree seeds: +0.0061420, +0.0152788 and+0.0115503, with positive conditional component intervals. Negative alpha hurts Muon at allthree. AdamW effects vary in sign. The original apparent negative-alpha/AdamW advantage does not reproduce consistently. Both signs, all optimizers and all21 comparisons are retained.

These are common-FP32 GRU32 fits on the same exposed16components. No pooled CI, multiplicity correction, fresh-generalization or public prediction follows. The next discriminating test is transfer to the two-layer incumbent topology at larger width, with an alpha-zero custom-backend control and fixed seeds. Historical default-precision backend diagnostics remain separate in the original E471 report.

See THREE_SEED_ACTIVATION_RESULT.json for exact identities and eda/all_recurrent_seeds.png for the complete matrix.
