# TL3_DIAGNOSTICS_R1 — exposed-F0 fixed-mixture diagnostics

This packet compares the exact cached TL3 endpoint, the fold-0 warm parent, the
selected R3 native q16x endpoint, and two mixtures fixed before target scoring.
It is a local diagnostic only. Fold 0 has been repeatedly exposed, the R3
alpha=-0.5 inference transform was adaptively selected on F0, and TL3's exact
training, teacher, and initializer membership is unknown; package metadata
identifies all-data parents. In particular, the TL3 point score must not be
read as clean validation or transfer evidence.

## Frozen sources and scoring

- `recipe.json` fixes the members, predictions, comparisons, two blend
  coefficients, and 10,000-resample component bootstrap seed before the score
  command. It binds the preflight receipt and grader source hashes.
- `preflight.json` binds fold 0 (split SHA
  `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`), 468
  evaluation groups, 141 components, the 337,186,198-byte target cache, the
  prediction arrays, their sidecars, and both ZIPs. It confirms all selected
  predictions are finite and checks sequence identity and row alignment. Its
  `label_values_accessed` field is false; label values are read only by the
  subsequent score run.
- TL3 archive `submissions/team_tl3_original_20261018.zip` is pinned to SHA
  `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`; the
  cache is pinned to SHA
  `3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3`.
  The [exact-ZIP replay receipt](../tl3_replay/full_f0_serving.json) covers all
  468 source-qualified groups (9.36M callback rows) and says labels and the
  scoring mask were not read during replay. It explicitly leaves exact model
  ancestry unknown and F0 exposed.
- R3 archive `submissions/geometry_r3_reverse_half_q16x_20261018_v2.zip` is
  pinned to SHA
  `27f01f53ba32f7256ce1663e8b163b382e673fc96e7870fdebdf310d4886f98e`; its
  prediction SHA is
  `e82e1a9844b323b9886927adab4d3d579a6f278654eef322488de7784d4823e6`.
  The [build receipt](../../geometry_variations_r3/geometry_r3_reverse_half_q16x_20261018_v2_build_receipt.json)
  identifies this as an inference-only derivative of the R2 checkpoint,
  adaptively selected on exposed F0. Its [full-F0 served receipt](../../geometry_variations_r3/geometry_r3_reverse_half_q16x_20261018_v2_f0_serving.json)
  reports 9.36M callback rows and the official score.
- The parent cache is `warm_mechanism_f0_s3_c_r3@ep1`, pinned by its sidecar to
  checkpoint SHA `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`
  and prediction SHA
  `c45c8d260d011fcc0c5900a4d11ee10d8261799716775ee6b26dd2cb6dd84482`.
  This grader verifies the pinned cache identity; it does not independently
  reconstruct that run's full fitting membership.

All scores use unchanged `global_wp_clip2_v1`, the cached official selection
mask and position >=99, 808,863 selected rows, and both targets pooled as
specified by the official metric. The two fixed blends are 75% TL3 / 25% R3
and 50% TL3 / 50% R3. No coefficient search was performed.

## Results

| Fixed endpoint | F0 score |
|---|---:|
| TL3 | 0.7018181213 |
| 75% TL3 / 25% R3 | 0.6982023797 |
| 50% TL3 / 50% R3 | 0.6937396526 |
| R3 native q16x | 0.6814234038 |
| Warm parent | 0.6812265132 |

On these exposed cached predictions, neither preregistered blend reaches the
TL3 endpoint. The fixed component bootstrap gives TL3-minus-parent
`+0.02059161` with 95% interval `[+0.01525849, +0.02705575]`, while
R3-minus-parent is `+0.00019689` with interval `[-0.00002585, +0.00043871]`.
The intervals resample the 141 evaluation components while holding every
prediction fixed. They do not capture fitting-seed variability, unknown TL3
validation overlap, or public-transfer uncertainty, and are not a clean-CV
claim.

The label-magnitude-weighted correlations between TL3 and R3 corrections from
the parent are `-0.602` and `-0.632` for the two targets; their corresponding
residual-error correlations are `+0.984` and `+0.984`. This describes the
cached predictions only. It does not establish a causal complementarity
mechanism, and the two fixed blends' lower point scores than TL3 give no
exposed-F0 basis to prefer them over TL3.

The [adversarial skeptic review](../../../unified/reviews/TL3_DIAGNOSTICS_SKEPTIC_REVIEW_20261018_R1.md)
classifies the result as ITERATE for transfer claims and recommends a fixed
F2 cross-fold robustness grade, gated on recovering TL3 member ancestry. The
reviewer also authored the grader, so this is not an independent regrade.

## Reproduction and artifact hashes

Run `grade_fixed_mixtures.py --mode preflight` before score mode, then execute
the fixed score command recorded in `execution_receipt.json`. Both runs used
`scripts/limited`, the `wnn-campaign.slice`, a 450 MiB memory cap, 20% CPU, one
thread, no swap, and physical scratch at
`/home/vlad/development/wunder/research_tmp`. The observed in-flight cgroup
memory was 270,979,072 bytes; this is not reported as a peak.

- Grader SHA-256: `d7b39aec6166a2496272beeca5a16497f9d7abcaabfe1b3bf4e7b415e4b567dd`
- Recipe SHA-256: `30296eb5f7e7a8e608d3e181c1d6bfd43074b8dae8ff056f6967387b695d7f76`
- Preflight SHA-256: `6991d856b6528af95d75f13904cdd1b13fcbe5e15a4a8d10ef0b031439289722`
- Result SHA-256: `10a42eefcda3d6703d3dee2006ebfc7bd69915830d34b9cfcf0a160dad8ae9451`
- Execution receipt SHA-256: `8f1d7f43ba92d9989675a14df0272f015c52d7ab1190a811e7555bf671c28100`

Files: [grader](grade_fixed_mixtures.py), [frozen recipe](recipe.json),
[preflight](preflight.json), [score and conditional intervals](result.json),
and [bounded execution receipt](execution_receipt.json). No candidate selection,
public score estimate, or scientific promotion follows from this packet.
