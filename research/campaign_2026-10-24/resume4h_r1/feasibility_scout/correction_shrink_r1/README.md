# Fixed correction shrink diagnostic R3

This folder contains a completed, independently reviewed, no-fit diagnostic for the
saved F0 online relationship endpoints. The scientific question is whether one
fixed additional 0.1 multiplier on each learned correction improves its score
against the frozen parent and its unshrunk endpoint. The online model already
starts from the parent and adds a zero-initialized residual at scale 0.1; this
proposal applies another fixed scale to the exported correction. It does not
test a missing parent-preservation path.

The frozen recipe scores five unique predictions: A/B/C at lag 1 and B/C at
lag 16. A lag-16 is omitted only because the label-free preflight recorded the
same full prediction SHA-256 as A lag-1, and the runner rechecks that equality.
The five primary contrasts are `shrink_<cell> − parent`. Thirteen secondary
contrasts compare each shrink with its raw endpoint and report C−B and B−A
within each lag for both raw and shrunken outputs. Existing raw-versus-parent
scores remain available in the original grade and are rechecked across all
five time bands before this adapter writes a result.

All evidence is F0, fold 0, fitting seed 1001, with 468 evaluation groups and
141 source-qualified components. It is exposed development evidence, and the
component bootstrap is conditional on these fixed predictions. The coefficient
is fixed at 0.1, no search or new fitting is permitted, and uncertainty is not
a kill gate. The historical 12-fit / 14-endpoint panel remains incomplete and
unchanged.

## Frozen files and inputs

- [R3 recipe](CORRECTION_SHRINK_RECIPE_R3.json), SHA-256
  `b2bf2a065687e3d970db852fa55cacd0d202da43eb587527b99683a3881e6802`.
- The [independent R3 review](REVIEW_R3.md) passes for root admission and
  records the exact checks and remaining resource caveat.
- The separate [post-execution result review](RESULT_REVIEW_R3.md) confirms
  exact raw-score reproduction against the original F0 grade and reviews the
  5 primary/13 secondary contrasts. It limits the interpretation to an
  exposed fixed-prediction diagnostic: all shrink-versus-parent intervals
  cross zero, and score proximity does not establish mean-only/relationship
  equivalence or an optimal scale.
- [R3 source manifest](CORRECTION_SHRINK_SOURCE_MANIFEST_R3.json), SHA-256
  `62d96dbe1881ef2e42939fecdb8179c3dfdd6caa79122b961e31b4acf28711bb`;
  it pins 62 direct and transitive implementation/configuration files and the
  synthetic validation receipt. R3 checks the exact six candidate-map keys
  before scoring and tests rejection of a malformed map.
- The earlier unadmitted R2 packet remains byte-for-byte preserved:
  [recipe](CORRECTION_SHRINK_RECIPE_R2.json),
  [source manifest](CORRECTION_SHRINK_SOURCE_MANIFEST_R2.json), and
  [synthetic receipt](CORRECTION_SHRINK_SYNTHETIC_R2.json). Its three source
  files are in the [R2 snapshot](r2_source_snapshot/README.md). R2 was
  superseded after review found its candidate-map gate checked only entry count.
- [Runner](grade_correction_shrink_r1.py) and
  [scorer](correction_shrink_scoring.py) validate the root admission, exact
  input identities, source-qualified order, candidate sidecars/provenance and
  parent ancestry before reading evaluation targets. The target archive is
  hashed before its first label read and after endpoint scoring; labels are
  streamed to physical scratch-backed maps.
- The [standalone synthetic checker](check_correction_shrink_synthetic.py)
  exercises the float32 formula, exact 0.1 recipe contract, metric clipping
  order, five primary / thirteen secondary contrasts, all-band official-score
  parity, and synthetic preflight/admission gates. It ran with generated
  arrays only under a 150 MB, one-core, 60-second cap. Its
  [R3 receipt](CORRECTION_SHRINK_SYNTHETIC_R3.json), SHA-256
  `55709fea6a8f16fb861ea76dd695a515079571753bf6d325c3370a0c77f47d85`, records
  all checks passing and no campaign prediction or target payload reads.
- The frozen F0 R2 recipe and source manifest are
  [here](../F0_EXPLORATORY_RECIPE_R2.json) and
  [here](../SOURCE_MANIFEST_R2.json). The reused baseline grade is
  [here](../F0_EXPLORATORY_GRADE_R2.json), SHA-256
  `bfc06c9820f48b5214bd32d13783a268a3e35da8ffc8c61f65b91912352e2899`.
- The label-free preflight is stored outside the repository at
  `/home/vlad/development/wunder/research_tmp/resume4h-r1/f0-online-preflight-r3.json`,
  SHA-256 `f8fd01dab2226fdc46660e72a1cc25d1da3457429f404d0d0f036e18f01d7692`.

The rationale is narrower than prior output calibration work. E-288 in the
[October 1 scale probe](../../../../../research/campaign_2026-10-01/class_scale_probe.py)
evaluated per-class multipliers on a different four-fold reference ensemble;
that result does not answer whether one fixed, uniform shrink helps these five
saved parent-residual endpoints. The [October 22 I1 history](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md)
also weakens a pure scalar-shrinkage explanation for a different blend, but is
not a direct evaluation of this endpoint set. This packet does not claim a new
general mechanism.

## Admission and execution

The [R3 result note](RESULTS_R3.md), [complete grade](CORRECTION_SHRINK_GRADE_R3.json),
and [execution receipt](EXECUTION_R3.json) record successful execution after root
admission. All five fixed shrunken corrections gain approximately +0.000129 to
+0.000136 over the parent in this exposed F0 cell; mean-only A gains almost as
much as the relationship variants. All conditional intervals span zero.
Execution took 48.8313 seconds, zero new fits, zero OOM and zero swap.

The [root admission record](CORRECTION_SHRINK_ADMISSION_R3.json) was created
by root and binds the recipe/source manifest hashes, original recipe and grade,
preflight receipt, metric, alpha, zero-fit scope, target-read authorization,
output path, and resource envelope. Until that record exists and validates, the
runner does not hash candidate arrays or read labels. The admitted R3 execution
completed on October24; the command below records the reproducible invocation.

The synthetic check can be rerun without reading campaign predictions or
targets:

```sh
WNN_MEM_MAX=150M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=60 \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/check_correction_shrink_synthetic.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_RECIPE_R3.json
```

After root admission and independent review, the bounded target-grade command is:

```sh
WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=300 \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/grade_correction_shrink_r1.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_RECIPE_R3.json \
  --recipe-sha256 b2bf2a065687e3d970db852fa55cacd0d202da43eb587527b99683a3881e6802 \
  --admission research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_ADMISSION_R3.json \
  --preflight-receipt /home/vlad/development/wunder/research_tmp/resume4h-r1/f0-online-preflight-r3.json \
  --baseline-grade research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_GRADE_R2.json \
  --output research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_GRADE_R3.json
```

The executed worker cap was 400,000,000 bytes, one CPU core, five minutes,
zero swap, and physical-disk scratch at `/home/vlad/development/wunder/research_tmp`.
The prior F0 grade took 32.71 seconds and reached 399,998,976 bytes, so this
cap required active monitoring for the expanded endpoint/contrast grade.
The R3 peak was 399,998,976 bytes with 6,836 memory-limit events; successful
completion had little memory headroom.
