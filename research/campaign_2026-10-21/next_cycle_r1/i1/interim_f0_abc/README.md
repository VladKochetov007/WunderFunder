# Interim F0 A/B/C grade

This packet independently regrades the first completed F0/seed1001 A/B/C cell
from cached complete-fold predictions. It reuses the frozen six-fit panel
grader for exact fit provenance, evaluation membership, full NPY coverage,
official scoring and paired-component bootstrap, and reuses the earlier
[A/B interim wrapper](../interim_f0_ab/grade_i1_f0_ab.py) for its bounded
mixture view, chunked bootstrap, target loading and memory estimate. The
earlier [A/B result](../interim_f0_ab/RESULT_R1.json) remains unchanged.

The wrapper reports A, B, C, fixed `0.9*B+0.1*A` and fixed `0.9*B+0.1*C`.
Its five frozen contrasts are A−B, C−B, C−A and each mixture−B. For every
contrast it reports official WP and both target-score deltas, a paired
source-component 2,000-draw interval, and point deltas in the four fixed
position bins. It does not compute bin intervals, so zero-contributor
components cannot silently disappear from bin resamples. It performs no
mixture search.

Before loading target labels, the wrapper checks the pinned F0 split, fitting
subset, target-cache bytes, grader/scorer/metric/evaluation source hashes,
all three run sidecars and provenance, exact 4,620-update/zero-skip fit
receipts, shared non-null initialization and normalization identities, common
trainer source identity, FP32 inference sidecars, and each prediction NPY
header, full payload byte count, row order and 468-group coverage. It estimates
decoded target NPZ storage, all three prediction mappings, selected target
and mask arrays, plus 128 MB chunk and 256 MB process reserves from metadata;
it refuses to read labels above 1.4 GB. The mixtures and bootstrap read
prediction mappings in eight-sequence chunks. The report is create-only and
rehashes the target cache and pinned source files before writing.

The fitted run score metadata currently records WP 0.643173 for A, 0.659074
for B and 0.655537 for C ([A](../../../../../artifacts/runs/i1_f0_s1001_a_r1/score.json),
[B](../../../../../artifacts/runs/i1_f0_s1001_b_r1/score.json),
[C](../../../../../artifacts/runs/i1_f0_s1001_c_r1/score.json)). The new wrapper
has not yet emitted a paired C grade or its uncertainty. C is the
history-retaining arm, so C−B is the matched history contrast; the A/B-only
result cannot answer that question. The [final-epoch C provenance](../../../../../artifacts/runs/i1_f0_s1001_c_r1/provenance.json)
reports a 0.9727 clip rate, 0.461 mean clip factor and 0.9135
squared-gradient share in the head. These are optimization diagnostics: they
show frequent clipping and a gradient concentration at the readout, but they
do not establish whether history is useful or redundant. F0 is an exposed
development fold with one seed; its component interval is conditional on
these fitted models, not training-seed, fold, or public-transfer uncertainty.
The full panel and F2 grade remain pending.

After integration into the canonical checkout, root can run the wrapper in
the admitted one-CPU/1.4-GB slot:

```sh
scripts/limited \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/grade_i1_f0_abc.py \
  --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/interim_f0_abc_grade_r1.json
```

This is preparation only: no prediction arrays, target labels, tests, fits,
provider calls or GPU work were run while building the packet.
