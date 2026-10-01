# Shallow student: first actual readout

1 October 2026, 13:05 UTC. Four fits and two fixed-epoch held-fold predictions completed in 2,959.9 seconds on beleriand. No public submission or GitHub operation.

| Equal-weight ensemble | Pooled WP | Student minus reference |
|---|---:|---:|
| GRU256×1 student + three H128 anchors | 0.702379667614 | — |
| Original distilled GRU256×2 + same anchors | 0.702367225514 | student +0.000012442100 |
| Three H128 anchors | 0.701742429604 | student +0.000637238010 |

Both historical reference scores reproduce exactly. The fixed comparisons cover 860 pure sequences in 250 chains across previously exposed folds 0/1. Student-minus-strong4 differences are +0.00006222 and +0.00011577 by fold. These do not average to the global difference: the official statistic pools weighted moments across folds.

The 2,000 paired whole-chain resamples give overall conditional 95% intervals of **[-0.00049625, +0.00046004]** against strong4 and **[+0.00024783, +0.00100798]** against fixed3. These pointwise intervals condition on the fitted models and do not cover seed variation, adaptive recipe selection or hidden-domain shift. The result does not establish superiority or formal noninferiority to strong4.

The useful finding is near-equal point quality with a cheaper recurrent model. Dense serving MACs fall from 1,741,952 to 1,348,736 per row for the four-member ensemble, a 22.6% reduction. This is arithmetic, not measured serving speed. Normalization and the recurrent trunk are byte-identical between each parent endpoint and its head-finetuned child; heads changed and all tensors in the selected endpoints are finite. Exact cache-producing epochs of the historical teachers remain unproven, as declared before the study.

## Next allocation selected before further execution

Root chose **one native runtime/correctness probe**, not additional seeds or head variants. It uses the fixed fold-0 student ensemble, the existing guarded VNNI INT8 kernel, five fixed validation sequences (100,000 callbacks), no target/mask decoding and no WP query. It checks one-layer support, reset/repeat and streaming agreement, then reports prefix cost. A projection cannot qualify the complete final submission. Source, limits and results live under `native-probe-01/`.

The positive quality/cost screen justifies preparing a final full-data parent and H2 recipe while the probe runs. Actual final training requires a separate root allocation after inspecting that probe. Historical full teachers and their caches are available in the base checkout; no teacher substitution or restoration download is needed.

Primary evidence: `execution-01/result.json`, `fit-inspection-f0.json`, `fit-inspection-f1.json`, `evaluation-01/report.json`, `evaluation-01/chain-bootstrap.json`, and `final-teacher-availability.json`.


## Follow-through, 13:18 UTC

The single native probe passed on the unchanged guarded VNNI kernel. Its 100k callbacks took 6.883 s; input reads and checks brought the trace to 7.173 s (47.10-minute projection at 39.4M). CPU sibling activity was recorded without retry or correction. Based on this evidence, root allocated exactly two final full-data fits, launched at 13:18 UTC, and two native INT8 OOF passes with one fixed comparison against the existing strong4 INT8 outputs. The final ZIP still requires actual full-runtime qualification. No additional architecture, seed, checkpoint, or blend search was introduced.


## Native quality and final training, 13:45 UTC

The fixed native INT8 student ensemble scores **0.702355321333**, versus **0.702337076243** for the reproduced native strong4 reference, delta **+0.000018245090**. Both use the same unchanged guarded VNNI implementation. Conditional paired-chain95% interval: [-0.0004887654,+0.0004685579]; no superiority or formal noninferiority is established. The native-minus-GPU difference of roughly -0.00002435 combines backend, AMP and quantization effects. See `native-quality-01/evaluation/report.json`.

An explanatory calculation on those existing sufficient statistics attributes the pooled change to +0.0000945596 from within-fold covariance/variance changes and -0.0000763145 from between-fold mean terms. This explains why the pooled delta is smaller than each fold-average WP delta. It is algebra on the same two fixed profiles, not a new candidate or calibration proposal; see `native-quality-explanation/note.txt`.

Final full-data parent+H2 finished in 24.89minutes; all8 checkpoints are finite, and the H2 recurrent/input/normalization tensors equal the fixed parent endpoint exactly. All six completed CV/final runs are preserved locally and remotely:24 checkpoints plus histories/logs. The final fixed4 ZIP, SHA `eec571629b0829aeb9c73db80a1170fcb7cb3ccfd132f72f3ddd51f779d2fdd4`, passed bounded final-weight repeat/reset/stream/prefix/wrapper checks.

**Full runtime completed at14:29:27UTC:**39.4M callbacks and39,204,970 emitted predictions in **2672.212005s =44min32.21s**, including process startup, imports/weights, raw I/O, before/after input hashes and validation. All97 full-sequence repeats have identical emitted hashes, all1970 sequence IDs are fresh and all four models remain active. Peak observed RSS348.246MiB; source, checkpoint, ZIP and input checks pass; scope and all three runtime PIDs are released. Headroom to the3600-second limit is927.788s. The corresponding strong4 actual process took3469.439s; these are separate shared-host runs, not an isolated causal benchmark. Sibling activity for the student run was recorded at~9% and not subtracted. See [outcome](../student-runtime/outcome.md). Qualification covers the measured natural v3/VNNI host, not unspecified hidden-grader hardware.

A separate prospective seed4 study completed four same-recipe CV fits and two fixed H2 predictions, with four fixed contrasts after the duplicate-seed3 amendment recorded before any new score readout. Its double-shallow profile scores0.7023535406, versus current student4's0.7023796676; delta−0.00002613[−0.0006354,+0.0004809] with23.9% more MACs. Single seed4 scores0.7022343699, duplicate seed3 scores0.7022761303. Double versus duplicate is+0.00007741[−0.0001540,+0.0003188], an uncertain contribution that does not make double better than the current recipe. No final seed4 fit/native follow-through is allocated. Current seed3 student4 remains primary. See [complete seed4 outcome](../student-seed4/outcome.json) for all folds, contrasts and the preserved foreign-GPU interruption plus one identical-recipe infrastructure recovery. This later study does not change the completed seed3 readout.
