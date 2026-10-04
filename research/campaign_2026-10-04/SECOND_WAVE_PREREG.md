# Second-wave contrasts and fixed decisions

These contrasts continue the accepted inventory. Historical negatives remain
context and controls. A scientific fit starts only after its implementation and
hardware/resource checks pass; an unsealed recipe is not a completed experiment.

## Final predictor and precision

E412 keeps epoch3 plus head epoch0 at equal weight. The q16x raw-error gate failed
and remains failed. Common FP32 inference of both saved checkpoints must precede
an engine-correctness decision. Preserve all existing FP16 caches under their
original names. Float and compressed packages must represent the same fixed
predictor; neither verification rows nor the historical holdout choose weights.

The next final seed is seed4 with the same full combo recipe and fixed epoch3,
followed by the same one-epoch head. Its equal seed bag is a separate candidate.
No distillation of that pair before both complete teachers exist. Deployment
depends on actual package accuracy and CPU timing, not summed local deltas.

## QAT and training precision: H069/H105

Use one fixed strong parent, `c20261003_combo_s3_f0@ep3`, with its inherited
preprocessing and exact fold0 training/exclusion lineage. Continue for one epoch,
fixed epoch0, seed3, AdamW at maximum learning rate0.0001, batch16, chunk1000.
The float and QAT continuations use identical data, teachers, objective, update
schedule, clipping and EMA policy. The checked implementation supports two fixed
presets: recurrent weights, and recurrent plus the first head matrix.

Required arms: float FP16 continuation, float FP32 continuation, recurrent QAT
FP32, recurrent+head QAT FP32; zero-update float parent and both post-training
quantized exports. Score every arm with common FP32 inference. QAT versus FP32
continuation isolates this quantized model/training intervention; FP16 versus
FP32 continuation isolates training precision. Additional changes cannot enter
only one arm. A bounded production-shape resource check precedes all fits; any
batch-size amendment applies to every continuation and gets a new specification.

## Factorized recurrence: H070

Use the existing W384 parent, rank128 SVD initialization and a zero-update SVD
export. Compare with dense W384 continued from the identical parent and a dense
GRU with matched learned parameter count. H281 is the match for211 inputs/two
outputs; recompute the count for actual auxiliary output dimensions before sealing.
Use AdamW/FP32 and identical continuation settings for the same-parent contrast.
The independently initialized budget-matched GRU is an architecture comparison,
not an initialization-matched ablation. Keep those interpretations separate.

Training currently materializes effective dense matrices. Saved factors establish
the learned model family, not a CPU speed gain. A factorized native backend and
full-sequence parity are prerequisites for a deployment claim.

## Width, clocks and loss

Replicate the original W384 AdamW/chunk2000/bootstrap0 recipe on fold1 against
its matching W256 control. Separately test width256/384 × AdamW/Muon with
chunk1000/bootstrap8 held fixed. Do not call the original recipe versus the combo
an isolated Muon interaction. Hardware stacks must match within each comparison.

H071 keeps EMA retention fixed in rows while changing update cadence. H072
separates update2000/detach1000 from update2000/detach2000 with pooled loss
numerators and denominators; controls share the same admitted batch size. H073
first computes exact mixed-training-pool denominator calibration (E414), then
compares unchanged, aligned and shuffled multipliers. Calibration alone has no
predictive endpoint.

## Literal conditional features and state probes

The separately versioned H082–H089 code specifies actual routing, clocks,
innovations and smoothing controls. E415's proxy losses do not close these
families. Every literal screen includes the unchanged backbone and global affine
calibration; a win only against a damaged capacity control is insufficient.

H099 uses a fixed component-hashed panel, no reset, four reset lengths and
same-boundary state attenuation. Input features retain their causal history.
This tests state-value disruption, not gradient-credit usefulness. H095 compares
state and input-summary probes on disjoint episodes and components. Any earlier
target-label oracle is explicitly nondeployable; a maturity buffer is an
assumption to report, not an independently proved property.

All families retain every fitted arm, attempted job, numerical failure and
resource record. Confirmation uses additional prespecified folds/seeds; public
transfer remains a separate observation.
