# E465 all-eight-TPU continuation draft

This directory is an isolated design and correctness-test draft. It is not a
launch bundle and must not be used for a scientific run before E463's TPU
real-data preflight completes and the objective tests run on a JAX-capable
host. `rnn.py` and `bridge.py` are copied byte-for-byte from the reviewed
`staging/jax_tpu` source snapshot; this draft adds only a distributed pooled
objective helper and tests.

The intended comparison is one fixed epoch (epoch 0) of FP32 and BF16 AdamW
continuations from the same converted `c20261003_combo_s3_f0@ep3` evaluation
EMA parameters. Both arms use the same new AdamW state, fixed seeded order of
all 373 full fit batches, global batch 32 (four sequences per TPU core), chunk
size 1000, and full 20,000-row state carry with a stop-gradient boundary
between chunks. The 11,940 fit row groups yield 373 full batches plus four
remainder groups; dropping that remainder matches the existing trainer's
`drop_last=True` behavior. The unchanged objective uses the equal mean of the two fixed fold-0
SBD teachers, alpha 0.5, stopped horizon-8 bootstrap, propensity-squared plus
0.03 weighting, 0.03 valid-sequence correction, 0.3 structural auxiliary,
0.3 next-change auxiliary, and consistency against the incoming EMA. EMA is
updated once after each optimizer step using the production row-clock decay.
No validation-based epoch selection is permitted.

`distributed_combo.py` accumulates four numerator/denominator pairs locally,
then `psum`s both across replicas before forming pooled ratios. The gradient
path uses a scaled local numerator contribution and a `pmean` of gradients for
replicated parameters; this avoids differentiating a replicated global scalar
with ambiguous collective transpose scaling. The tests compare pooled loss
and parameter gradients to the existing unsharded `combo_sbd_loss`, including
unequal shard mass, a zero-mass shard, and globally zero mass. They require
two devices and should run on CPU first with two forced host devices. Passing
these loss tests is necessary but not sufficient: the final TPU worker still
needs a recurrent state/carry parity test, both-precision Torch forward parity,
EMA update parity, and strict input/checkpoint/teacher hash checks.

No scientific fit, accelerator allocation, data conversion, or score claim has
been made from this draft.
