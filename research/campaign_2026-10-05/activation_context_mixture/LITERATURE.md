# Causal activation routing: related work and limits

Primary sources checked October5,2026. This is background for E484, not
evidence of a score improvement or an exact paper reproduction.

[Activate or Not: Learning Customized Activation](https://arxiv.org/abs/2009.04759)
introduces learned activation behavior and meta-ACON's parameter switching.
Its reported experiments concern vision tasks. This motivates testing
input-dependent activation choice separately from globally learned
coefficients, without forecasting transfer to these market labels.

[Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
uses a simplified gating mechanism for language modeling. Gating is an
established representation choice; adding it to a causal recurrent readout
does not establish architectural novelty.

[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
compares nonlinearities in multiplicative feed-forward gates, including
alternatives to sigmoid, in a sequence-to-sequence Transformer. Those
results do not require us to replace a recurrent trunk with attention,
but they also do not demonstrate that our proposed readout will improve.

Our design uses a sigmoid gate to interpolate GELU and LeakyReLU on the
same head preactivation. The gate's context is an existing recurrent output,
input projection, or head preactivation, normalized independently per row.
This is our adaptation of a gated activation, not ACON or SwiGLU as defined
in those papers. It changes the use of available history, adds64 parameters,
and changes gradient paths. Equal parameter counts and initially identical
functions do not equalize optimization geometry or functional capacity.

No batch statistics, future windows, or extra information enter the causal
predictor. A recurrent-context advantage over a static gate alone would not
identify recurrence as the cause; the two alternative context sources are
registered controls. New initializations on an exposed cohort are not new
market validation. All12 fits and16 contrasts remain exploratory until
the source, numerical, resource and independent scoring gates pass.
