# Learned mixtures and optimizer context

Primary sources checked October5,2026. These motivate controlled experiments;
their benchmark gains are not forecasts for our competition.

[ACON, CVPR2021](https://arxiv.org/abs/2009.04759) learns activation behavior,
including a switch between linear and nonlinear responses. The authors test
vision models. A causal recurrent adaptation must preserve streaming state and
avoid using future rows or batch statistics in a dynamic coefficient. E479
instead learns only two globally shared, bounded candidate-shape scalars, one
per recurrent layer. This is our model, not an ACON reproduction.

[Adaptive Blending Units](https://arxiv.org/abs/1806.10064),2018, is earlier
background for learning combinations of activation functions. Fixed mixtures
and separate starting shapes are necessary controls: otherwise a learned
mixture can win simply because it starts with a better fixed response.

[Learning Activation Functions for Sparse Neural Networks, AutoML2023](https://proceedings.mlr.press/v224/loni23a.html)
jointly studies activation choice and hyperparameters in pruned networks. Its
image-classification evidence suggests revisiting defaults after compression;
it does not establish that our dense GRU, recurrent state or integer deployment
will improve. A later quantized/pruned student should be tested against its own
equally tuned controls, rather than inherit a dense-model activation ranking.

[Descending through a Crowded Valley, ICML2021](https://proceedings.mlr.press/v139/schmidt21a.html)
finds optimizer rankings depend on workload in a large benchmark. Our small
constant-LR AdamW/Muon comparisons rank recipes under a particular budget;
they do not rank optimizer families under equal tuning effort. E476/E480 keep
all activation/optimizer/schedule combinations and match arithmetic LR sums
where possible. Parameter-update telemetry remains a separate measurement.

The user-specified [Mining Generalizable Activation Functions](https://arxiv.org/html/2602.05688v1)
contains both small-lab discoveries and downstream transfer failures. We use
its pointwise sine/sinc forms as hypotheses, retain approximation differences
and controls, and require transfer to larger recurrent models before promotion.
E481 tests a polynomial candidate with matched central slope and endpoints.
This is a curvature comparison; neither coefficient learning nor a positive
sine result would establish real periodicity in the anonymous market features.

Next mechanisms remain distinct: learned coefficients, per-feature causal
coefficient routing, parallel fixed activation channels, and activation-aware
compression. Only the registered fixed-head E477 and bounded-shape E479/E481
experiments have a current fit contract. Do not label these broader mechanisms
tested merely because a related activation failed.
