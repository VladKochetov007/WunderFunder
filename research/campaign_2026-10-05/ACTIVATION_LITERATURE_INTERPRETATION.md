# Activation literature and the current empirical questions

Primary sources rechecked October5:

- [Mining Generalizable Activation Functions (2026)](https://arxiv.org/abs/2602.05688) studies evolutionary activation search and selecting inductive biases using out-of-distribution performance. Its empirical setting includes small synthetic datasets; it does not establish financial-sequence gains or a preferred optimizer for this campaign.
- [Activate or Not: Learning Customized Activation (CVPR2021)](https://arxiv.org/abs/2009.04759) introduces learned activation switching and meta-ACON. Its image-model results motivate testing conditional activations. Our GELU/LeakyReLU recurrent-context mixture is an inspired comparison, not an ACON reproduction.

The useful questions here are conditional. E473's linear/slope controls and E481's matched cubic perturbation challenge a uniquely periodic explanation for the sine result. E474 tests where a nonlinearity is applied; changing an input embedding and changing a recurrent candidate are different interventions. E483 tests activation with Muon momentum rather than assuming a universal optimizer pairing. E486 crosses training and inference shapes on frozen weights: all nine training-by-inference interaction intervals are positive, but only one of nine inference-only swap intervals is positive; this argues for testing learned adaptation as well as output-function changes. It does not isolate curvature from amplitude or eliminate local search bias.

E488 now asks whether learned static/context gates have different effects after three epochs, retaining a fixed-mixture control and all epochs/targets. E487 demonstrates one-step sensitivity of native/custom update parity to orthogonalizer precision; E489 is separately designed to test duration and recurrent activation under the explicit FP32 variant. A numerical parity pass is not evidence of better optimization or prediction.

Every small-model evaluation uses historically exposed components. Next-stage transfer requires complete incumbent training and head procedure, independent fitting seeds and component evaluation, measured inference cost and compatible packaging. No small-model activation delta is added to0.6660 to forecast0.7000.
