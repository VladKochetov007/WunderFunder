# Independent review record

This records the separate audit agent's source-review findings and the coordinator's
corrections. No model, numerical probe, test suite, training, inference or profile
was executed locally. These are source-level observations, not B300 acceptance.

## Findings and corrections

1. The copied readout contained `6_325` instead of the original `10_325` fitting
   sequences. The independent reviewer identified this deterministic rejection
   bug; it was corrected and the reviewer rechecked the value. Other cell-count
   substitutions and native data, epoch, update and output-schema contracts were
   inspected separately.
2. The initial eval RNG assertion surrounded the first compiled call. Compilation
   or autotuning could consume RNG independently of the perturbation. The reviewer
   requested a warm call; the coordinator added it before taking the RNG snapshot.
   The assertion now covers the warmed evaluation path only. Fresh-process
   stochastic resume is not established and the wrapper exposes no resume stage.
3. Terminology was narrowed to **pairwise rotation versus pairwise gain**. Independent
   pair gains can change the complete head's direction. Matching first moments,
   expected energy and displacement does not match logit covariance or entropy.

## Source paths inspected

The reviewer inspected the payload formulas, signs and shape validation; config
whitelist and constructor propagation; train-only post-RoPE/pre-SDPA hook; native
cache construction and unchanged parameter initialization; receiver CLI/profile
and grade schemas; six-cell MIDAS exporter and actual metadata-only preview.
Native reference: `8626000a4b965763a70011a0291a891aa6535d7e`.

The numerical probe uses explicit-sign means, energies and a rotation determinant
to avoid testing the production formula solely against itself. It also requires
active model-level treatment, same-seed initial tensors, neutral eval, direct
one-layer K/V equality, causal prefix behavior, nonzero finite gradients and warm
compiled RNG replay. All these checks are prepared for the receiver and unexecuted.

The separate reviewer reported no further deterministic API/schema blocker in the
inspected runtime and handoff paths. The coordinator ran only Ruff, Python3.11
AST parsing, JSON/manifest consistency and shell syntax checks. Successful static
checks cannot establish CUDA compilation, numerical equivalence, fit quality,
runtime, memory admissibility or a public-score improvement.

## Remaining evidence

Require the one-B300 numerical admission, six real-shape profiles and explicit
resource reviews before any fit. Report all six fixed-E4 endpoints and all three
paired contrasts; preserve negatives and partial attempts. The official valid
remains exposed and two seeds do not establish significance or external validity.
No result or old control from PR99 is substituted for this new comparison.
