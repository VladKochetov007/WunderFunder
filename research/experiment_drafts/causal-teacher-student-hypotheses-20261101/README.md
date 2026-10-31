# Causal teacher–student hypotheses: research portfolio

**Status: proposal only.** [Draft PR #55](https://github.com/VladKochetov007/WunderFunder/pull/55) collects eight distinct, reviewable ideas
for the Wunder Connectome competition. It contains no implementation, accepted
receiver command, training job, result, new submission ZIP or benchmark claim.
No local tests, model execution, GPU profile, training, inference or upload were
performed while preparing it. Every initial panel needs its own frozen config,
source-qualified ancestry, adapter and receiver acceptance before it can run.

The user requested one draft PR covering the eight ideas, so this packet is a
portfolio index rather than one pooled experiment. Future implementation and
results should each be recorded under a mechanism-specific H/E ID, with separate
code reviews and outcome receipts. The repository workflow's per-idea PR rule
applies when those independent studies are implemented or submitted for review;
this umbrella does not merge or replace them.

## Contents

- [HYPOTHESES.md](HYPOTHESES.md) — mechanisms, paired controls, falsifiers,
  teacher costs, CPU implications and overlap with existing research.
- [recipe.json](recipe.json) — machine-readable, non-executable inventory of
  the eight proposed panels, two seeds and common admission conditions.
- [PR.md](PR.md) — the eleven-section experiment review contract and the PR body.
- [ideas index](../../unified/ideas/README.md) — project navigation; individual
  proposed H/E records are appended to the existing research ledgers.

## Evidence anchor and boundaries

[PR52](https://github.com/VladKochetov007/WunderFunder/pull/52#issuecomment-6532373933)
reports canonical TL3 full-valid WP 0.692377425 and TL3 + 10% GALAM 0.692496447;
the latter's delta is +0.000119022 on an exposed evaluation surface. The
[PR53 final follow-up](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545905799)
reports the separate TL3 + 10% LSTM mixture at 0.692502063, also exposed and not
independently confirmed. PR53's 6/6 positive KD comparisons were an inner proxy,
not a public score. No local score is a predicted leaderboard result.

[PR54](https://github.com/VladKochetov007/WunderFunder/pull/54) separately
proposes a final-fit-only TL3 + LSTM on all supplied labels. Its requested full-data
members include official-valid groups and cannot be reused as clean controls,
initializers, normalizers or teachers for this portfolio's train-only evaluation.
The final-fit plan is separate from this proposal and its runtime state is not
established by this note.

Related prepared work: [PR47](https://github.com/VladKochetov007/WunderFunder/pull/47)
compares a Transformer teacher with labels-only GRU students;
[PR48](https://github.com/VladKochetov007/WunderFunder/pull/48) studies width
versus context; [PR50](https://github.com/VladKochetov007/WunderFunder/pull/50)
prepares GALAM block/BatchNorm diagnostics; [PR24](https://github.com/VladKochetov007/WunderFunder/pull/24)
proposes a GRU/Transformer/Mamba comparison. The notes below identify incremental
comparisons rather than claiming all underlying ideas are new.

Before interpreting a deployment candidate, audit exact served ZIP predictions
against the scored Torch model, including 4000-row feature boundaries, then
compare native FP32 with the actual INT8 mode. The existing
[PR51](https://github.com/VladKochetov007/WunderFunder/pull/51) covers an fbc
replacement rather than canonical W18 and has no published CPU equivalence for
the canonical candidate. Its native library's semantics are opaque; the risk is
a question, not a demonstrated bug. Serving checks are engineering work, not
scientific fits or evidence that a validation/LB gap has one cause.

Source and split identities must be verified against the actual fitted members;
train-only and component separation do not prove hidden chronology. The official
development validation is already exposed. For the BF16/BF16x9 Transformer
scope, [the user's scoped amendment](../b300-transformer-distillation-runtime/PRECISION_AMENDMENT_20261024.md)
supersedes historical strict-FP32 instructions; other precision changes require
an explicit dated recipe. No quantization is part of scientific training. The
submission environment is Linux Python 3.11, one vCPU, 16 GB RAM, 60 minutes,
20 MB uploaded ZIP and no network; successful training is not CPU admission.
