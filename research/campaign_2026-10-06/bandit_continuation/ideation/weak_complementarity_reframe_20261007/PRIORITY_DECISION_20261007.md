# Allocation decision: weak-model complementarity

## Decision

**Stop** the exact E-514 pool, correlation selector, and scalar-mixture rule.
**Reframe and retain** the broader idea that a weak, diverse model may contain
useful residual information. Low correlation is a candidate-generation clue,
not a selection objective. E-514's least-correlated member was still strongly
correlated with the anchor, and every calibrated blend reduced to the anchor.
That closes the tested pool/rule; it does not test a new representation,
residual corrector, training-time use, or a genuinely different specialist.

## Priority within the active campaign

The complementarity branch does not displace the current leading work:

1. Recover and reconcile the independently trained own-baseline/raw-teacher
   Kaggle path. E-508-v4 terminated before fit because the pinned trainer
   rejected `--device cuda:0`; it allocated no CUDA memory and did not OOM.
   E-508-v5 changes only that selector to `cuda`, passed an independent packet
   review, and is pushed profile-only. The first poll reports `RUNNING`, but
   logs are empty and payload phase is unobserved. E-463-v2 remains `QUEUED`
   on the separate TPU pool. These are execution states, not model evidence.
2. Keep the separate E-511 comparison uncertain; its current exposed-fold
   interval crosses zero.
3. Keep heterogeneous-teacher compression as its own untested student arm.
   A newly trained candidate from that or another independent lineage could
   later supply a meaningfully different complementarity test.
4. Keep the new complementarity hypotheses as deferred exploration. Do not
   spend on another direct blend from E-514's fixed candidates.

This ordering follows the current campaign reflection and resource gates; it
does not rank ideas by their novelty alone.

## What to test when a new candidate exists

First make its ancestry and group-OOF predictions auditable. Then test whether
its output carries stable target-aligned information about the anchor's
cross-fitted residual on separate connected groups. If the signal is absent,
stop before fitting a corrector. If present, compare one small output-only
residual satellite against an input-only corrector, shuffled-candidate control,
and unchanged anchor; freeze the complete endpoint and score it with
`global_wp_clip2_v1`. A second pathway, disagreement-guided student training,
starts with a no-fit disagreement/error association screen and proceeds to an
equal-compute uniform-versus-guided-versus-sham training comparison only if
that screen is informative.

All inspected validation folds remain exposed. These comparisons can guide
development but cannot be described as fresh confirmation. No numerical
prior, expected public score, or local-to-public translation is assigned.

## Scheduled review

The one-shot read-only Grok 4.7 bandit reflection remains scheduled for
2026-10-07 08:59:24 Europe/Kyiv. Its updated prompt asks it to inspect E-508-v5
payload/terminal evidence if available, E-508-v4's diagnosed CLI failure,
E-463-v2 status, this priority decision, the new hypothesis cards, and the
tree, then recommend whether to stop, hold, reframe, or continue. Its output
will be treated as allocation advice, not scientific score evidence.
