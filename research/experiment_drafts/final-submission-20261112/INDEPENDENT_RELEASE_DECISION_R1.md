# Independent release decision — 2026-11-12

## Recommendation

**SHIP conditionally**: allow the frozen full-all-label TL3 + 10% LSTM candidate to proceed as one prospective public submission after the exact package passes the CPU engineering gates and its receipt binds the final ZIP bytes to the admitted weights and recipe.

This is a decision to test a plausible incumbent-derived candidate, **not** a claim that it improves public score. Do not use any post-fit valid score as a model-quality gate: the final model is trained on all 12,480 official sequences, so those labels are training data. CPU checks should establish package identity, callback compatibility, finite outputs, reset/causality behavior, repeatability/parity, resource use, and checker runtime only. Freeze the recipe before upload; do not retune it against post-fit valid results.

## Evidence and limits

| Evidence | Result | Decision relevance |
|---|---:|---|
| Current signed-in public record, captured in the final-submission authority | TL3, ID `1LMEHFI3`, WP **0.66832** | Strongest known team public result; supports keeping TL3 as the base. |
| Latest W18 single-model public submission | ID `1H7KP3OQ`, WP **0.66453** | Lower than TL3; does not support replacing TL3 with W18 alone. |
| Historical fixed-mixture comparison in PR76 | 90/10 TL3/LSTM exceeded TL3 by **0.00012464** on exposed valid | Weak directional support only: selected from six mixtures on already exposed valid, no uncertainty estimate, and no CPU-served result. Do not describe it as an independent gain or public-score forecast. |
| Clean since32 K16 train-only screen, two seeds | **0.686306 / 0.685904** official full-valid monitor | Useful clean research evidence, but below the historical TL3 valid scores and without a matched TL3 comparison under the same source. It does not justify forcing a since32 replacement before a public trial. |
| Legacy lineage caveat | PR76 flags incomplete consumed teacher-cache ancestry for historical TL3 members | Do not use those historical research scores as clean chronological evidence. The proposed final candidate is admitted on its separately verified all-label ancestry; retain that distinction in the release receipt. |

The all-label refit is authorized for the final recipe and avoids presenting valid labels as held out. Reusing the already completed, qualified all-label DAG is preferable to spending the remaining window on an untested since32 all-label refit: since32 is a reasonable future challenger, but its current evidence neither beats the incumbent publicly nor establishes complementarity with TL3.

## Required release record

Before upload, the receipt should pin the exact ZIP SHA-256, every member checkpoint hash, recipe weights, all-label membership/split identity, ancestry/admission evidence, source and inference entrypoint, and successful CPU gate outputs. Record the portal submission ID/result afterward. Keep the submission language to “prospective trial based on the known public incumbent and a weak, exposed-valid mixture screen”; report no independent-valid or expected-public improvement.

## Evidence locations

- [Final-submission authority and shortlist](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r1/AUTHORITY_R1.json)
- [B300 all-label model inventory](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r1/b300-final-models-inventory-r1.json)
- [W18 all-label release record](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r1/W18_RELEASE_R1.json)
- [PR76 candidate/status evidence snapshot](C:/Users/admin/Documents/Codex/artifacts/cluster-all-experiments-20261112/new-pr-review-r1/PR76_FILES.json)
- [Independent since32 evidence review](C:/Users/admin/Documents/Codex/artifacts/cluster-all-experiments-20261112/since32-independent-evidence-review-r1.md)

