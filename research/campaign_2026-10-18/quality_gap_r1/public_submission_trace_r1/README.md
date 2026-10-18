# Public submission trace — user transcript, 18 October 2026

The user supplied all 100 displayed Team kotiki submissions in response to the incumbent-identity question. The highest successful score in that supplied list is **0.66832**, submission **`1lmehfi3`**, **`tl3.zip`**, submitted by **NaiveBayes**, displayed **23:52 2026-10-15**, **39m12s**, **656MB**. The near-tie `tl4.zip` is separately identified as `219y3mlk`, **0.66830**, **50m7s**, **668MB**, at the same displayed minute. Their difference is 0.00002 in displayed public score; no uncertainty or causal benefit of a member change is established.

This resolves the public ID/filename ambiguity in the earlier [identity audit](../INCUMBENT_IDENTITY_AUDIT_20261018T1550Z.md). It does not verify the portal-uploaded bytes. The archived local `tl3.zip` is 12,538,394 bytes with SHA-256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`; the October 17 inventory reports an identical server copy. Those are local/server archive identities, joined to the reported public submission by filename, not a recovered portal payload hash.

## Evidence and coverage

- [SUBMISSIONS.json](SUBMISSIONS.json) contains all 100 rows in user-supplied order: ID, filename, submitter, displayed date/time, status, duration, memory, score or null, comment UI and parsed comment. Score text is retained to preserve displayed precision.
- [TRANSCRIBED_ROWS.tsv](TRANSCRIBED_ROWS.tsv) is the normalized manual transcription used for parsing. It is not a verbatim raw-message copy.
- [VERBATIM_EXCERPT.txt](VERBATIM_EXCERPT.txt) preserves exactly the supplied header and first five full submission blocks, through `tl3.zip`, with a final file newline added. It deliberately does not claim to preserve the whole message. The full original message remains in the conversation.
- [VALIDATION.json](VALIDATION.json) reconciles 100 rows with the supplied “100 total” header: **100 unique IDs, 95 SCORE, 4 SCORE:TIMEOUT, 1 CHECK:ERROR**. There are no duplicate IDs. Two filenames recur across distinct IDs; filename equality alone is not byte equality. The supplied `20o154uy` has a successful score despite a displayed 61m22s duration, and is retained as reported. No timezone was supplied, so there is no UTC conversion or correction from filename dates.
- [INCUMBENT_BINDING.json](INCUMBENT_BINDING.json) joins the two identified packages to the hash-bound [October 17 inventory](../../../unified/reviews/audit-2026-10-17/submission_inventory.json) and [model catalog](../../../unified/models/CURRENT_MODEL_CATALOG.md), recording the exact source digests and copied inventory entries. No ZIP was reopened, rehashed or uploaded for this packet.
- [MANIFEST.json](MANIFEST.json) seals the files in this packet. This is metadata validation, not a new model evaluation.

## What the incumbent contains

The inventory's exact `tl3.zip` member recipe is:

| Coefficient | Package weights | Package member name | Catalog lineage |
|---|---|---|---|
| 0.34 | `weights_s5.npz` | `distilled_s5@ep1` | `w18final_distilled_s5@ep1` |
| 0.33 | `weights_tl5.npz` | `fttall_tlead_s5@ep3` | Same name |
| 0.33 | `weights_tl6.npz` | `fttall_tlead_s6@ep3` | Same name |

The fixed coefficients sum to 1.00. `tl4` instead gives 0.25 each to distilled s5, distilled s6, tied-lead s5 and tied-lead s6. This identifies the package composition without attributing the tiny public difference to a particular member or blend coefficient.

The catalog records the distilled family as GRU256×2 with **211 inputs** (112 raw plus 99 causal event features), warm continuation of fbc1k parents, two epochs of AdamW and distillation alpha 0.7. The tied-lead family has **263 inputs** (211 plus 52 lead features), a shared instrument encoder and GRU256×2, four-epoch scratch parents, Muon/AdamW, EMA and alpha 0.5. Its teacher treatment and fitting weights differ; the w18 teacher's stored coefficients sum to 0.8125. These source distinctions matter more than labeling the whole recipe “distillation.” They are documented catalog facts, not newly replayed training evidence.

Both production families fit **train plus valid**, so their supplied-valid scores are fitted diagnostics. The new cold labels-only F0 width/depth pair does not reproduce this fit scope, feature set, initializer, teacher treatment or multi-member recipe. Its public **0.65839** is **0.00993** below this now-identified incumbent. The trace establishes this descriptive regression; it does not isolate which recipe difference caused it.

## What is still needed

The public score-to-ID-to-filename link is now supplied by the user. The remaining identity gap is the uploaded portal payload SHA; the archived local file has not been proved identical to that upload.

A clean held-out local counterpart of the **complete tl3 recipe** is still not established in this packet: exact source/config bindings, fold-qualified fitting membership of every member, teacher and warm-parent ancestry, consistent preprocessing, and same-row predictions against the new candidate are needed. Production train+valid checkpoints cannot be relabeled as clean F0 controls merely by selecting F0 rows afterward. The warm Main T3 control currently being reconstructed is a useful separate strong recipe, but it is neither this three-member tl3 package nor evidence that its production training excluded F0.

Preserve the ongoing Main T3 diagnostic and label it accurately. Use this trace to make the subsequent incumbent-relative comparison concrete. Do not tune a new blend or calibrator to these public scores. The list also supplies no evidence of other teams' methods, and unmatched public rows cannot establish causal architecture, feature, distillation or ensemble effects.
