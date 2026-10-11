# Can a validation fold be a better public proxy?

Question: user asks whether a fold resembles public enough to improve submission selection. We can investigate recipe-ranking transfer; no current evidence establishes a uniquely public-like fold. This is a diagnostic of existing evidence, not permission to choose a favorable fold after seeing a candidate's result. ADR0030 remains in force unless the user explicitly changes it: valid local/CV evidence selects, public confirms.

Absolute score similarity is insufficient. Fold difficulty, target/magnitude distribution, class mixture, fit membership and model ancestry can all change a global correlation. Fold1 is numerically closer to public than the easier F0/F2 panels, yet its selected scratch model beat the student locally and lost substantially publicly. Higher F0/F2 scores alone do not imply poorer transfer. Average fold WP is also not identical to the official metric computed from pooled weighted moments.

Four reported fold1-held-out endpoints have both local and public outcomes:

| Endpoint | Reported exposed F1 local | Public |
|---|---:|---:|
| Earlier student | .66938 | .6639 |
| New T3 student, ZIP7 | .67103 | .6652 |
| Selected scratch, ZIP9 | .67195 | .6616 |
| Fixed half student/scratch, ZIP8 | .67329 | .6650 |

The descriptive Spearman and Kendall correlations of these four rankings are both0.0; [the exact calculation](PUBLIC_FOLD_AFFINITY_DIAGNOSTIC_R1.json) is saved. This is a tiny dependent example: the pair shares both component predictions, the students share a lineage, the local scores are exposed/selected float evidence, and tiny public differences may be within repeat variation. It is not a population estimate, a significance test, or evidence that another fold has correlation1.0. Full-data ZIP4/5 are deliberately excluded from this exact-weight held-out comparison.

The stronger existing bank has29 recipe pairs across F0/F1. Claude reports Pearson.82/Spearman.76 with residual standard deviation.0015, but that compares local folds, not either fold with public. The H230 baseline correction and completed F0 old-pipeline scores show that cross-fold matched controls help explain the scratch miss. They do not yet identify F0 as a replica of the public distribution.

Useful next analysis:

1. Join each public result to exact ZIP hash, full-data versus fold-held-out membership, serving precision, seed, checkpoint and teacher/initializer source. Use same-recipe clean-fold siblings explicitly, not a mislabeled score of final full-data weights.
2. Compare recipe improvements/rankings against the same strong H230 or current family control on each fold. Separate teacher/student/scratch families and propagate seed variation; dependent blends are not independent trials.
3. Assess prediction of held-out **families** or later upload batches, rather than fitting fold weights and evaluating on the same public observations. Shrink any inferred preference toward balanced cross-fold evidence; report missing coverage and selection exposure.
4. Diagnose supplied-data class/run/magnitude/mask differences and component-level behavior with existing bank/moment tools. Hidden test inputs and labels are not distributed; no public sample membership is known to us.
5. Rank complete fixed served blends using robust CV and actual CPU feasibility. Treat public canaries as transfer checks. Do not manufacture an expected public score by adding a local delta.

The immediate scientific test remains the clean F0 T3 teacher/student pipeline alongside the reconstructed strong old family. Local F1 reconstruction is also running. Additional repeats on already-selected F1 alone would not resolve the public-affinity question.

Sources: user-provided portal results, `review_public_gap_20261011/REVIEW_PUBLIC_GAP.md` (especially sections2/25), the existing `prediction_bank/LATEST_CROSSFOLD_F0_F1.json`, and local supplied-data documentation. Live official rules/data pages were opened but returned a JavaScript shell to the web reader; no fresh rules-content verification is claimed.
