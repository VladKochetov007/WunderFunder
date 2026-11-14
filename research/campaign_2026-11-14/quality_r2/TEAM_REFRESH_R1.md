# Team evidence refresh during the quality wave

Read-only authenticated GitHub refresh, November 14. These are producer reports and immutable PR source heads; no remote fits or artifact imports were initiated by this refresh. Root's current running recipe remains unchanged.

| PR / retained snapshot | Head | Finding and implication |
|---|---|---|
| [92](PR92_SNAPSHOT_R1.json) | b107418a | B03 eight-fit since×KD receiver is implemented but unrun. Our completed local factorial uses a different Muon recipe; do not treat it as the same implementation or mix absolute scores. |
| [93](PR93_SNAPSHOT_R1.json) | 68e5b270 | Event clocks, frozen readouts, residual boosting and pair encoders remain proposals. Quiet hidden-state update proposals overlap this backlog; post-hoc holding was already negative. |
| [94](PR94_SNAPSHOT_R1.json) | aff5efb1 | Fullmix upload ID 1Y4RW907 is confirmed, hash a0fb8b1e254dd8be94301f6553f11b9e1b1ae758c9543f38aae3f021c944894f. The latest PR comment still says pending; the user's later receipt supersedes it: score .66543, 60m22s. Earlier host projections were not observed timeouts. |
| [95](PR95_SNAPSHOT_R1.json) | 9540f7da | Sixteen fixed endpoints complete. Since32 mean gain +.002904; tau8 −.000012 and tau128 −.000211 versus tau32; bilinear Transformer mean −.000236, mean endpoint .679156 vs .679392 linear. This supports keeping since32 and gives no new stronger teacher artifact for this wave. |
| [96](PR96_SNAPSHOT_R1.json) | 7926582a | Four K16/AdamW student fits: KD coefficient .2 gains +.001944/+ .001747, mean +.001846, on seeds203501/203502. Objective is label WMSE + .2 teacher WMSE, not alpha .2 convex targets with identical gradient scale. It independently supports KD in another recipe, not a matched alpha comparison. |
| [97](PR97_SNAPSHOT_R1.json) | f6207b38 | Six versus twelve attention heads at width768 is prepared, zero fits/profiles, no stronger endpoint available. No local reuse or launch is inferred. |

The PR95 table and PR96 endpoints come from fixed-E6 monitor evidence; PR95's since32 endpoints additionally have independent prediction exports agreeing within 6e-10. PR97 is source preparation only. Public receipt values are user-reported, not a new portal query from this session. All research validation remains exposed.

This refresh changes neither trial admission nor blending coefficients. Final refits and the paired future-teacher panel remain the best supported next work within the declared budget.
