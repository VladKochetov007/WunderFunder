# Response to Claude's comments and current execution evidence

Source: `writeups/cc_comments.txt`, read 2026-10-11 23:05 UTC. This is a controller interpretation, not a new independent review. Historical monitoring snapshots in that file do not describe the current lanes.

Claude's warning about the H388 trio was correct. The uploaded ZIP12 timed out after the portal reported 61m56/647 MB. Completion is censored: 61m56 is not a measured successful runtime. Its READY recommendation has been withdrawn. Float16 transport reduced ZIP size while retaining Q8 inference; it did not reduce the recurrent workload. We should have given the runtime warning more weight before recommending the trio.

The strongest scientific recommendation is the matched H230 baseline. New mechanisms must be compared against the strong own H230 lineage on the same evaluation groups, not only against a weaker plain/combo control. Fold-1 gains are discovery evidence, not a portable advantage. The clean T3 fold-0 pipeline remains necessary. Its completed paired Kaggle profiles did not become fitted teachers: R4 stopped at host-memory admission. R5 is an identified resource-handoff repair with unchanged science and strict reservations; it is not a scientific negative or a completed cross-fold confirmation.

The student-pair recommendation is a sensible prior, not a universal family rule. H390 is an explicit changed-parent/weighting interaction: it uses own Muon-all/prop0 parents and since weighting, rather than the publicly confirmed fold student's combo parent and since0. The fold student's public transfer does not certify this final recipe. Likewise, the .672574 mixed float prototype is related evidence, not an exact score for the full-data H390/H388 package. Neither full-data final has a held-out grade. The final decision must compare complete fixed endpoints, their clean ancestors and native precision, then apply the runtime constraint.

Claude's numerical minutes-per-member conversions should be treated as planning heuristics. New matched one-core/Python3.11 callback measurements cover ten complete sequences, 400,000 calls and deterministic replay:

| Exact endpoint | Local median microseconds/row | Portal evidence |
|---|---:|---|
| Public ZIP4, old pair | 52.74 | Completed in 27m1s, public .6668 |
| Public ZIP5, old single | 21.77 | Completed in 15m41s, public .6649 |
| ZIP11, H390/H388 pair | 74.84 | Not uploaded |
| ZIP10, H388 pair | 104.48 | No completed portal measurement |
| ZIP12, H388 trio | 158.62 | TIMEOUT, 61m56 |

The compact mixed pair has substantially more runtime margin than the trio. Intel-to-EPYC transfer, shape dependence, frequency, cache sharing, Python/scorer overhead and portal variability remain unmeasured. Two historical durations do not establish a universal conversion or an upper bound. The PGO experiment is being measured with interleaved controls because host timing changed; compiling successfully is not a speedup.

New completed local evidence: exact own F0/seed3 parent CP3 .6828233414, student CP1 .6831418134, and head-finetuned CP0 .6824217225, official fixed-mask/clip2/magnitude-weight metric on 468 exposed groups/141 components. HFT minus student is -.00072009, paired component CI [-.00270527,+.00116850]. It is one training seed, not fresh CV and not a universal rejection of HFT.

The prospective fixed 50/50 student–HFT comparison is .6834706265, +.00032881 over the student, paired CI [-.00071700,+.00132679]. This small uncertain conditional contribution matters: the weaker standalone HFT should not be rejected as an ensemble component automatically. The two checkpoints have exactly identical frozen input/RNN/normalization tensors; only four head tensors changed. That provides a concrete future exact backbone-sharing opportunity. It does not establish useful shared backbones among the current H388/H390 members, which have different weights. No local delta is converted to a public-score forecast.

The fit has finite checkpoint/state, 870 optimizer attempts, 868 applied updates and two protected AMP skips. The earlier strict profile remains throughput-invalid after one nonfinite gradient/skip. Separate resource-only admission and independent terminal checks authorized this bounded fit and export; no profile failure was relabeled as success.

Action: keep the timeout retraction, finish the corrected clean-fold cohort and old full-pipeline counterpart, select only the strongest complete feasible endpoints, and give exact reuse/cache improvements precedence over speculative member count. New user authorization for two independent Sol6.1/xhigh algorithmic reviewers overrides the default Luna preference only for those two review roles; implementation remains owned by the Luna native actor.
