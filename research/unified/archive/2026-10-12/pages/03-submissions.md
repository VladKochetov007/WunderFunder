> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Best current packages and serving evidence

The create-only package records are in
[submissions/10-12-best-r1](../../../../../submissions/10-12-best-r1/README_R1.md) and
[submissions/10-12-best-r2](../../../../../submissions/10-12-best-r2/README_R1.md).
Candidate order B, A, C reflects runtime-risk preference, not a local quality
ranking. Public observations below were transcribed by the user from the portal
and were not independently browser-verified.

| Candidate | Fixed endpoint | ZIP bytes | ZIP SHA-256 | User-reported portal result |
|---|---|---:|---|---|
| B | 0.5 H390 s3 + 0.5 exact merged Z4 | 9,053,484 | `486a34c46989361428aa2e34f1c43dd246e2b7293f6dfcb5dc5031c989420440` | 0.6668, 25:56, 634 MB, portal `09853lyw` |
| A | 0.5 H390 s3 + 0.5 H390 s4 | 8,556,250 | `8834774c58ff31a77547fc8ec59988aef61e1bd1e7198c4c52d1d3835c43e6e5` | 0.6656, 24:31, 632 MB, portal `0rg9ujj5` |
| C | 0.25 H390 s3 + 0.25 H390 s4 + 0.5 exact merged Z4 | 12,680,868 | `6013d20a7ff26ce6ef46821f80ed01da848ef4da565e2cd91656417ddd5186ea` | 0.6671, 36:36, 642 MB, portal `1ya21vpd` |

Each ZIP is under 20 MB, has root-level solution.py, and contains the
precision-aware native Q8VNNI scorer and dispatched library tiers. Exact ZIP
extraction tests ran with CPython 3.11.16 and Q8VNNI mode. Each callback
profile covered ten unchanged complete 20,000-row validation sequences, two
reset-separated passes, 400,000 callback calls, None during warm-up, two
finite values on required rows, and bitwise-identical reset/replay outputs.
Package bindings, ABI checks, prediction receipts and independent review are
linked from each package README and
[the independent package review](../../../../../artifacts/submission_review_10_12_best_r1/SUBMISSION_PACKAGE_REVIEW_R1.json).

The H390 source members are
vt_all_h390_ds_homogeneous_s3_r3@ep5 and
vt_all_h390_ds_homogeneous_s4_r3@ep5, with distinct seeds and checkpoint
hashes. Their lineage records since_weight=32, since_floor=0.25, and
10,606 train plus 1,873 validation sequences in fit membership; they are
all-data endpoints. Their Muon parent lineage differs from the strong F1
student reference. The Z4 member is the unchanged merged endpoint
bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1 paired
with z4_seed99_student_hft_alldata_v2@ep0, not a bare Z4 trunk. Its NPZ
member hash is preserved in the package receipts.

Local callback timing came from an Intel i9-11900H and is capacity evidence,
not an AMD EPYC portal guarantee. Initial A/B/C callback means were 44.97,
49.82, and 83.33 microseconds per row; concurrent CPU activity made those
unpaired observations noisy. A later sequential reference-C-reference-C
probe measured C at about 1.44 times reference callback cost, with substantial
pass variation. Portal elapsed times are separately observed end-to-end
package runs. Neither local timing nor the three public points establish a
linear runtime-quality law.

Two further native Q8VNNI packages are technically reviewed and runnable, but
their quality is unmeasured:

| Candidate | Fixed mixture | ZIP bytes | ZIP SHA-256 | Local callback |
|---|---|---:|---|---:|
| D | 0.25 H390 s3 + 0.50 Z4 + 0.25 merged Z5 | 13,178,260 | `bf9f48dbb05668181038d28386a076774d41146efb7c4f1680fcd724a303df09` | 113.60 µs/row |
| E | 0.1875 H390 s3 + 0.1875 H390 s4 + 0.375 Z4 + 0.25 merged Z5 (0.75*C + 0.25*Z5) | 16,805,643 | `2aed2137c347344146e3481c83388d64bd5060b1c13bb66554b0bf9cc0027024` | 152.62 µs/row |

The source-qualified Z5 pair is
bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2@ep1 paired
with z5_seed9_student_hft_alldata_v2@ep0. Its head fitting used 1,505
supplied validation groups. A targeted search found no matched clean-CV grade
for that Z5 endpoint. Its older i16 ZIP has a user-reported 0.6664 result, but
this does not establish Q8 quality because served predictions change with
quantization. The exact Q8 package technical review covers package/source/
runtime only; no official metric has been computed for D or E. E's paired
local capacity heuristic is around 50.9 minutes when scaled from C's portal
duration; this is not a measured portal runtime and is not evidence E will
complete under the 60-minute limit. See the
[D/E package review](../../../../campaign_2026-10-12/independent_reviews/submissions_10-12-best-r2/D_E_TECHNICAL_REVIEW_R1.json)
and [paired C/E timing receipt](../../../../campaign_2026-10-12/z5_blend_runtime_pair_r1/README_R1.md).

The exact native full-fold scorer is a separate, previously exposed Q8 result:
F0 0.6834944365, F1 0.6700246323, one pipeline seed per fold. It processed
about 9.36 million and 9.14 million rows, with measured callback totals of
201.729 and 198.042 seconds and a 512.34-second worker wall time including
decode and scoring. It has local bit-parity and source/membership receipts,
but does not establish AMD target runtime or the quality of A/B/C/D/E. The
terminal record is
[QUALITY_RESULT_R2.json](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_shared_cleanfold_q8_r2/quality_output_r2/QUALITY_RESULT_R2.json).

An earlier Q4 conversion lost 0.00180125 against Q8 on its ten-sequence
screen; Q8 differed from FP32 by 0.00007090 on those same inputs. This is a
screen, not full validation. No useful native INT4 package was established.
Keep Q4/mixed precision/QAT open only for a changed method with measured
quality, size, and runtime; the tested naive Q4 conversion does not justify
defaulting to Q4.
