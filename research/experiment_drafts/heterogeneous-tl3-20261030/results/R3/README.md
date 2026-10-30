# Complete fixed ensemble comparison

All six new fits completed their frozen four-epoch recipes. The missing original
W18 dependency completed its separate two-epoch recipe. Every comparison below
uses all1873 official-valid sequences and3,528,273 scored rows, with10325 train
members and zero validation fitting members under the adopted official boundary.
Direct teacher memberships are qualified; legacy consumed-cache byte ancestry
retains the separate limitation recorded in the report and independent review.
Canonical fresh TL3 is34%W18 +33%tiedlead seed5 +33%tiedlead seed6. Historical
`tl3.zip` and the separately named fbc replacement are different endpoints.

## Actual results

[The complete12-row table](COMPARISON.md), [producer inputs](INPUTS.json) and
[full report](REPORT.json) retain t0/t1/WP, every coefficient and artifact hash.
No coefficients were fitted to valid; every new endpoint is fixed zero-based3.
The best observed epoch is retained in the weights packet as a separate role.

- Canonical TL3:0.6923774245652725. The best tested fixed mixture is90%TL3
  +10%GALAM:0.6924964468441978, delta+0.0001190222789252.
- The heterogeneous teacher pool gives0.6891635642037302 versus0.6886081210674475
  for its two-GRU control pool:delta+0.0005554431362827.
- Paired student seed1301:control0.6885969711677301,
  heterogeneous0.6892163196970573, delta+0.0006193485293272.
- Paired student seed1302:control0.6887607762940369,
  heterogeneous0.6889800502140899, delta+0.0002192739200531.
- The mean paired singleton improvement is+0.0004193112246901. Averaging the two
  heterogeneous students gives0.6898425573461273 versus0.6894367838826270 for
  control students:delta+0.0004057734635003.
- Adding20%heterogeneous-student mean to TL3 yields0.6922595270636154,
  delta-0.0001178975016571. Adding all three alternate families at10%each gives
  0.6922021335033716, delta-0.0001752910619010. Preserve both negative results.

The pool and transfer results are directionally positive at both tested student
seeds, but the mean transfer remains below the working+0.0005 threshold. Two seeds
do not establish significance, teacher-seed robustness or hidden transfer. This
composition merits targeted follow-up; it does not justify automatically scaling
every family or adding the students to TL3 at the tested20%weight. The small GALAM
gain is an exposed-validation candidate result, not a public-score forecast.

## Independent numerical agreement

[Independent actual-array receipt](INDEPENDENT_RESULT.json) and its
[25-output table](INDEPENDENT_COMPARISON.md) verify13 singletons and12 ensembles,
both targets and pooled WP. The verifier imports no production scorer/mixer/metric
or Torch. It independently joins sidecar row groups and actual GALAM scoring-mask
offsets, constructs raw hierarchical FP32 blends, clips after blending, and uses
two float64 centered passes. Maximum absolute disagreement is5.551115123125783e-15
against the frozen1e-10 tolerance. Both passes cover3,528,273 rows; runtime33.29s,
peak RSS328,794,112 bytes. Numerical agreement does not make validation unseen.

[Actual paired-admission evidence](../../receiving/PAIRED_STUDENT_ADMISSION_ACTUAL_R1.json)
shows exact matching normalization, non-treatment args and trainer source within
each seed. The [independent review](../../INDEPENDENT_REVIEW_R1.json) states the
separate source/ancestry and inference limits.

## Execution and retained weights

[Terminal resource summary](../../receiving/TERMINAL_SUMMARY_R2.json) records
six times25,760 plus12,900 W18 successful updates:167,460 total, with zero recorded
nonfinite-gradient steps. Fit-command reservation time sums to4.4289L4 GPU-hours;
the union of actual GPU-command intervals including support is5.0944GPU-hours.
These include loading/evaluation and are not FLOPs or SM-active hours. The final
GPU inventory contains no scoped experiment PID; the existing GPU4 service stays
outside this campaign.

[Node checkpoint export](../../receiving/CHECKPOINT_EXPORT_R1.json) contains all
seven completed models and every original epoch snapshot, with fixed/runtime/
best-observed roles, original normalization/provenance/history and separate source
ZIPs. Every member passed CRC/SHA256 roundtrip checks, and transport to the local
backup matches the node checksums. External data and ancestor weights remain
separate. GitHub download verification is recorded after actual publication.
