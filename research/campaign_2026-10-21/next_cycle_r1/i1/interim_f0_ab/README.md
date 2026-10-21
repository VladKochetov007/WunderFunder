# Interim F0 A/B grade

This packet grades only the first matched F0 cell of the i1 screen: one A fit
and one B fit at seed1001. It reuses the complete-panel grader's frozen
provenance, evaluation-membership, NPY-header, official metric, fixed mixture,
and paired-component bootstrap helpers. It does not modify the six-fit grader
or any recipe. F2 and arm C remain pending, and the fold was previously
inspected, so the result is interim and exposed.

A and B both disable all156 history channels. A also masks raw and derived i1
inputs; B keeps those inputs. Their contrast measures the value of the raw
and event i1 inputs in this fitted cell. It cannot establish whether history
adds value; that requires C versus B, where only history differs.

Before reading target labels, the grader checks the pinned target-cache digest,
the exact F0 split and fit-subset identities, complete run provenance, all
4620 updates with zero skips, matching non-null initialization and normalization
identities, exact ordered evaluation membership, FP32 sidecars, and complete
float32 NPY payloads. It then reports official WP and both target scores for
A, B, and the fixed 90%B/10%A mixture, plus the four fixed position bins.
Overall A-minus-B and mixture-minus-B uncertainty uses2000 source-component
bootstrap draws with seed271828. It performs no coefficient search.

The mixture is produced in eight-sequence chunks. The memory preflight reads
only ZIP central-directory sizes, then estimates decoded target-cache storage,
prediction-file pages, the selected F0 arrays, a128 MB chunk reserve, and a
256 MB process reserve. It refuses to load targets if this estimate exceeds
1.4 decimal GB. This is a conservative estimate, not a measured peak.

After root integrates this packet into the canonical checkout, run:

\`\`\`sh
scripts/limited \\
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \\
  /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_ab/grade_i1_f0_ab.py \\
  --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_ab/runtime/grade_i1_f0_ab.json
\`\`\`

The expected output directory is created on first run; existing output files
are never overwritten. The report records source hashes and both run
identities so it can be compared with the root-owned panel ledger.


Root executed [RESULT_R1.json](RESULT_R1.json) under the [1CPU/1.4GB admission](../../I1_INTERIM_GRADE_ADMISSION_R1.json) ([log](../../i1_interim_grade_r1.log)). A−B−0.01590144, conditional component95%interval[−0.019737,−0.011972]; fixed90B10A−B−0.00002308,[−0.000309,+0.000266]. No useful mixture improvement is established. All four fixed bins favor B. C/F2 remain pending and this is not training-variation uncertainty.
