# E412 public confirmation: 0.6655

The user reported **0.6655** for `c20261004_combo_s3_i16`, submission **10tq248b**, on October 4. This is the new recorded public best: **+0.0014** over 0.6641, with **0.0005** remaining to 0.6660. The portal reports **37m47s (2267s)** and **625 MB**, within the 60-minute hard limit and 52-minute campaign margin. It reports 1/5 submissions used today and 21 total. The displayed submission time is 14:58; the portal timezone was not explicitly supplied.

The submitted predictor is the fixed seed3 GRU256 Muon/chunk1000/stopped-bootstrap recipe, epoch 3 parent, followed by epoch 0 head fine-tuning, combined in an equal-weight shared-trunk predictor. The historical holdout was excluded from fitting and selection. The native package uses i16 and fixed membership.

- Archive: `submissions/c20261004_combo_s3_i16/submission.zip`
- SHA256: `21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb`
- Sources: `c20261004_combo_s3_final_r1@ep3|c20261004_combo_head_s3_final@ep0`
- Raw reported result and interpretation: `E412_PUBLIC_RESULT.json`
- Numerical/reset checks and the earlier runtime forecast: `E412_READY_REVIEW.md`

This confirms a useful public gain for the complete recipe. It does not separate the optimizer, chunk cadence and bootstrap contributions or resolve the exposed-fold/teacher-lineage limitations. The public gain is smaller than E410's complete-predictor local delta +0.002414 and the earlier parent-level gain ~+0.0018. One result does not establish a transfer slope or justify a narrow forecast for another candidate.

The actual runtime is faster than the earlier uncertain 40.27–43.55-minute projection. Use the observed 2267-second endpoint as an additional runtime reference while retaining the forecast history. A second dense copy cannot be assumed to fit simply because one fits; pair engines and compression still need direct measurements.

Continue matched seed4/pair diversity, width, volume-feature and compression research, alongside the already-running specialist and QAT experiments. Preserve the creative/retest slate: this result changes the incumbent, not the eligibility of those hypotheses. Complete-predictor comparisons and measured deployment cost remain required. The four remaining daily uploads can test verified, fixed candidates; no candidate is promised to cross 0.6660.

This entry records a manual user upload. The agent did not submit to the competition.
