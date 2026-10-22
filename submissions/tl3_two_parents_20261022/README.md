# TL3 two-parent public quota probe — October 22

**Exploratory release requested by the user, not a promoted best model.** The fixed change removes TL3's distilled student and retains `fttall_tlead_s5@ep3` / `fttall_tlead_s6@ep3` at equal 0.5 coefficients. Original solution/native libraries/retained NPZ contents are unchanged. No fitting or coefficient search ran.

[Upload ZIP](tl3_two_parents_20261022.zip): 8,903,420 bytes, SHA256 `7a76f1cb90c8140a20d29b0566c1d0b9e3037511d22dc3b9c79881f4c84ba23f`.

The [build receipt](BUILD_RECEIPT_R1.json) binds the original TL3 archive and every retained entry. [Technical checks](TECHNICAL_CHECK_R3.json) pass all three native ABIs, offline Python 3.11, eight complete feature-only sequences, finite two-target output/warmup, bitwise A–B–A reset/replay and prediction parity with the saved pair after its declared normalization correction (maximum 2.3842e-7). Same-host callback cost is about 0.831 of TL3 across the measured sequences; competition runtime is unmeasured.

Question: how does this fixed equal-weight pair transfer publicly? A single score cannot establish the student’s marginal value without an exact matching public reference. The existing eight-component diagnostic favored the unnormalized retained pair by +0.00255, but it kept coefficients 0.33/0.33. This package uses the frozen unit-sum 0.5/0.5 ablation, so that number is not its exact post-clipping score. Selection followed exposed diagnostic results; full-fold quality and public transfer remain unestablished. All-data/unknown recursive ancestry prevents a clean held-out claim. Earlier `pair_ftt2` has four trunks and is a different package.

[Authority, source and failed-check progression](../../research/campaign_2026-10-22/quota_retained_pair_r1/README.md) preserve the explicit quota-probe exception and the two test-harness corrections. No authenticated portal connector or local browser debugging session was available at preparation, so no upload, submission ID, live quota or public score is claimed. Upload through https://wundernn.io/connectome/submit and bind its resulting portal ID to the exact ZIP hash above.
