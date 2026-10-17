# R2 native q16x endpoint receipts

This evidence bundle contains the frozen `RELEASE.json`, its nine referenced small JSON receipts, three later interpretation/parity receipts, and the optional adapter-result figure. The submission ZIP, weights, predictions, data, and logs are not copied.

The selected archive identity is `geometry_adapter_r2_q16x_20261018.zip`, 5,323,200 bytes, SHA-256 `955bff715871ff8a977d96a43f4edb89b1f5ecec72ae8ee49c0c2b53a3b6688b`. The release manifest marks it as an explicitly exploratory endpoint; no portal upload occurred.

The native q16x callback was graded over all 468 exposed F0 groups (9.36 million callback rows, 808,863 scored rows) at WP **0.6804519697**. Its Torch FP32 reference scored **0.6804499040**, a +0.0000020656 served-minus-reference difference. The underlying adapter fit remains negative against its fixed parent (0.6804499040 versus 0.6812265132); the tiny precision delta is not a model gain or public forecast.

The exact ZIP passed its offline Python 3.11 smoke and archive CRC/path checks. The later uint32 receipt confirms exact same-precision q16x native-cache output on eight source-qualified groups, including signed zero: 0 of 318,416 values differed. A separate, predeclared FP32 native-versus-Torch panel **failed** its `rtol=1e-5, atol=1e-6` elementwise tolerance: 287,989 of 318,416 values were outside tolerance; maximum absolute error 0.0007533133, mean absolute error 0.0000555117, RMS 0.0000763595. The native model uses `Activations::Fast`; the cause was not isolated and the tolerance was not relaxed. Generic `passed` fields in the serving receipts mean replay/output checks completed, not that Torch equivalence passed. The q16x endpoint itself was scored directly end to end.

Full F0 replay took 633 seconds wall and 628 seconds process CPU. Callback-only measurement projects to 40.43 minutes over 39.4 million rows; the full-replay-cost projection is about 44.41 minutes. These are local row-cost estimates, not complete competition runtime or a target-host guarantee.

`RELEASE.json` binds the archive identity and its nine receipt hashes. The three later JSON receipts add a tolerance assessment and correct the scope/label of the historical native-cache comparison; their SHA-256 values are recorded below. Some verbatim receipts retain absolute paths from the producing machine and hashes for private prediction/cache files. These are archival provenance fields, not required B300 inputs or dependencies; payloads remain outside this packet. The [R2 training summary](../README.md) preserves the negative one-exposed-F0/seed3 adapter result and its fit limitations.

## Included receipts

- [RELEASE.json](RELEASE.json), SHA-256 `431668c46eeb76f8789acae8cccfdf7b2b23c06a923ac9b522e9084836d56848`: selected artifact identity, exploratory status, review scope, runtime limits, and nine receipt hashes.
- [Build receipt](geometry_adapter_r2_q16x_evalonly_build_receipt.json): package members, native library and model/source hashes.
- [Full F0 serving receipt](geometry_adapter_r2_q16x_f0_serving.json): official metric, complete exposed-F0 score, reference difference, and observed timing.
- [Python 3.11 smoke](geometry_adapter_r2_q16x_python311_smoke.json): exact-ZIP offline import, CRC/path, callback, finite-output and reset checks.
- [Submission alias receipt](geometry_adapter_r2_q16x_submission_alias.json): confirms the selected archive alias points to the tested bytes.
- [Legacy native-cache replay](geometry_adapter_r2_q16x_native_cache_parity.json) and [label correction](geometry_adapter_r2_q16x_native_cache_parity_label_correction.json), SHA-256 `6253bbe98c85cf41acb53cb507cf849b345b4be047c60b610ee614a074bf9ba2`: the legacy score field used the q16x cache as reference, not Torch FP32.
- [Corrected uint32 cache parity](geometry_adapter_r2_q16x_native_cache_uint32_parity.json), SHA-256 `1bce0debd60f266152ed7814f437bdaf28456aa0c9a984121c2c630901a89eb1`: exact bit-pattern and signed-zero check on the first eight groups.
- [FP32 native/Torch panel](geometry_adapter_r2_f32_native_torch_panel.json) and [predeclared protocol](geometry_adapter_r2_f32_native_torch_panel_protocol.json), with [tolerance assessment](geometry_adapter_r2_f32_native_torch_tolerance_assessment.json), SHA-256 `75a7a56d8e9d137329543993c21e7afe654d02573118a5fc5e0759096b3f82af`: preserve the failed tolerance and its scope.
- [Frozen-parent check](frozen_parent_check.json) and [adapter comparison](comparison.json): repeat the one-pair training result already indexed at the packet level.
- [Adapter result figure](adapter_result.png): visual summary of the negative adapter-versus-parent fit.
