# Final full-corpus TL3/LSTM submission, 12 November 2026

The user explicitly authorizes choosing the strongest supported submission, using necessary resources, fitting on all supplied labels and uploading to Connectome. This packet reuses the 18 completed final fits instead of repeating their teacher/student DAG. It is a submission trial, not a new clean validation improvement.

The fixed candidate is three-member TL3 (W18 / tiedlead seed5 / tiedlead seed6, weights 0.34/0.33/0.33), then 90% TL3 + 10% Source4 LSTM192x1. The all-labeled membership has 10,607 train and 1,873 valid whole sequences. No independent validation exists after this final fit; no final-fit score or checkpoint/weight tuning on valid is permitted.

Historical evidence: the signed-in portal snapshot published in PR76 on November12 reports incumbent TL3 public0.66832 and the November4 full-corpus W18 singleton0.66453. The historical exposed-validation fixed90/10 panel reports0.69250206 versus TL30.69237742; the small gain and incomplete historical cache ancestry do not establish significance or public improvement. A new all-data since32 fit is not required to unblock this frozen trial. The public score of the delivered ZIP must be observed separately.

## Frozen artifacts and independent decision

- [Current authority](AUTHORITY_R1.json) records fresh execution/upload authority, resources and primary-versus-final-fit policy. Old wave clocks are not reused.
- [Frozen recipe](FROZEN_RECIPE_R1.json) binds all four fixed endpoints and FP64-to-FP32 nested arithmetic; no new scientific fits are planned.
- [Independent decision](INDEPENDENT_RELEASE_DECISION_R1.md) conditionally admits one public trial after exact-ZIP engineering gates, without an improvement claim.
- [17-fit intake](FINAL17_ARCHIVE_INTAKE_R1.json) and its [producer manifest](FINAL17_ARCHIVE_MANIFEST_R1.json) verify all17 checkpoint hashes and exit-zero fit receipts against the published archive, with12,480 fitting members and zero independent evaluation.
- [W18 intake](W18_EVIDENCE_INTAKE_R1.json) recovers the missing eighteenth checkpoint from the complete authenticated13-role evidence release. The older truncated local ZIP is preserved and not used.
- [Export attempt](ROOT_R2_BUILD_ATTEMPT_01.json) records successful R2 build after the strict Source4 training-counter compatibility repair; original weights/checkpoints were not modified. The14,804,012-byte ZIP SHA is `a0fb8b1e254dd8be94301f6553f11b9e1b1ae758c9543f38aae3f021c944894f`. Runtime admission remains pending.
- [Private fallback resource](HF_CPU_SPACE_CREATION_R1.json) records the newly created free account-owned receiver. No GPU calls were requested. Both native SSH channels later recovered, so execution returns to the owned B300 CPU host.

## Engineering acceptance and status

Exact delivered ZIP checks cover Linux/Python3.11, byte identities, FP32 reference and quantization drift, deterministic repeat, sequence reset, causal prefix invariance, the20MB ZIP and16GB memory constraints, and measured one-core time against the platform60-minute envelope. Reading features from fitted valid for engineering is not a quality evaluation. No labels or official score are needed in these final-package checks.

Current state: built, **not admitted for CPU runtime, not uploaded**. The baseline callback takes3.43seconds per20,000-row sequence on one Xeon6767P core. A linear projection to the documented1,970 sequences is112.6minutes, above the60-minute envelope; this is a host extrapolation, not an observed platform duration. The cache-order/prefetch patch passed actual member/callback bitwise qualification and reduces the paired measurement to2.663seconds, still approximately87.4minutes on the same projection. See [results and limitations](RESULTS_R1.md).

The Windows desktop is unavailable: the authenticated admin session is disconnected, and a normal attempt to transfer that session to the console returned Windows error5, Access denied. Upload authority is already explicit. An accessible interactive session and CPU admission remain technical dependencies. Record the actual platform submission ID, filename/SHA, acceptance and public score before claiming completion. Do not duplicate the previously submitted W18 ZIP or silently promote a failed check.

Original artifacts remain in PR54 / its W18 release and PR76; source checkout for export is `6493556e853e02538d50baed049073eeb717e9d7`, with an explicitly pinned consumer-only compatibility overlay. New source/check receipts will be added here once frozen.

## Reproduction and evidence map

- [Exporter](exporter/README.md) contains the archive-to-ZIP builder, Source4 compatibility overlay, strict history-counter test and repair receipt. [Build evidence](build/README.md) binds the exact delivered bytes and final model ancestry.
- [Engineering sources](engineering/README.md) contain the feature-only offline checker, native parity checker, component/mode diagnostics and math-preserving Rust patches. R1 parity-checker failure is retained; R2 repairs only the misnamed parsed argument.
- [Measured receipts](evidence/README.md) include actual baseline qualification, drift, timings, component diagnostics, native builds and rejected kernel variants. No new scientific score was computed from all-labeled checkpoints.
- [Independent packager review](reviews/INDEPENDENT_PACKER_REVIEW_R2.md) checks the narrow compatibility repair and distinguishes verified four-endpoint identities from externally retained transitive teacher-cache evidence.
- [Draft PR contract](PR.md) records H-FINAL-TL3-LSTM-20261112 / E-FINAL-TL3-LSTM-20261112, the frozen endpoint and release conditions. Its publication is not a completed submission.

[Packet file inventory](PACKET_FILES_R1.json) binds the initial published evidence/source bytes; later evidence will be added as successor receipts.

The scoped `.gitattributes` preserves imported source/receipt bytes so Git line-ending normalization cannot invalidate their SHA256 pins.

## Later actual upload,13 November2026

[The user-directed platform trial](submission-attempt-20261113-r1/README.md) was uploaded once despite the host-runtime projection. The signed-in platform confirmed upload success and a new PENDING record for the exact14.8MB R2 ZIP at14:53,1/5 today and109 total. Earlier not-uploaded/Desktop-unavailable observations above are historical. Checker outcome, actual runtime and public score are pending; this upload does not turn the runtime projection into a passed gate.

[Public status check,15 November](status-20261115-r1/README.md): official public API now reports team best0.668579656155, rank62 and117 submissions. This does not identify the result of1Y4RW907. The submission-history API requires authentication; its latest specific outcome remains unverified.
