# Fixed-weight CPU-policy probes

[October15 competition results](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/PUBLIC_CPU_POLICY_RESULT_20261015_R1.md): user reports D47m27s/651MB/0.6681, C57m27s/656MB/0.6681 and B TIMEOUT61m57s/650MB. Historical A50m16s/650MB/0.6681. Prefer D provisionally (169s/5.60% less time), retain A fallback; same-day A–D–A would help assess host variation after explicit resume. Structured receipts preserve all supplied rows and expected local ZIP hashes, not portal-verified payload hashes. Research remains paused.

[October15 handoff verification](HANDOFF_FILE_VERIFICATION_20261015_R1.json) rechecks all four current ZIP sizes, full SHA256 values and archive CRCs: PASS. This verifies the delivered files separately from the earlier prediction/ABI review. Portal uploads and target-host timings remain pending; local research continues independently.

These checked packages implement the requested original-xteam_h4 engine comparison. They preserve all four original weights, member order,25% coefficients, preprocessing and Q8 representations. They are runtime probes, not newly selected student architectures. Uploads are user-owned and asynchronous; local research continues independently.

Use the available team slots in this order, carrying the sequence across days if necessary:

| Order | ZIP | Policy | Bytes | SHA256 prefix |
|---:|---|---|---:|---|
|1|[A: original xteam_h4](../../artifacts/team_models_20261013/xteam_h4.zip)|Shipped512-bit VNNI|16,308,160|f1635d14c0e1|
|2|[B](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/packages/xteam_h4_B_q8vnni256.zip)|Genuine EVEX256 VNNI|16,464,825|5aed417e57bd|
|3|[C](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/packages/xteam_h4_C_q8vnni_blocked.zip)|Alternative packed/register scheduling|16,464,830|c17c6415728e|
|4|[D](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/packages/xteam_h4_D_q8vnni_fused.zip)|Fused GRU projections, separate scales|16,464,828|e11ff6a3232c|
|5|[A: repeat original](../../artifacts/team_models_20261013/xteam_h4.zip)|Same ZIP as order1|16,308,160|f1635d14c0e1|

Full hashes and build provenance are in the [package manifest](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/PACKAGE_MANIFEST_R1.json). The rebuilt-A ZIP is an internal numerical-equivalence check and is excluded from submission order. Later useful comparisons reverse candidate order and repeat controls; one faster target-host run remains provisional.

[Independent review](../../research/unified/reviews/CPU_POLICY_IMPLEMENTATION_REVIEW_20261014_R1.md) verifies original/member/library identity, CRC, strict selected-policy assertions, actual EVEX256 opcodes, and bitwise prediction/reset parity on two complete20,000-row sequences under Python3.11. Six callback variants include originalA, rebuiltA and finalA repeat. The [ELF receipt](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/ELF_ABI_R1.json) finds a maximum GLIBC2.34 requirement, compatible with the documented Bookworm2.36 baseline. B/C/D deliberately fail on unsupported CPUs so a successful probe cannot silently measure a fallback.

The [local Intel report](../../research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/REPORT.md) favors originalA in short measurements. It is not an AMD or competition-runtime certificate. No new public score, quality improvement or upload is claimed.

When results arrive, append the ZIP hash, policy, portal receipt, score, duration, memory and any failure detail to the campaign's submission receipts. Controls estimate host variation. Public probe scores must not tune member weights or select architectures. The [complete research plan](../../research/campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md) and [live frontier](../../research/campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md) own the independent student and theory work.
