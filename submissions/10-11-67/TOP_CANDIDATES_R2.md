# Current strongest provisional submission candidates

Updated 2026-10-12 Europe/Kyiv. Human uploads; no public upload was made by the
controller. Original frozen ZIPs and READY.json are preserved.

| Priority | ZIP | What is blended | Size |
|---|---|---|---:|
|1|`12_h388_teacher_trio_all_fp16store_q8vnni.zip`|Three independent full-data H384 Muon-all teachers, seeds3/4/5 CP2, equal thirds|13.735 MB|
|2|`11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip`|Full-data H256 distilled student CP5 and H384 teacher CP2, equal halves|12.914 MB|
|Fallback|`10_h388_teacher_s4_s5_all_q8vnni.zip`|Two full-data H384 teachers CP2, equal halves|17.272 MB|

The trio is first because its corresponding fixed fold-1 teacher recipe benefits
from three independent seeds and has the closest recipe correspondence to its
full-data members. Its exposed fold-1 float reference is .6725416, versus
.6723361 for the teacher pair. These are discovery scores before the exact new
native-precision screen, not validation scores of full-data checkpoints.

The compact alternative uses a different family. Its related fold-1 prototype
scores .6725743, but the all-data student's initializer and since-weight differ;
that number must not be assigned to the final ZIP. All three final packages fit
12,479 supplied train+valid groups and have no held-out final-checkpoint score.

Every listed ZIP has independent Python3.11 exact-package Q8VNNI, finite-output,
ABI and deterministic reset checks. The trio's selective FP16 disk storage keeps
input projection, normalizers, biases and final head in FP32; inference still
uses the existing Q8VNNI backend. On ten complete in-sample sequences, its
matched transport rounding costs .00000563 WP. This verifies fidelity, not
generalization. Its local 400,000-callback review took59.58s with248MB cgroup
peak. Competition-container wall time is not certified by that host measurement.

The student-plus-three-teacher four-member endpoint is preserved for research,
but is not a current top nomination: conditional advantage is unconfirmed and
its in-sample diagnostic was weaker. Q4 is likewise not preferred from its
single measured screen; a materially different larger mixed-bit blend can still
earn inclusion by its complete endpoint score.

Expected public performance is uncalibrated. The trio is our leading new attempt
to improve the user-reported .6668 best; modest upside and material no-gain risk
remain. **A public .670+ expectation is not established.** These are our strongest
new provisional candidates, not a claim of competition-wide SOTA. Local scores
are never added to an old public score.

Exact byte hashes, roles and review links are in
`READY_TOP_CANDIDATES_R2.json`. The native41-group/14-complete-component fold-1
screen and clean fold-0 teacher/student pipeline continue in parallel.
