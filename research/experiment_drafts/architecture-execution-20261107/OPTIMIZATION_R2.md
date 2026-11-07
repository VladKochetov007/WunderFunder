# Measured cluster optimization and online review, 7 November 2026

This dated amendment supersedes the live-status and timing claims in frozen
[EXECUTION_R1.md](EXECUTION_R1.md). The user requested optimization on the existing
eight B300 and eight L4 GPUs. No additional architecture or seed is admitted.
ROOT retains the single persistent file-RPC transport and GPU admission authority
per host. Foreign services, old sources and historical results are preserved.

B300 now runs four new matched B128/A4/world2 configurations to fixed30E/24600
updates. The original B4 attempts were retired at1291/1269/1264/1263 updates and
remain partial historical evidence. The measured engineering throughput changed
from about11k to217k forwarded rows/s; this does not establish quality parity.
B256 added only8.6% throughput while doubling memory and halving update count.
The exact native receiving/profile/admission record is in the separate
[B300 optimization packet](https://github.com/VladKochetov007/WunderFunder/blob/experiment/pr67-execution-20261107/research/experiment_drafts/tl3-transformer-complement-execution-20261107/OPTIMIZATION_R1.md)
and [PR69](https://github.com/VladKochetov007/WunderFunder/pull/69).

The first complete B300 epoch used820 updates, approximately20 minutes training
and6.4 minutes paired evaluation. The earlier8h estimate is core fitting only;
first-cycle arithmetic is approximately13h including recurring evaluation, before
remaining startup/export overhead. This is a planning estimate. The optimized
controller has its own finite24h receiving envelope and1h collection reserve;
the initial B4 nine-day amendment is historical, not the current allocation.

At20:02UTC all four optimized B300 fits were near1.9/30E with fresh heartbeats.
The first complete valid results of the fixed90%cleanTL3+10%Transformer output:

| Objective | Seed | Epoch1 valid WP |
|---|---:|---:|
|labels |101001 |0.6923690015420718 |
|labels |101002 |0.6922338596770409 |
|complement |101001 |0.6896017033442927 |
|complement |101002 |0.6911783882637030 |

Reference cleanTL3 is0.6923774248169172. These are mixture scores, not standalone
Transformer scores. No improvement is established. Complement loss is rescaled
and cannot be compared numerically with labels-only training loss.

Second complete epoch: labels1010010.692246309, labels1010020.692508503;
complement1010010.691693153, complement1010020.691902804. One labels seed is
slightly above the reference and one slightly below. Averaging the two scalar
scores would not establish an ensemble score or robust improvement.

L4 retains eight original12E/77520-update fits. All were live at20:02UTC:
plain GRU around7.8E, local-attention GRU7E, Mamba2 around10E and EMA/custom
TabM-style around11.6E. Mamba2 is genuine pinned Mamba-2; the custom readout is
not official TabM and no model is an exact GALAM reconstruction.

Actual plots were inspected, including Mamba2 seed101001 through epoch9. Its
train-eval WMSE fell1.2604→1.1633 from epoch6→9 while valid WMSE rose
0.9747→0.9972 and valid WP fell0.680358→0.670524. Seed101002 fell from its
best-observed epoch6 WP0.682011 to0.667586 at epoch10. This is evidence of
overfitting, not an execution failure. Plain/local GRU latest WP was approximately
0.680–0.683; EMA approximately0.671–0.673. Fixed final endpoints remain unchanged.
Observed peaks are descriptive development-exposed checkpoints, not new clean
selection evidence. Train-eval has a different cohort/mask from official valid;
absolute train/valid score gaps alone must not be interpreted as overfitting.

ROOT identified expensive GRU diagnostic replays as an L4 bottleneck. The
[receiving packet](l4/throughput_probe_r1/CONTINUATION_S3.md) preserves exactly the
existing two GRU trajectories, optimizer/scheduler/RNG and12E endpoint. Its
bounded diagnostic prefix and external full-state archive alter monitoring and
retention mechanics only. Two actual20-update seed/GPU profiles passed; logged
loss and preclip gradient match the original at steps10/20 exactly. The native
supervisor was detached20:03:45UTC as PID2230567/birth2147894167, plan SHA
dfc7a770c5b5ac14e0b09edc3395d50aae7fdcf10924130966dd961104f55a83.
Both read-only full-state preflights passed at next_epoch7 before any old stop.
At that observation the supervisor awaited a later whole-epoch checkpoint;
arming is not proof that migration or resumed steps have occurred. Original
deadline1794154066.717658 is unchanged. No new fitting seed or scientific fit.

Subsequent20:06–20:08UTC observation found a real continuation failure. Both
original GRUs stopped at a complete epoch8/51680 checkpoint and migrated with
exact state proofs; external archive readiness passed. Canonical load_resume then
failed before new updates because monitor.restore_offsets returns None while
coordinated_call requires a value. The initial CPU preflight exercised the state
loader but missed this full canonical path. ROOT stopped only the exact owned
supervisor PID2230567/birth2147894167, preserving original and S3 checkpoints.
The two original GRU trajectories are paused, not completed; other ten fits are
unaffected. S4 repair is explicitly within the same continuation authorization:
a narrow nonnumerical resume-return fix, new immutable source/profile identities,
actual canonical CPU resume admission, fresh sibling outputs and original
deadline. The failed S3 attempt and all old sources remain historical. Do not
claim S4 active until native resumed updates are observed.

S4 was built and verified natively: inventory28ac81b46517f1923d4ffd9751bd0d615f2657c5721d2e3459726e8c05ef4182,
producer source5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0.
It differs from S3 only in sequence_fit_resume.py and from S1 in exactly four
files. The callback now explicitly acknowledges successful monitor restoration;
real exceptions still propagate. Both new20-update profiles passed, with exact
original logged loss/gradient at steps10/20 and stable original E8 state hashes.
The R2 canonical CPU probe then reached a successful actual load_resume but its
own comparison rejected plain dict versus the saved OrderedDict before comparing
tensors. That failed test is preserved. R3 corrects only this test projection,
retains strict container/key/dtype/shape/stride/byte checks, and uses fresh
control/output paths; model source and approved S4 profiles are unchanged.
Independent six-test review includes a real OrderedDict and rejects a mutated
tensor. Native CPU then real CUDA zero-update restoration remains a hard gate
before the same original fits may continue. R3 arming alone is not resume proof.

Actual R3 receiving passed at20:26UTC: both real canonical CPU restores passed
(CPU-only CUDA setter explicitly stubbed), then both real CUDA restores passed
with no stub, no forward and zero optimizer updates. Each verifies model,
optimizer, scheduler, RNG, safe offsets, parent checkpoint and actual source/
configuration/runtime. Both external archivers became ready before production.
The two original trajectories then resumed at20:26:44UTC and reached51750
successful updates by20:26:51, beyond the original51680 frontier. Current output
siblings are runs/gru_s101001_monitor_opt_r3/fit and
runs/gru_s101002_monitor_opt_r3/fit. Detached controller2238454/birth2148028384,
plan58eb238d83144937555fa839993592892b40c5096940eef706474fedc480df82.
No model source/profile/recipe/seed/deadline change followed S4 qualification.
The corrected callback guard also avoids the old helper's long TERM grace on
progress-callback failures; only exact new owned groups are stopped and reaped.

Two EMA/custom TabM-style fits completed their original12E/77520 endpoints around
20:08UTC, WP0.672695/0.671266. They are completed models, not successful promotion
over TL3. The remaining local/Mamba fits continue; frozen checkpoint histories
and final metric artifacts remain distinct from artifact upload/verification.

The local [dashboard](http://127.0.0.1:8769/) reads actual native journals and
displays train-eval/full-valid loss and score, both targets, LR, preclip gradient,
clipping incidence, useful row throughput, phase/progress ages and host/GPU
telemetry. It retains real PNG/SVG epoch snapshots under the local execution
artifact directory. Native5s heartbeats and10s resource observations are mirrored
every15s; occasional delivery jitter is distinct from native progress failure.
Clipping is frequent in the current tail samples; finite norms and live progress
do not imply that this optimizer setting is optimal. No live numerical recipe
was silently changed based on these exposed validation curves.

Primary fitting remains10325 official train members, zero valid fitting; every
primary epoch evaluation covers all1873 valid sequences and3528273 scored rows.
Split SHA0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Normalizers/initializers/reference ancestry remain train-only. No public-score
forecast, statistical significance, final winner or completed final endpoint is
claimed in this progress record. Automatic progress publication targets
[PR68](https://github.com/VladKochetov007/WunderFunder/pull/68) and PR69 under
SemyonEpanov every30minutes or terminal change; upload/verification is a separate
stage from fit completion.
