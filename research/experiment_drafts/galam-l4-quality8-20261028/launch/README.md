# Authorized fixed eight-fit launch

**Full training is running on all eight L4s.**
[Execution status](EXECUTION_STATUS.json) and [owned process receipt](receiving/full-launch_receipt.json)
bind the fresh fits to launch `e635250b62fa489d94c987d4898ff417`, source R9
commit50053909 and training-source SHA `f9685716d64ecb5ceef2ec821781fa4078ff21661201edfcdac8292867990230`.
The [first complete GRU epoch](FIRST_EPOCH_RESULT.json) reports full-valid
WP0.6681328342; this is an intermediate single-seed result, not a final endpoint.
[Independent epoch metadata review](receiving/INDEPENDENT_FIRST_SCIENTIFIC_EPOCH_G0_REVIEW.json)
checks its coverage/pointers without rescoring model predictions.
The [first-epoch GitHub update](FIRST_EPOCH_UPDATE.md) keeps the distinct masks
and intermediate-result interpretation next to the reported scores.

All10325 retained train sequences fit;282 train sequences linked to validation
components are excluded. Full official valid keeps1873 sequences and3,528,273
scored rows. Normalization uses retained train members only. The protocol is
the user-adopted official train/valid file boundary; current calendar timestamps
remain unverified. No seq_ix ordering or random primary folds are introduced.

The [live dashboard](http://127.0.0.1:18766/) displays all eight actual runs via
the existing SSH forward. Training, the dashboard service and the node GitHub
publisher have detached ownership. The [first node update](https://github.com/VladKochetov007/WunderFunder/pull/46#issuecomment-6461962327)
was posted as SemyonEpanov; further hourly/per-experiment/final updates run on the
node independently of SSH. [Exact source and engineering evidence archives](ARTIFACTS.json)
preserve receiving bytes, successful profiles and the excluded startup failures.
Do not duplicate this active eight-fit launch.

The user explicitly authorized full execution of the allocation in
[v3](../L4_ALLOCATION_V3.md), without changing hypotheses. [PANEL.json](PANEL.json)
materializes exactly eight fits; [configs](configs/README.md) are complete runtime
recipes. Old v1 files remain untouched. Final batches are32 for plain GRU and384
for all seven GALAM cells, with one prefetched input batch and eval batch384.
The [current draft PR contract](PR.md) describes this full eight-fit scope.
[Draft PR46](https://github.com/VladKochetov007/WunderFunder/pull/46) was created
as SemyonEpanov; [publication receipt](PR_PUBLICATION.json) records its identity.

All use a common20-epoch maximum, strict FP32, fresh seed1001, official earlier
train fitting and complete official valid reporting. No pretrained initializer,
teacher, propensity model, KD or quantization is introduced. GRU211 differs from
raw112 GALAM as a whole-system benchmark. Shared warmup-cosine-v2 is an explicit
adaptation from the historical short OneCycle GRU recipe. All AdamW groups use
beta1.9/beta2.999/eps1e-8/wd.01, including hybrid non-Muon parameters; only the
declared matrix update rules differ in the hybrid arm. Muon momentum.95,5NS
steps, epsilon1e-7 and strict FP32 state/matrix work are explicit.

Keep every epoch snapshot plus strict-best/last pointers. Full official valid
WP determines the separately declared development selection; early stop has
min_epoch5/patience4/min_delta.0005. Strict-best improvement and the patience
threshold are distinct and recorded. Train-eval uses audited all-after-warmup
rows; valid uses official is_scored. These masks are explicitly labelled.

Engineering synthetic shape probes are discarded and provide no model quality.
Full scientific admission also requires actual-source profiles with diagnostics,
normalization/coverage verification, independent recipe review and matched
source/config/runtime receipts. All eight passed those gates in
[the independent resource review](receiving/INDEPENDENT_PROFILE_R4_ALL8_RESOURCE_REVIEW.json).
Receiver software is the checked isolated Torch2.8+cu128 environment on driver570.
Foreign services remain outside this run's ownership.

[prepare_data.py](prepare_data.py) is the bounded receiving-side preparation
owner: it waits for the transfer's finalized train filename, then runs the pinned
source audit, thin indexes, complete source-qualified component graph and primary
manifest. It prewarms train-only raw112/event211 normalization once. Each phase
has a durable receipt/log; it starts no scientific fit and never promotes a
partial file by logical size.

The [receiving engineering evidence](receiving/README.md) contains discarded
strict-FP32 shape probes and actual Torch2.8 CPU checks. GALAM's common batch384
passed the full-input real-source profiles, with one prefetched full input batch;
plain GRU retains historical batch32. Report-only evaluation uses batch384.
This changes optimization step count relative to the provisional GALAM16 start,
so final batch/step counts are recorded before fitting, without LR scaling.
Telemetry/diagnostics are sampled every25 GALAM updates and100 plain-GRU updates.
None of these engineering measurements is a trained-model quality result.

[Profile cohort owner](profile_cohort.py) waits for completed data preparation,
then starts exactly eight bounded, discard-only actual-source profiles with the
final config hashes, GPU and CPU affinity. It records owned process incarnations
and exits; it never approves itself or starts scientific fits.
[Monitor budget preflight](MONITOR_BUDGET_PREFLIGHT.json) bounds journals using
the complete10607-group train count as an upper bound. The largest projected
journal is317,390,848B, within the enforced1GiB cap, before any early stop.

The [fixed-eight launcher](../../../../scripts/launch_galam_quality8.py) and its
[operational components](../../../../scripts/galam_l4_quality8/) reconstruct the
profile-bound identity, verify the eight L4 UUIDs and current CUDA-usable memory,
then run a detached supervisor with owned PID/start-tick records. NVIDIA SMI
physical memory is recorded separately from CUDA-usable memory. Fresh run
directories belong to the trainer's exclusive-create operation. The supervisor
accepts the actual producer's declared early-stop result even on epoch20; it
never resumes or retries a scientific fit automatically. Eight mocked receiving
contract tests cover the corrected ownership/output/memory semantics.

The [node publisher](../../../../scripts/publish_galam_quality8.py) implements
the user's standing request for hourly and per-experiment GitHub updates toPR46
as SemyonEpanov. It checks exact source/split/run identities and full-valid masks,
records comment URLs and distinct retry-safe event keys, and has a96-hour bound.
Nine mocked checks cover identity and repeated hourly publication. It will run
independently of SSH after launch, using a private ephemeral credential outside
the artifact-serving directory and removing its exact credential inode on exit.
