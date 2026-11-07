All six fresh fits are active under the separately approved native S4 producer.
Native137 records controller2261449/birth2148815876 starting22:37:22UTC;
native141 records all six training at4530–4760 successful updates at22:42:47UTC
on7November. These receipts establish progress, not completed6E fits or a gain.
The immutable raw receipts and their hashes are retained in native_receipts/.

The bounded panel contains six fresh GRU256x2 fits, three arms with paired new
seeds101101/101102. Every arm uses6E, batch32, chunk1000, accumulation1,
AdamW3e-4/betas.9,.999/eps1e-8, decay.01, clip1 and warmup_cosine_v2 with5%
warmup. Each fit has38760 updates. Primary readout is fixed equal-parameter SWA
of epochs3/4/5/6; secondary is fixedE6. Each is graded standalone and as the
fixed0.9 clean-TL3 +0.1 candidate mixture. There is no validation-adaptive
checkpoint, LR, early stop or blend weight. Two seeds do not establish
significance, and repeatedly inspected validation remains exposed development.

| Arm | Input | GRU interlayer dropout | GPU assignment |
| --- | --- | ---: | --- |
| raw112_d0 | raw112 | 0 | 0,1 |
| raw112_d02 | raw112 | .2 | 2,3 |
| event211_d02 | raw112 plus99 causal event features | .2 | 4,5 |

Dropout is the existing nn.GRU interlayer dropout, disabled during evaluation.
Raw dropout arms have identical parameter initialization under each paired seed.
The event arm changes input width and therefore parameter shape/initialization;
it is a representation treatment rather than an identical-weight ablation.
Comparisons against old12E fits are historical, not isolated dropout effects.

The mechanism hypothesis is that regularization and the shorter frozen schedule
reduce the late degradation observed previously, while event clocks/deltas expose
causal state to the GRU. The matched new control distinguishes these treatments.
Failure to improve both paired fixed readouts or their fixed-mixture contribution
is negative or inconclusive evidence for this recipe, not permission to select a
different exposed epoch. All arm/seed/readout results and failures remain visible.

ROOT supplied the official split
0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Fit only10325 train sequences; fit-valid ancestry is zero. Evaluate all1873
valid sequences/3528273 scored rows. Qualified native audit130 reports train
has_is_scored=false and valid has_is_scored=true. Fitting explicitly uses all
rows after the99-row warmup; evaluation uses the official is_scored mask after
warmup. Every row advances recurrent and event state, including unscored rows.
No shuffled seq_ix order is interpreted as chronology.

Event211 is the existing causal transform of each complete raw sequence before
TBPTT, with a fresh event state per sequence. CPU checks cover full versus
streaming assembly across the1000-row boundary, reset, future perturbation,
Torch/NumPy agreement and real GRU evaluation prefix/chunk equivalence. Moments
read features alone from fitting members. Schema/feature width and exact source,
split and fitting membership bind the cache. Raw112 moments cannot satisfy the
211 identity. No complete feature cache is created.

Native130's16-sequence CPU timing sample took.832s and estimated537s for a full
pass, explicitly without publishing training moments. The actual canonical
full job found an already qualified cache and finished in.0093s. Its211 moments
identity is d6343f6077eb5a445aa3464ff28cef67435c09abc140eea8951f197fa65e35d4,
and moments SHA is8094dea7c59d4e8c1ae88bcde084ed77e8d50b45be7261e9078785ae826b04e2.

Actual fitting/profiling reuses the immutable previously qualified S4 capsule:
Rold/source-gru-monitor-resume-r1, canonical source SHA
5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0.
Rold=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1.
NewW canonical source also imports the proven prefix diagnostics and resume
acknowledgment fix, but its additional post-fit modules are a separately
admitted readout source, not a relabelled fitting producer. The native old
source/capsules/results remain unchanged. No old weights initialize these fits.

The first actual new profile clock is1794090445.7631443. Fitting ends no later
than1794101245.7631443; the entire wave ends1794104845.7631443, reserving the last
hour for readouts/collection. No restart extends that scalar clock. Limits are
six fresh fits, no automatic retry,4,000,000,000 new output bytes,3GB free floors
on data/data2,128GiB owned RSS and64GiB host available-memory headroom. Native
workers use8 CPU threads each. Foreign GPU4 services are preserved.

ROOT134 reports all four raw profiles passed30 successful updates with no
errors, actual elapsed3.73–3.82s and peak reserved1.596/1.663GB. ROOT136 reports
both event profiles passed30 updates in4.864/4.876s, peak2.143GB. These are
engineering receipts, not completed fits or a quality gain. The receiver adopts
their exact native profile paths; it does not repeat them. The event profiles
used the qualified211 normalization receipt. Every actual profile/config/
source/runtime/GPU identity must match a separate ROOT-approved copy before fit.

Stage the thin receiving modules together outside the training capsule:
run_wave.py, wave_checks.py, wave_execution.py, wave_postfit.py and
observed_worker.py. They reuse canonical run_owned_command, PID/birth ownership,
the existing observer and the strict canonical trainer; there is no trainer
copy. The latched resource/cancellation callback stops only its exact owned
group with zero grace and returns normally, avoiding the legacy executor's
long callback-exception grace path. ROOT TERM/INT cancellation follows that path.

The following records the fit launch interface. ROOT137 has already launched
all six fits; do not issue duplicate launches or alter the frozen live Q files.

```sh
R=/data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1
S=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1
py=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q=$R/control/l4-wave-r1
# ROOT approval copies: Q/approved/<cell>.json, preserving all native fields.
PYTHONPATH=$S/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$py" "$Q/run_wave.py" --runtime "$R" --python "$py" --phase fit \
  --arms raw112_d0,raw112_d02
PYTHONPATH=$S/src "$py" "$Q/run_wave.py" --runtime "$R" --python "$py" \
  --phase profile --arms event211_d02 \
  --event-normalization-receipt "$R/receipts/event211-allfit-r1.json"
# After separate ROOT approval of the two event profiles:
PYTHONPATH=$S/src "$py" "$Q/run_wave.py" --runtime "$R" --python "$py" \
  --phase fit --arms event211_d02
```

The runtime/project/runs placement and ROOT-preparation-authority-r1.json are
already supplied by ROOT. Configs are the exact six frozen JSON files under
R/configs; raw and normalized hashes are in CONFIG_INVENTORY_R1.json. Fresh-fit
claims prevent duplicate admission. Each fit saves every epoch's weights,
whole-epoch model/optimizer/scheduler/RNG state and six full-valid journals.
Fit completion requires terminal completed6E/38760, matching actual model.pt
bytes/identity and exact six-epoch validation coverage. No parameter-only resume
is introduced.

Post-fit execution uses the NEW separate run_readouts.py revision and namespace
R/control/l4-readouts-r1. The live fit Q remains immutable; its older optional
readout route is historical and must not be used for this admission. Stage
run_readouts.py with unchanged wave_checks.py and wave_execution.py, and qualify
a separate full postfit source capsule. ROOT supplies ROOT-admission.json with
approved=true, reviewer=ROOT, source, source_sha256, reference_root and
reference_manifest_sha256. The source hash must be computed from the actual
staged capsule by canonical source_identity; the fitting producer stays5bbbf68e.
The runner waits for exact completed6E/38760 receipts, then independently uses
spare GPUs6/7 with at most two readout jobs. It claims each cell once, keeps the
original four-hour deadline, and launches zero new fits. ROOT can start it as:

```sh
Qreadout=$R/control/l4-readouts-r1
P=/path/to/ROOT-admitted-postfit-capsule
PYTHONPATH=$P/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$py" "$Qreadout/run_readouts.py" --runtime "$R" --python "$py" \
  --admission "$Qreadout/ROOT-admission.json"
```

Each completed cell builds fixed SWA from the
pinned complete6E index, then reads swa3456 and model.pt/epoch6. Equal quarter-
scaled FP32 parameter averaging preserves every named buffer byte-for-byte,
including fitting-only mean/std; all four parents must agree. Parent hashes,
seed/config/data/training source and readout implementation source are recorded.
SWA has a distinct inference-only schema and carries no optimizer/RNG resume
state. PyTorch's [averaging documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.swa_utils.AveragedModel.html)
likewise makes parameter/buffer handling explicit; no BatchNorm recalibration
or normalizer recomputation is used for these GRUs.

ROOT131 transferred the original sequence-reference-cache-v1 evaluation bank
unchanged to R/reference/evaluation, manifest
1863e5e6f844c8f172f3fd54879ff908e0e415183766c354482b1ac9c5bfd985.
The accepted ReferenceCache reader validates the known17-role producer, complete
membership and exact array bytes; no alternate reference format is invented.
Historical published-cache ancestry limitations remain explicit. The reference
must reproduce WP.6923774248169172 during grading.

Each readout forwards every row, then uses the unchanged canonical official
GlobalAccumulator for standalone and fixed-mixture WP, plus monitor-compatible
WMSE. Composition is raw FP32 0.9reference+0.1candidate before official clipping.
Excluded nonfinite values are sanitized before WMSE arithmetic. Compact outputs
retain raw FP32 predictions at scored rows with original int32 row indices and
source/group hashes, about42MB/readout instead of300MB/full-row exports. Thus
all twelve readout prediction artifacts fit comfortably inside the4GB cap.
The24 fixed score readouts are additional to36 exposed per-epoch valid looks;
reused reference values are not additional training evidence.

CPU verification so far: nine model/feature/normalization/SWA fixtures passed,
five receiver lifecycle/config tests passed, and the excluded-NaN WMSE regression
passed. Independent critic ran the five receiver tests and found no fit-path
blocker, then independently passed five focused postfit tests and reviewed the
separate readout runner without a static blocker. Model/CPU tests were serialized with the other actor under two aggregate
threads. Native profile/fit/grade execution and independent source admission are
ROOT-owned; actor preparation/tests do not promote any scientific result.
