# Optimized B300 receiver, outside frozen S3 and accepted Q

The user's7 November optimization request and ROOT's new bounded allocation
authorize four newly identified optimized scientific fits after retiring the
four incomplete B4 fits. Retired partial fits remain evidence and are never
resumed or counted complete. ROOT alone stages/launches GPU work. This helper
prepares the exact same four arms/seeds on the selected batch128 or256 per rank,
world2/A4, T768/L24/context1024/chunk256, fixed30epochs and unchanged S3 source.
Batch changes optimizer update/schedule/weight-decay exposure explicitly; these
are new recipes. No LR scaling, objective, split, labels or precision changes.

Select one prepared folder after the actual R2 engineering results:

| Folder | Batch/rank | Updates/epoch | Updates/30E | Panel SHA256 |
|---|---:|---:|---:|---|
| prepared-b128 |128|820|24,600|78ac20e1fb651c314ae38ded8728fb0826db65823742d47fff2e6b80e9b438f0|
| prepared-b256 |256|420|12,600|3e6b55bbf7a930da6eb32ca7207ce4155cdc9a0449e61e8218be6fd0ebc73235|

Copy only the chosen folder's contents to `R/throughput-engineering-r1/production`.
Config paths are independently frozen per cell in panel.json. Any config differs
from its original S3 recipe only in batch_size; hashes and full JSON equality
are checked before admission. All four cells must use one selected batch.
To rebuild locally use `optimized_plan.py --batch128|256 --output FRESH_FOLDER`;
on the receiver also supply `--base-config-root "$S3/research/experiment_drafts/tl3-transformer-complement-execution-20261107/configs"`.
Use a space between `--batch` and its value.

Stage optimized_controller.py, optimized_plan.py and phase.py separately at
`R/receiver-b300-opt-r1`. Keep the accepted `R/receiver-b300-r1` read-only. The
new controller imports its Controller resource/ownership methods and uses its
unchanged native-tested subreaper relay. It checks the exact CDB/AF03/688 core
digests and imported module path. The controller imports known support helpers;
every model child resets PYTHONPATH to S3/src. The optional Muon S4 source is not
part of this receiver or either config.

```bash
R=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1
PYTHON=/data/epanov_s/tabdata/.venv/bin/python
Q="$R/receiver-b300-r1"
O="$R/receiver-b300-opt-r1"
P="$R/throughput-engineering-r1/production/panel.json"
PANEL_SHA=<chosen-table-value>
PYTHONPATH="$Q" "$PYTHON" "$O/optimized_controller.py" --python "$PYTHON" \
  --receiver "$Q" --panel "$P" --panel-sha256 "$PANEL_SHA" --detach
```

Before profiles the controller verifies existing native S3 acceptance, unchanged
data-check, split/source manifest, complete reference preparation and exact
full-valid TL3 baseline; a direct canonical capsule CLI checks S3 bytes again.
It copies the unchanged qualified baseline into the new run root for canonical
comparison. Fresh outputs are `R/throughput-engineering-r1/runs/<cell>`.
The pairs are labels101001=0,1; complement101001=2,3; labels101002=4,5;
complement101002=6,7. The immutable implementation/attempt/owner/resource records
are under `R/control/b300-optimized-r1`.

Each cell independently runs one50-successful-update canonical real-data profile
on its actual seed/config/physical pair, with a900-second bound. Then it waits
for ROOT's unchanged copy at `<cell>/approved_profile.json`: native
`<cell>/profile/profile_receipt.json` fields unchanged except approved=true,
plus reviewer="ROOT" and actual independent review details. The source's fit
admission rechecks exact complete identity. A ready cell runs without waiting
for other cells: fit -> evaluate -> bundle. No automatic retry or resume exists.
Four profiles are scheduled; the explicit cap is8, leaving any additional
production/peak diagnostics under separate ROOT admission rather than an
automatic queue. A failed profile/fit is terminal for that cell.

The authority starts at the first production profile:24hours total, final1hour
collection reserve, fitcap4/profilecap8. Fit deadlines use remaining pre-reserve
time; profile900seconds and source/evaluate/bundle/compare1800seconds are also
bounded by the campaign deadline. Observer max86400 is within its native cap.
Existing ten-second resource guards enforce1,024GiB receiving RSS,200GB new
outputs and200GB free/data, with exact owned descendant cleanup only. Compiler
settings and BF16/BF16x9 remain S3's original values; short TMPDIR avoids AF_UNIX
limits. No7/9-day continuation is needed or admitted here.

The terminal gate requires completed30epochs and the selected derived count,
never774,600. Evaluation reuses the original canonical predict_sequence.py
reference composition alpha0.1, then score_sequence.py writes fit/score.json.
S3's full1873/3528273 per-epoch monitor remains report-only. Bundles retain final
weights/state, journals, paired metrics, exact profile/approval and prediction
manifest. Canonical compare_panel.py runs only after all four finish and uses
the new config root and homogeneous-device UUID amendment explicitly.

Preparation creates no GPU result or quality claim. CPU checks cover derived
counts, exact CLI paths/reference separation, unchanged ROOT approval and failed
phase short-circuiting. The existing native relay proof remains the lifecycle
basis; no relay code changes occur. Actual production profiles, ROOT review,
full training/validation and collection remain receiving gates.

[Independent review R1](INDEPENDENT_REVIEW_R1.md) binds the three implementation
hashes, both panel hashes, exact config-only batch changes and five independently
passing CPU gates. Its readiness finding applies to production profiles;
ROOT retains every cell's later scientific fit admission.
