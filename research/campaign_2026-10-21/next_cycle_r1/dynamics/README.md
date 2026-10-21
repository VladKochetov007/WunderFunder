# Future-input dynamics execution packet

This campaign folder holds the finite plans and paired grader for
`E-FUTURE-INPUT-DYNAMICS-20261021-R1`. The implementation and scientific
contract live in the [experiment draft](../../../experiment_drafts/dynamics-aux-20261021/README.md).
Root owns all profile and fit execution and the three-GPU-hour stage admission.
Nothing here has been run; the existing CPU and static checks are implementation
checks only.

`profile_f0_s1001_aux005_r1.json` at the draft root specifies a sustained
110-batch profile of the exact F0/seed1001 weight-0.05 treatment recipe: ten
warm-up B8 batches and one hundred measured B8 batches, each with ten TBPTT
windows of 2,000 rows. The profile deliberately omits `--save-training-state`;
the fit plans include it. Set `WNN_LOADER_WORKERS=0` in the admitted worker
environment to prevent automatic worker creation. `kaggle_sequence.py` records
the full subprocess wall time, which includes selected-group target-statistics
scanning, input normalization and loader setup before the 110 batches. Keep
this full wall time alongside the trainer's per-batch `PROFILE` output when
estimating panel cost; the per-batch timer excludes the setup scan.

The four files in
[`recipes/`](../../../experiment_drafts/dynamics-aux-20261021/recipes/) each
contain one fixed CP2 fit, one feature-only complete-fold prediction export,
and one official `score_run.py` invocation. Run only after profile review and
root admission. From repository root, the exact finite invocations are:

```sh
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_LOADER_WORKERS=0 scripts/limited .venv/bin/python scripts/kaggle_sequence.py --plan research/experiment_drafts/dynamics-aux-20261021/profile_f0_s1001_aux005_r1.json
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_LOADER_WORKERS=0 scripts/limited .venv/bin/python scripts/kaggle_sequence.py --plan research/experiment_drafts/dynamics-aux-20261021/recipes/dynamics_f0_s1001_control_r1.json
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_LOADER_WORKERS=0 scripts/limited .venv/bin/python scripts/kaggle_sequence.py --plan research/experiment_drafts/dynamics-aux-20261021/recipes/dynamics_f0_s1001_aux005_r1.json
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_LOADER_WORKERS=0 scripts/limited .venv/bin/python scripts/kaggle_sequence.py --plan research/experiment_drafts/dynamics-aux-20261021/recipes/dynamics_f2_s1002_control_r1.json
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_LOADER_WORKERS=0 scripts/limited .venv/bin/python scripts/kaggle_sequence.py --plan research/experiment_drafts/dynamics-aux-20261021/recipes/dynamics_f2_s1002_aux005_r1.json
```

The frozen memberships imply 154 F0 fit batches and 161 F2 fit batches per
epoch, with ten 2,000-row optimizer windows per batch. Across the four runs,
the planned fitting work is 1,890 B8 batches/18,900 optimizer updates plus four
complete fit-only target-statistics scans. The separate profile adds 110 B8
batches and one such scan. These are schedule counts, not measured throughput;
use the actual profile wrapper wall, memory and thermal evidence before
admission.

After all four full-fold predictions finish, run the grader once with a new
output path:

```sh
env WNN_MEM_MAX=4700M WNN_CPU_QUOTA=100% WNN_THREADS=1 scripts/limited .venv/bin/python research/campaign_2026-10-21/next_cycle_r1/dynamics/grade_dynamics_panel.py --output research/campaign_2026-10-21/next_cycle_r1/dynamics/runtime/dynamics_panel_grade_r1.json
```

The grader reuses the official prediction-bank scorer and component bootstrap.
It checks complete NPY headers and payload lengths, exact evaluation group
order/coverage and sidecars, fixed fit/update/checkpoint settings, matching
initial parameter and input-normalization identities, exact fit-only target
membership/statistics hashes, and saved auxiliary-head identities. It reports
each arm and the fixed `0.9*control + 0.1*aux005` mixture on the official metric,
both targets, four position bins, every paired delta, and component-bootstrap
intervals. It does no coefficient search. Fold exposure and conditional
uncertainty are recorded in the output. The prediction-side arrays cover 468 F0
and 443 F2 evaluation groups; all public-transfer, seed and cross-fold claims
remain outside the evidence this two-cell screen can support.

The fixed grader is organized as [`grade_dynamics_panel.py`](grade_dynamics_panel.py)
for official scores and paired uncertainty plus
[`dynamics_panel_artifacts.py`](dynamics_panel_artifacts.py) for checkpoint,
target-statistics, prediction-identity and NPY-header checks. Both are grading
code only; neither contains or copies training orchestration.

## Numerical gate before profiling

A [bounded CPU counterexample](../DYNAMICS_NUMERICAL_PROBE_R1.json) found that global FP32 cumulative-energy subtraction can erase later small changes after an early large pulse. At sequence position16000, both horizons should have log energy−9.21024 but return−18.42068 (the epsilon floor). This verifies a target-construction bug on a valid synthetic fixture; its prevalence in supplied data is unmeasured. The [fixture source](precision_probe_r1.py), [execution receipt](../DYNAMICS_NUMERICAL_EXECUTION_R1.json), and [log](../dynamics_numeric_probe_r1.log) preserve the check:5.74seconds/657476KiBpeak, zero swaps, no dataset arrays/GPU/fits.

Root holds the dynamics profile until a stable FP32 window calculation passes direct-window, boundary, mask, and long-sequence tests plus independent review. No auxiliary profile or scientific fit ran with the faulty construction. The52 mathematical targets, four-fit panel, weight0/.05, and main metric remain unchanged. This is an implementation repair, not a new scientific arm; the active frozen i1 panel is unaffected.
