# i1 causal-history A/B/C screen

[CPU delivery packet, 23 October](submission_r1/README.md) adds the causal
367-input serving path and fixed 90%B/10%C FP32 packaging for each graded pair.
Weights are unavailable on the receiving Mac; the user requested proceeding
without them. Model ZIP, exact streaming parity and timing remain pending.

This folder records the amended six-fit i1 memory screen in the October21
allocation. The sustained profile and all six fits completed; the terminal
[fixed grade](runtime/GRADE_R1.json) covers F0/seed1001 and F2/seed1002. The
original F0 A/B-only interim result remains frozen, and the separate new
[F0 A/B/C interim wrapper](interim_f0_abc/README.md) was not needed for the
complete panel grade. The independent [grade review](../../../unified/reviews/I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md)
records the small exposed fixed-mixture signal, its conditional uncertainty,
and the native-serving and initial-function limits; root owns follow-up
allocation and promotion decisions.

The panel has two paired fold/seed cells: A/B/C on F0 with seed1001 and A/B/C
on F2 with seed1002. Each arm has two fits, for six new fits total; there are
no teacher fits, initializer fits, or reused model fits. F0's historical
10% subset is reused byte-for-byte, while F2's 10% whole-component subset is
newly frozen. The old October20 recipes, experiment ID, and membership remain
untouched.

The screen uses cold GRU H512×1, 367 fixed input slots, B8 complete 20,000-row
sequences, three epochs, fixed zero-based epoch2 checkpoint, FP32, WMSE with
chunk normalization, AdamW, and OneCycle. Ten 2,000-row TBPTT windows are
updated per batch. F0 has 315 of 3,148 eligible fitting components, 1,118
train and118 valid fit sequences, 154 B8 batches per epoch and4,620 update
attempts per fit. F2 has315 of3,145 eligible fitting components, 1,148 train
and141 valid fit sequences, 161 B8 batches per epoch and4,830 update attempts
per fit. The sampler omits only the final incomplete B8 group each epoch; it
never truncates a sequence.

Arm A masks raw i1, derived i1 event inputs and all156 history columns after
normalization. B masks only the history columns. C retains the history. All
arms retain model width367. The history is causal: residuals against preceding
EMAs use half-lives32 and512; change energy uses the current transition and
half-life512; the first row is zero and state carries across chunks but resets
between sequences. Its scale and model normalization use fitting groups only.

The corrected profile used 100 measured B8 batches after10 warm-up batches,
with ten 2,000-row TBPTT windows per batch. The [parent profile receipt]
(../I1_SUSTAINED_PROFILE_R3.json) records 454.953s wall time, 4.36124s median
per measured batch, 1.83 sequences/s, 1,100 finite updates,
and zero skipped updates. CUDA peak allocation was 0.98GB and reservation
1.216GB; the 3GB cgroup experienced reclaim pressure but no OOM. For the
complete panel, 2,835 fit batches project to 12,364s; with a 20% fit-time
margin, 360s setup, 1,800s reserved for all six exports and grading, and the
455s profile, the total is 17,452s of 18,000s, leaving 548s for recovery.
Export time is reserved rather than measured, so parent must retain the
full-panel gate while executing. The profile-only plan omits
`--save-training-state`, which the trainer rejects on a profile early exit;
the fit recipes retain it. It also omits the three redundant independent-width
options, which the validator rejects when causal research features are active.
Those options had equalled `--hidden 512`; a small CPU regression test asserts
that equal explicit widths and the default H512 model initialize to identical
tensors with `research_features=none`. Another CPU test parses the actual
profile plan and runs its static argument validators without opening campaign
data, and a third validates all six literal fit commands the same way.

The test file also checks prefix causality against an independent scalar EMA,
chunk carry/reset, 367-column A/B/C mask behavior after normalization, and
checkpoint restoration. Parent-owned profile invocations R1 and R2 failed at
CLI validation before reading data or performing updates; corrected R3
completed successfully. The recipe audit found repeated `-u scripts/train_rnn.py`
tokens in the generated fit commands. All six packet-only plans now contain one
trainer entrypoint, and the all-plan static test guards against recurrence.

All six recipes use `scripts/kaggle_sequence.py` with three sequential steps:
fixed-epoch fit, full-fold evaluation-only feature export, and `score_run.py`.
The exports cover exactly468 F0 or443 F2 evaluation groups, each with shape
`(N, 20000, 2)` and float32 payloads of74,880,000 or70,880,000 bytes. The
grader parses every NPY header, checks the complete file length, requires the
sidecar's ordered row groups to equal the fold manifest, and uses the official
`global_wp_clip2_v1` metric. It reports A/B/C and the fixed mixtures
`0.9*B+0.1*A` and `0.9*B+0.1*C`, per-target scores, four fixed position bins,
and source-component paired bootstrap deltas for A−B, C−B, C−A, and both
mixtures versus B. It verifies matching A/B/C initialization and normalization
identities, three complete fit epochs with the exact per-fold optimizer update
count and no skipped updates, and FP32 inference exports. It performs no
coefficient search. The folds have prior exposure; bootstrap intervals are
conditional on each fitted pair and do not measure training-seed, fold, or
public-transfer uncertainty.

The independent [prefit recipe audit](PREFIT_RECIPE_AUDIT_R1.json) records the
source hashes, fold and data identities, fit memberships, implementation
findings, prediction contract, and profile status. The [F2 membership
manifest](fit_f2_10pct_s1002.json) is reproducible from the small frozen fold2
JSON by [`build_fit_subset.py`](build_fit_subset.py), without opening target
or feature caches. The historical F0 manifest was regenerated for a check and
matched its recorded component list, group lists, and member hash exactly.

The panel files are:

- [`profile_c_f0_s1001_r1.json`](profile_c_f0_s1001_r1.json): exact corrected
  sustained C-profile argv; profile only, not a fit.
- [`recipes/`](recipes/): six `kaggle_sequence` plans with distinct new run
  names and complete-fold prediction and scoring steps.
- [`grade_i1_panel.py`](grade_i1_panel.py): fixed two-fold grade using the
  repository's row-group alignment, official scorer, and component bootstrap.
- [`interim_f0_ab/`](interim_f0_ab/README.md): preserved first matched F0 A/B
  grade and its fixed 90%B/10%A comparison; it predates arm C completion.
- [`interim_f0_abc/`](interim_f0_abc/README.md): new memory-gated F0 A/B/C
  wrapper with all five fixed contrasts, point scores in four time bins, and
  paired component intervals for whole-fold comparisons; no F2 result or
  full-panel promotion claim.
- [`fixed_mix_geometry_r1/`](fixed_mix_geometry_r1/README.md): code-only, source-bound F0 diagnostic for the fixed 90/10 B/C mixture with the same 90/10 B/A control. It streams existing complete-fold predictions and reports raw/clipped weighted geometry; root review and execution are pending.
- [`tests/test_i1_history.py`](../../../../tests/test_i1_history.py):
  synthetic causal, reset, mask, restoration, initialization-equivalence, and
  profile-argv validation checks.

Run the synthetic tests in the admitted low-quota CPU slot with:

```sh
PYTHONPATH=/home/vlad/development/wunder/research_tmp/i1-screen-20261021/src \
WNN_MEM_MAX=1800000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_TMP_ROOT=/home/vlad/development/wunder/research_tmp \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
-m pytest -q /home/vlad/development/wunder/research_tmp/i1-screen-20261021/tests/test_i1_history.py
```

After all six fits and exports finish, grade the complete panel with:

```sh
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/research_tmp/i1-screen-20261021/research/campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py \
--output /home/vlad/development/wunder/research_tmp/i1-screen-20261021/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/grade_i1_panel_r1.json
```

The parent cycle README and frontier should link this folder; root owns those
files and the execution, admission, review, and final evidence update.

Packet/grader label `E-PLATEAU-I1-20261021-R1` and the ledger amendment `E-PLATEAU-I1-20261021-R2` refer to this same six-fit continuation panel; they do not count separate fits. The original fitting-subset label `E-PLATEAU-I1-20261020-R1` remains frozen provenance.
