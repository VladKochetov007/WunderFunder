# Child CONTROL terminal checker v1

`child_control_terminal_check_v1.py` is a bounded, source-qualified checker for the already admitted seed-1001 CONTROL fit. It is prepared for root-owned use only after the exact unit invocation exits and root has captured typed native terminal evidence. A successful checker run emits a private, exclusive-create JSON receipt; it does not capture native exit status, admit or stop a unit, or accept a fit.

The checker binds run `fg19_child_control_f0_s1001_fit_v1`, unit `computer-vision-homework-fg19-child-control-fit-v1.service`, invocation `e66e0d0703314e4fb5d47dcdbfcaf202`, source commit `68ff682977b822f38351aac0f621146f5f13b440`, and the admitted 167-file source manifest SHA `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`. It requires exact typed success (`active/exited`, PID 0, `ExecMainCode=1`, status 0) before inspecting the completed artifacts.

The gates cover launch/admission/source/split/cache identity; accepted parent checkpoint and all-three shape-profile ancestry; exact child CLI and continuation contract; the 467-input/25-output FP32 model and inherited normalization; fit-only history/logs and six epoch counters; AdamW state and the 44,760-step OneCycle schedule; restricted CPU checkpoint reads; and the child’s final `EpochOrderSampler` state (`sampler_epoch=5`, continuation `next_epoch=6`). The scheduler gate follows the pinned trainer’s OneCycle setup and PyTorch cosine behavior at saved `last_epoch=44,760`, one step beyond the last phase endpoint at `44,759`; it checks terminal LR `4.0000136407623314e-8`, terminal beta1 `0.9499999998635378`, and optimizer group endpoints (`initial/max/min LR = 4e-6/1e-4/4e-8`, base/max momentum `0.85/0.95`). It rejects the nominal minimum LR or any legal-but-unrelated beta1. A 100-step CPU-only synthetic OneCycle fixture validates the one-step-past-end cosine behavior without running the full fit. For the seeded loader it validates source-shaped RNG state and records the child’s observed restricted-load digests. It does not compare child RNG bytes with the parent fit’s content digests.

The child uses `representation_mode=base`, which still carries the fitted input-dictionary payload through every selected checkpoint and the training state. The checker restricted-loads the accepted 22,557-byte dictionary artifact, checks its source-defined recipe and fit-member lineage, invokes the pinned source deserializer and state-hash function, then requires the checkpoint, training-state, and provenance copies to match that artifact. It does not reuse the parent checker’s disabled-representation auxiliary assumption.

The complete private-D and transitive source import pins, launch input receipt paths, runner/runtime identity, and retained isolated-import result are in [CHILD_CONTROL_TERMINAL_CHECK_V1_DEPENDENCIES.json](CHILD_CONTROL_TERMINAL_CHECK_V1_DEPENDENCIES.json). The cold-import receipt is [child_control_terminal_check_v1_cold_import_receipt.json](child_control_terminal_check_v1_cold_import_receipt.json). Its focused unittest copies only the four exact pinned private-D files into a temporary preparation directory and imports them in a fresh `python -I` child. It asserts each imported private-D module resolves to that copy, rejects any other absolute preparation-module origin, and fails if any transitive `wnn` module resolves outside the canonical source checkout. With `CUDA_VISIBLE_DEVICES` empty and an audit sentinel rejecting opens beneath B’s `data` and `research_tmp`, the proof observed no CUDA initialization and zero blocked-payload open attempts.

The local smoke test is deliberately synthetic and does not load a model, dataset, cache, or dictionary payload. Run it from the maintenance checkout with one-thread limits:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WNN_THREADS=1 RAYON_NUM_THREADS=1 \
  .venv/bin/ruff format \
  research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_control_terminal_check_v1.py \
  research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/test_child_control_terminal_check_v1.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WNN_THREADS=1 RAYON_NUM_THREADS=1 \
  .venv/bin/ruff check \
  research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_control_terminal_check_v1.py \
  research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/test_child_control_terminal_check_v1.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WNN_THREADS=1 RAYON_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover \
  -s research/campaign_2026-10-19/full_gru_tweaks_r1/preparation \
  -p 'test_child_control_terminal_check_v1.py' -v
```

The retained cold-import receipt records the local test interpreter (`.venv/bin/python`, Python 3.13.11, NumPy 2.2.6, Torch 2.14.1). The actual fit worker’s required runtime is Torch `2.14.0+cu130`; the checker enforces that version when root eventually runs it on the worker. Cold import is not a substitute for running the post-exit gates, and no terminal result or acceptance receipt exists in this preparation.

The checker CLI requires the completed run directory, the pinned source root and manifest, canonical split, CONTROL and profile evidence roots, admission/launch/launch-acceptance records, root-captured terminal evidence, cache root and exact cache acceptance/stop receipts, plus a new output path. The exact paths are supplied by root after terminal capture; they are not inferred from the launch record. The checker imports only the four pinned private-D Python files named in the closure manifest and source helpers whose resolved paths lie within the verified staged source tree.
