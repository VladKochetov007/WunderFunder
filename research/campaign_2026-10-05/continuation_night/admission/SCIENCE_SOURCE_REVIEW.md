# Independent readout and reanchor source review

Research-source disposition: GO for separately admitted resource probes and
registered paired training after the controller freezes each source/input spec.
Deployment is conditional on exporter refusal/support described below.

Read src/wnn/nn.py, src/wnn/training_controls.py, scripts/train_rnn.py, the
two workstream designs, draft-spec materializer and focused tests. The fixed
mixture changes only head activation, adds no parameter or RNG consumption,
preserves old default arithmetic, and is carried in checkpoint arguments.
Head initialization inherits it and rejects explicit conflicts. The causal
predict_run loader reconstructs through make_model and preserves the treatment.

The .1 reanchor is a fraction of the final teacher blend on rows satisfying
strict survival and sequence-boundary checks. Nonsurviving rows retain the old
target; weights remain based on original clipped labels. Ordinary and pooled
macro paths call the same target formula for the nonzero treatment. CLI checks
reject nonfinite/out-of-range coefficients, absent bootstrap/cached teachers,
and far-alpha interactions before opening data. Zero treatment retains the
historical path; bootstrap-off is a separately registered control.

Independent rerun (one CPU, two GiB, no GPU):

```bash
env WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G \
  scripts/limited .venv/bin/pytest -q \
  tests/test_continuation_treatments.py tests/test_training_controls.py
```

Result: **27 passed in 2.27 seconds**. These cover archived default
output/gradient/update parity, mixture algebra/RNG/state keys, causal chunk
inference, checkpoint/head inheritance, surviving/nonsurviving/end boundaries,
invalid arguments, and independent whole-loss/gradient oracle versus pooled
microsteps including reanchor.

Finding requiring deployment guard: native package_submission.py and
export_onnx.py assume GELU, while treatment checkpoints share the same tensor
keys. The provenance flag native_export_supported=False alone does not enforce
rejection. Root and implementation actor were notified. An unchanged exporter
must refuse non-GELU checkpoints until explicit support and shipped parity
exist. This issue does not alter the reviewed training/predict_run path.

No full fits, exports, submissions, evaluation selection or CUDA operations
were executed by this reviewer. CPU synthetic parity is not evidence of score
gain or production-shape resource feasibility.
