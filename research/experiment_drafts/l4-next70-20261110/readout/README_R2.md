# L4 fixed-endpoint readout R2

R2 repairs the launch failure observed in R1: resolving the virtualenv's
`bin/python` symlink replaced it with the system interpreter, which did not have
NumPy. The receiver now converts the supplied interpreter to an absolute path
without dereferencing symlinks. It records that lexical path in
`control/readout-r2/owner.json`. A CPU regression fixture models a venv symlink
whose resolution points at a different system interpreter; a second fixture
uses an actual filesystem symlink where the test OS permits it. Neither fixture
executes the interpreter.

R2 uses distinct receiver and export locations (`control/readout-r2` and
`runs/{cell}/readout-final-r2`). The six fit receipts and fixed endpoint models
remain the already-completed R1 artifacts. Source, source-manifest, commit and
chronological split pins are unchanged. The receiver does no training and does
not retry an export or score stage. It uses the original first-profile clock:
the fit wait ends at the existing 11-hour cutoff and all work must finish before
the original 12-hour wave deadline. It never starts a new clock window.

Launch from the ROOT-owned POSIX receiver with the supplied virtualenv path
preserved exactly as written:

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  <deployed-readout-folder>/readout_r2.py \
  --root /data/lc-epanov_s/tabdata-next70-20261110-r1 \
  --source /data/lc-epanov_s/tabdata-next70-20261110-r1/source-specialist-r2 \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src
```

Prediction and scoring children use that same interpreter path. The receiver
still imports only the established owned-command executor from `--old-src`,
checks the frozen source capsule and existing fit receipts, and enforces the
existing per-stage 600-second cap, output cap, ownership receipts, exact score
comparison and no-retry policy.

Focused CPU checks:

```bash
python -m pytest research/experiment_drafts/l4-next70-20261110/readout/test_readout_r2.py -q
```

These tests do not load PyTorch, contact a GPU, or execute an interpreter
symlink.
