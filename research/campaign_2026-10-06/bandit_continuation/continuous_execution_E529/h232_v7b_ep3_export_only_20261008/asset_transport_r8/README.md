# H232 fixed-epoch-3 export R8

R7 reached the export worker and passed the payload, input identity, and
source-qualified membership checks. Its fixed-epoch predictor subprocess then
exited before inference because `scripts/predict_run.py` could not import the
`wnn` package. The parent bootstrap inserted the extraction directory into its
own `sys.path`, but the child process does not inherit that Python-only state.
The terminal receipt and original stderr are preserved under `reference/`.

R8 carries the already completed checkpoint and the exact R7 private payload
dataset version `vladkochetov07/h232-v7b-ep3-payload-r7-dda776b3f9b7/1`. Its
only runtime repair prepends the unchanged worker's future extracted source
root `src` directory to inherited `PYTHONPATH` before importing the worker. The
worker uses `KAGGLE_WORKING` (default `/kaggle/working`) plus its pinned
`SOURCE_ROOT_NAME`; the bootstrap binds those exact values. The existing
environment remains appended. Kernel title and ID, output receipt name, and
payload extraction directory receive the new R8 identity. There is no source
archive, worker, checkpoint, recipe, data, membership, metric, training,
profiling, scoring, or grading change.

The actual import smoke receipt records `predict_run.py --help` running from
the extracted R7 source root, using the worker's child environment and fixed
thread/loader settings with `CUDA_VISIBLE_DEVICES` empty for this local check.
It exits before dataset or checkpoint access. The R8 unit tests check the
rendered kernel insertion order, child-environment preservation, exact
terminal diagnosis, and frozen payload/member hashes.

The packet is source-only and has not been pushed. Independent root review and
a new quota/monitor admission are required before one identified provider
attempt. No provider call, fit, export, score, or grade is included here.
