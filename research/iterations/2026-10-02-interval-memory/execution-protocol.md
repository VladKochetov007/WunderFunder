# Root-reviewed launch scaffolding

No allocation, fits, data or metrics are executed by preparing these files. Root owns all invocations below. The immutable scientific commands remain `e4deda87a9520aad4ec0c3a95886bfddacbaa4e11a4bdc36ed838f2d0de3c663`; the unmodified generic queue remains `a6d51d912635c071bcf001d98e1e7e251718992a40c84db2a4586345019bd5c2`.

Allocation requires both atom execution and grader COMPLETE with no live owned scopes/controllers, plus `cuda-01/result.json` PASS_GENERATED_INTERVAL_CUDA and `cuda-execution-01/result.json` PASS_GENERATED_CUDA_SMOKE. GPU0 must have 8 GiB free; affinity is fixed16–21, quota6CPU/20GiB RAM, one sequential GPU job, allocator fraction.375 and owned GPU monitor7GiB. Global deadline is at most1.25h, including all12 jobs; per-stage caps1200s parent/900s H2/600s OOF cannot extend that deadline. No retries or edits of running sources.

The allocator verifies the two original shallow256 seed3 EMAep3 SHA values from commands and writes fresh stat/hash bindings. It inherits the existing verified teacher/data/propensity binding. Every fit requires fixed endpoint(s), model.pt, history.json, initial-state.json and the106-argument validator receipt. Parent endpoint is ep0, H2 endpoint ep1. Every OOF requires the full1873-index array and its provenance JSON. Source/input manifests are rechecked between jobs by the unchanged queue.

Root CLI on SSH, from the creative checkout, in this order:

```sh
.venv/bin/python research/iterations/2026-10-02-interval-memory/allocate.py --gpu 0 --cpus 16,17,18,19,20,21 --hours 1.25
.venv/bin/python research/iterations/2026-10-02-interval-memory/freeze_evaluation.py
.venv/bin/python research/iterations/2026-10-02-interval-memory/launch.py
.venv/bin/python research/iterations/2026-10-02-interval-memory/launch_finish.py
```

Finish waits for the exact8fit/4OOF completion, then runs the frozen grader once under CPU28,29/quota2CPU/4GiB/noGPU/600s. It writes terminal failure without retry if training, integrity or grading fails. Three fixed contrasts are candidate-incumbent primary, accumulated-instantaneous scientific, and control-incumbent.

Copy the new training-allocation.json and evaluation-source.json from SSH to their identical local paths before starting the local-only collector:

```sh
python3 research/iterations/2026-10-02-interval-memory/launch_collector.py
```

The fresh interval collector leaves existing collectors unchanged. It waits for both known controller identities to exit and terminal receipts, verifies no owned scopes remain, then copies only this campaign and its8 registered run directories. Large absent `.npy` files are streamed through deterministic gzip level1, with a3GiB local free-space floor throughout. Every saved representation is read back and checked against the uncompressed remote SHA/length; source artifacts remain untouched. It writes a local+SSH preservation receipt. Failed runs are preserved too; preservation success is not training or scientific success.
