# R10: fixed recent joint-support screen

One target-free CPU diagnostic for H-NF19-08 compares the convex hull and axis-aligned box of the same preceding256 raw input pairs. Fixed columns0/52,256-row window, original stride40 and64 fitting components from the accepted exposed R3 panel. This tests whether the two support descriptions yield distinct cases; it does not test label usefulness, quality or transfer.

[Frozen design](DESIGN.md), [selection history](SELECTION_HISTORY.json) and [independent design review](DESIGN_REVIEW.json) precede any actual panel reads. [Preparation manifest](PREPARATION_MANIFEST.json) pins11 source/metadata files, including the semantic-only canonical `wnn/data.py`; the four-module runtime imports no model, scorer, Torch, PyArrow or canonical trainer. `geometry.py` implements the fixed normalized-tolerance hull; `panel_io.py` validates all96 metadata identities but opens only64 fit x/mask pairs; `run_probe.py` performs the bounded queries and writes scalar receipts. `test_r10.py` and the [two exact R3 metadata fixtures](fixtures/README.md) support six tiny checks recorded in [TEST_RECEIPT](TEST_RECEIPT.json).

[Actual results](RESULTS.md) and [execution records](records/README.md) now complete this one diagnostic; it exited0, was captured and stopped. The exact launch-time preparation README is retained at [PREPARED_README.md](records/startup_v1/PREPARED_README.md). Author-only metadata/test revisions preceded this root freeze; the final manifest is `49f39bf3e32faed6ecdcfef38e1b93cd95a5386177999c78b866b0de7dd1e483`.

The single intended B CLI is:

```sh
python -B -m idea_screens_r10.run_probe \
  --sample /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-r3-v2 \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r10-v1 \
  --expected-sample-sha256 b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64 \
  --expected-preparation-manifest-sha256 49f39bf3e32faed6ecdcfef38e1b93cd95a5386177999c78b866b0de7dd1e483
```

It requires a fresh one-off [resource/staging gate](records/startup_v1/README.md):0.5CPU,1.4GB high/1.5GB max, zero swap,600seconds, hidden CUDA, BLAS1 and physical B scratch. No target/eval array, model, training, grade, new GPU cell or provider action is involved. Arrays remain on Beleriand. The existing F3 pair is complete and is not polled or rerun.

This README is navigation outside the scientific manifest. Exact launch-time bytes will be retained before later result annotations.

[Result audit](RESULT_AUDIT.json), [resource reconciliation](RESOURCE_RECONCILIATION.json) and [resource metadata review](RESOURCE_METADATA_REVIEW.json) preserve scientific and operational boundaries. Current result annotations do not change the11scientific source pins.

The canonical result audit corrects two metadata labels before publication; [original audit](RESULT_AUDIT_V1.json) is preserved with identical scientific values. The root resource review is a byte-identical navigation alias of the captured review.

[Publication validation](PUBLICATION_VALIDATION.json) binds the final staged source, metadata, navigation and append-only ledger snapshot.
