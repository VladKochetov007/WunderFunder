# E482 independent six-arm grader source packet

This is a separate candidate scorer; it does not modify the Kaggle worker, packet, or any fit. It has not read evaluation rows or produced scores. Its `--verify-fit-only` mode validates provider identity, downloaded-file hashes, worker/probe/fit receipts, resource evidence, fit order and telemetry, and checkpoint contents, then returns before the test loader is called. Scoring is behind a separate explicit `--score` flag.

The grader validates the actual private-kernel proof shape: root-relative downloaded outputs, separately captured COMPLETE status observation, posted kernel metadata/source, exact dataset/input identifiers, and the remote `e482-results/` subdirectory. It expects the exact registered 23-file output inventory, checks each file's size and SHA-256, and rejects missing/added paths and symlinks. It pins the local E482 packet, worker and helper sources, E481 reconstruction/scoring implementation, and official metric code. For the six completed fit arms it checks source and dataset continuity with the three-shape probe per optimizer, distinct T4 UUID assignment, successful exits, stable OOM counters, resource ceilings, row-group traversal, telemetry cadence, finite checkpoint tensors, parameter count, and identity model buffers.

The later score path evaluates all six arms on the fixed 16 exposed components, retains raw finite targets and predictions before metric clipping, computes official global and per-target weighted Pearson scores, and registers 10 contrasts: six shape contrasts within optimizer, three optimizer contrasts within shape, and one shape-by-optimizer interaction. It uses shared component bootstrap draws (2,000 draws, seed 271828) and refuses to overwrite output.

Precision receipt fields are intentionally not treated as active backend state; see `PRECISION_METADATA_REVIEW.md`. The source and native-parity path are the evidence for the guarded recurrence computation. The unchanged Muon BF16 orthogonalization is not described as IEEE FP32.

Synthetic validation was run under a systemd scope capped at 1 CPU, 2 GiB memory, and no swap:

```sh
.venv/bin/ruff check research/campaign_2026-10-05/activation_t4_replication_v2/score_e482.py research/campaign_2026-10-05/activation_t4_replication_v2/test_score_e482.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/activation_t4_replication_v2/score_e482.py research/campaign_2026-10-05/activation_t4_replication_v2/test_score_e482.py
.venv/bin/python -m unittest research.campaign_2026-10-05.activation_t4_replication_v2.test_score_e482 -v
```

All five synthetic tests pass. They use synthetic receipts/checkpoints and metadata only; they do not read evaluation data. The source awaits independent review before the actual fit-only gate or any score run.
