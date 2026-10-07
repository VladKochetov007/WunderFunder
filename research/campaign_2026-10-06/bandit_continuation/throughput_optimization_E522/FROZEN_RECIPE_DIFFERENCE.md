# E-520 frozen recipe versus E-522 IO-only proposal

| Dimension | Frozen E-520 scratch-pair recipe | E-522 proposed handoff integration |
|---|---|---|
| Model and ancestry | Independently scratch-initialized raw-only GRU, hidden 128, two layers, dropout 0; seeds 3 and 4 | Unchanged |
| Fit membership | Fold 0 train/valid fit groups, component-disjoint evaluation groups, corrupt group 528 excluded | Unchanged |
| Training target and loss | `wmse`; target clipped to ±2 by pinned `weighted_mse_loss`; weights use absolute clipped target; chunk normalization | Unchanged |
| Optimizer and schedule | AdamW, lr 0.001, weight decay 0.01, OneCycle | Unchanged |
| Update and precision | Four epochs, batch 16, chunk 2,000, fp16, `--fast`; fixed zero-based epoch-3 checkpoints | Unchanged |
| Prediction and grading | Streamed 4,000-row exports; frozen E-520 pair grader and fixed E-511 anchor comparison | Unchanged |
| Input identity | Initial full raw train/valid SHA-256 plus footer/stat and mounted asset/split checks; portable registry checks full content SHA again in `load_campaign_split` | Reuse the first digest only if it matched both frozen expected and portable-registry SHA-256 and the exact canonical path/stat identity is unchanged; otherwise use the pinned full-hash verifier |
| Trainer source | Pinned archive `78f9e5c3...6606081fc`; `train_rnn.py` and `wnn.splits` are not edited | Unchanged; only the worker-local `wnn.portability._content_digest` lookup is wrapped before first split load |
| Execution cost | One repeat full content pass over train and valid in the profile/fit input preflight | Removes one repeat pass per raw input in that worker if identities match; no cross-run digest persistence and no raw data skip |

E-522 is an execution-only change. It does not alter fitting membership,
features, targets, losses, sampling, checkpoints, predictions, or scoring.
