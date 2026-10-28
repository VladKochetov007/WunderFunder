# TL3 dashboard bridge command

This sidecar reads the receipt-owned `tl3c_fin256_s3` trainer output and writes
only a new direct-child view folder under the existing dashboard runs root. It
does not change the fitted source, run output, or trainer process. Copy
`scripts/tl3_dashboard_bridge.py` and `scripts/tl3_checkpoint_eval.py` together
to the runtime helper directory shown below before starting the bridge.

```bash
taskset -c 0-7 /data2/lc-epanov_s/tabdata/.venv/bin/python \
  /data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1/bridge-helper/tl3_dashboard_bridge.py \
  --receipt /data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1/fin256_s3.launch.json \
  --dashboard-runs-root /data2/lc-epanov_s/tabdata/runs/quality8-full-20261028/runs \
  --poll-seconds 60 \
  --timeout-seconds 25200 \
  --eval-batch-size 4 \
  --eval-timeout-seconds 1800
```

The bridge verifies the receipt-owned timeout process and trainer descendant
using PID start ticks. Resource events are emitted only when `nvidia-smi`
reports the same trainer PID on the receipt GPU UUID. Frozen-checkpoint metrics
are written only after source, checkpoint, run identity, group order, explicit
mask, and pooled full-valid WP checks pass. Train-evaluation uses a fixed
32-sequence prefix with `all_after_warmup`; official validation uses the full
valid membership with its actual `is_scored` mask. Missing worker results stay
pending rather than being filled with placeholders.
