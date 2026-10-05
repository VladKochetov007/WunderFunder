# Independent W384 v7 review

**Decision: GO for controller-owned private upload and intercepted/live one-batch probes after a fresh provider quota and resource-admission check. NO-GO for fit launch until both W256 and W384 probes finish on distinct T4s and pass their recorded gates.** This review performed no provider action and did not inspect current provider inventory; it establishes local artifact integrity and bootstrap behavior only.

I compared the assembled v7 candidate with v6. The scientific payload is unchanged: the tar archive is byte-identical, all four fit/probe specs are byte-identical, and the phase and resource runners are unchanged. The kernel change is exactly the v6→v7 dataset slug plus one behavior line: `archive_members` now receives `seen`, the set built from actual tar members, instead of every path in the manifest inventory. The existing mount-collision rejection remains in place. The metadata differs only in dataset slug, kernel ID, and title. This repairs v6's false rejection of external mount paths that were inventory records but not archive members without weakening rejection of a real archive/mount collision.

Independent archive accounting found 287 manifest file records, 265 actual regular tar members, 26 external mounts, 26 external paths also listed in the manifest inventory, and zero external paths in the actual archive. Each archived manifest member matched its recorded size and SHA; all four archived specs matched the staged specifications semantically. Tar members were unique, regular files, and path-safe. The v7 receipt identities were recomputed and matched the assembled artifacts:

| Artifact | SHA-256 |
|---|---|
| Archive | `db08107d0df5ee9194959bbba4862385189d1fc0c91324ecd28ed12e8bebdd6e` |
| Bundle manifest | `4518e583e9eab4bcaa5c5eb499958e9329748191006e68cd89a101d12cee1958` |
| Kernel bootstrap | `498c0a5d1105a9fbbe34d02fffb7d4a6c12cd5d2f6b10f8ef647b7bac03bb90a` |
| Kernel metadata | `66b5db1754956b0ca619835d4fb629f1f74dbd343cad604946654da80aad6d77` |
| Phase runner | `4f19bd7629dee6c16f6b8b1c1249cbe439c0b46c34284be1393f23f4cf6af656` |
| Resource runner | `01bb1846c6825b0af860f86607ef6ecd8acdc1b311c34ba16bab70c0bf328b9d` |

The assembled bootstrap fixture passed under the one-CPU/2-GiB limiter:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  WNN_SLICE=wnn-campaign.slice scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/w384_v7/bootstrap_integration_fixture.py
```

Receipt: `status=passed`; the valid bundle reached the intercepted phase invocation; the injected real tar member colliding with an external mount was rejected before phase invocation; traversal and mount-allow-map mismatch guards passed. Fixture counts matched the independent archive accounting above. The resource-admission suite reports 12/12 passes in the local build receipt/review; I did not rerun it for this review.

The candidate preserves private T4 execution, disabled internet/TPU, per-worker 8-GiB RAM reservation, concurrency-two admission, bounded cache advice, 25,180-second internal watchdog, and 25,200-second provider cap. The four sealed specs retain their recorded hashes. Controller should use the v7 kernel identity `vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7-runner`, refresh Kaggle quota/availability immediately before upload, keep timeout at 25,200 seconds, and require the recorded two-probe gate before fits. No changes to science arguments, inputs, checkpoints, masks, or selection rules were found.
