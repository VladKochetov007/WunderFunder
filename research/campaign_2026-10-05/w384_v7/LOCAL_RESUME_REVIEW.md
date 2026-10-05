# W384 v7 transport-only repair review

Status: **built offline and tested locally; not uploaded or launched**. v6 failed in the bootstrap after 2.08 seconds, before any worker or fit launch. See [v6 failure diagnosis](../w384_v6/PROVIDER_FAILURE_DIAGNOSIS.md) and `../w384_v6/PROVIDER_FAILURE_DIAGNOSIS.json` for exact evidence.

The v6 bootstrap falsely treated all manifest input inventory entries as archive members. The uploaded bundle has 287 manifest file records and 265 actual archive regular-file members. All 26 external mount paths are listed in the manifest inventory but none occurs in the archive. The first rejected path was `artifacts/runs/c20260929_prop_f1/model.pt`.

v7 changes exactly one bootstrap logic line: after extraction, `archive_members` now refers to the `seen` set accumulated from actual archive members. The existing collision rejection `if logical in archive_members` remains unchanged. The bundle slug and kernel metadata ID/dataset source use a new v7 identity. The external mount allow-map, path checks, private T4 settings, 25,180-second kernel watchdog, provider wall cap 25,200 seconds, all four specs, science inputs and plans, 8 GiB worker reservations, cache advice, CPU/RAM/VRAM gates, concurrency-two gate and phase runner are unchanged.

Exact kernel source diff against v6:

```diff
-BUNDLE_SLUG = 'wnn-w384-fold1-dualt4-20261005-v6'
+BUNDLE_SLUG = 'wnn-w384-fold1-dualt4-20261005-v7'
 ...
-    archive_members = {item["path"] for item in manifest["files"]}
+    archive_members = seen
```

The metadata changes only the v6→v7 dataset slug, kernel ID, and title. Specs, phase runner, resource runner, external mount map, tar payload bytes, and v5/v6 guard implementation are otherwise byte-identical. The archive SHA remains `db08107d0df5ee9194959bbba4862385189d1fc0c91324ecd28ed12e8bebdd6e` because its science/code payload is deliberately unchanged; the new dataset manifest, bootstrap, and kernel metadata have distinct v7 hashes.

The actual assembled-bundle integration fixture (`bootstrap_integration_fixture.py`) passed. It loads the built bootstrap and actual archive, supplies fixture-mounted external files in an isolated temp tree, intercepts subprocess execution, and verifies the successful path reaches the phase-runner invocation without running training. It reports 287 manifest inventory entries, 265 actual archive files, 26 external paths in inventory, and zero external/archive collisions. A second integration case injects an external mount path as a real extra archive member, updates the archive identity for the fixture, and confirms the bootstrap rejects it before phase invocation. Traversal and mount allow-map mismatch guards also reject. `test_admission.py` passes all 12 resource/cache-advice/fail-before-worker tests. Python compilation passed.

Current v7 artifact identities:

- resource runner: `01bb1846c6825b0af860f86607ef6ecd8acdc1b311c34ba16bab70c0bf328b9d`
- phase runner: `4f19bd7629dee6c16f6b8b1c1249cbe439c0b46c34284be1393f23f4cf6af656`
- bundle manifest: `4518e583e9eab4bcaa5c5eb499958e9329748191006e68cd89a101d12cee1958`
- archive: `db08107d0df5ee9194959bbba4862385189d1fc0c91324ecd28ed12e8bebdd6e`
- kernel bootstrap: `498c0a5d1105a9fbbe34d02fffb7d4a6c12cd5d2f6b10f8ef647b7bac03bb90a`
- kernel metadata: `66b5db1754956b0ca619835d4fb629f1f74dbd343cad604946654da80aad6d77`
- kernel ID: `vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7-runner`

To reproduce locally, run:

```bash
.venv/bin/python research/campaign_2026-10-05/w384_v7/build_v7.py
.venv/bin/python research/campaign_2026-10-05/w384_v7/test_admission.py
.venv/bin/python research/campaign_2026-10-05/w384_v7/bootstrap_integration_fixture.py
```

No provider action has been taken for v7. The receipt remains `built_offline_not_uploaded`; independent controller review and resource admission must happen before any upload or retry.
