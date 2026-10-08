Final source-only check uses `preflight_r5_sealed.py` against `r5_capacity_packet_v5.tar.gz`, under the campaign support flock and 1GB RAM cap. It verifies the actual archive/source members, actual successful V4 profile consumer, exact archived trainer parsers, and rejection of pending authority. No fit or GPU runs locally.

Root first integrates the final commit and the controller gzip compatibility patch, then reruns `build_r5_packet.py` to bind the final main guardian hashes and runs the sealed preflight. Freeze these files after root review.

Root refreshes four signed inputs with:

```
.venv/bin/python <controller>/sol_continuation/probe_signed_inputs.py --runtime .runtime/r5-fresh-signed --attempt-id e529-vast-capacity-muon-f0-20261008-r5 --campaign-label wnn-e529-capacity-muon-f0-r5-20261008 --receipt .runtime/r5-fresh-signed/public_receipt.json
```

Use the support wrapper for both preparation CLIs. Do not print or commit private URLs. The runtime directory must be new; private plans use mode0600. The final signed URLs must be fresh at dispatch. Prior HEAD verification proves the four expected object lengths, not the future remote content hashes.

Prepare the exact final intent and embedded authority (no provider mutation):

```
.venv/bin/python <controller>/sol_continuation/prepare_r5_root.py --packet-directory <controller>/sol_continuation/r5_packet --offer .runtime/<fresh-single-offer>.json --destination .runtime/r5-root-sealed --account-scope-sha256 <verified-account-scope> --signed-plan .runtime/r5-fresh-signed/signed_plan.private.json --authorize-packet
```

This produces gzip source packet, exact six-input intent, four-input prefetch admission, and an explicitly unauthorized paid admission draft. Root must recheck fresh quote/account/credit/current private ledger, independent final packet review, exact staging model, and reservation before setting paid admission authorized and renewing its110second timestamps. Root's existing r4 lifecycle/systemd dispatcher pattern owns create, heartbeat, collection and teardown; use the new intent/admission/prefetch paths and exact attempt ID. Never call a direct create command.

The packet authorization fields are authorized, attempt_id, recipe_sha256 and worker_spec_sha256. No conditional profile flags exist: actual3090/4090 numeric/precision/memory/time gates precede each fresh four-epoch fit. Profile weights never initialize fits. The packet requires30 completed smoke updates, zero AMP skips, finite loss/gradient, measured GPU peak and at least1024MiB free. Its process supervisor conservatively sums all descendant RSS every second, kills nested trainer process groups above28GB and preserves terminal receipts. Host available RAM must be35GB before loading; selected host totalRAM must be64GB for this root request. Runtime is further bounded by root's hardstop minus setup/collection;2700/6300second empirical fit projections are not guaranteed on a new worker. The900second per-fit cushion plus remaining aggregate5400second export budget is a conservative admission reserve, not an added fit.

Pre-authorization compressed closure measured1,625,051bytes. At historical4,000B/s laptop throughput, transfer406.3seconds.37,102,250,127cloud bytes at prior678.4Mbps gives437.5seconds ideal,590.7seconds with35% network margin. Add450seconds setup and180seconds SSH/hash overhead:1627seconds,173seconds margin under1800. Root must recompute with the fresh offer and final authorization-sealed packet size; fail closed if inadequate. Remote actual throughput/hash completion is unproven until execution.
