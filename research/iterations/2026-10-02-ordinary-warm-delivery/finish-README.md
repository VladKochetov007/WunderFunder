# Conditional delivery finisher and preservation

Source only; no jobs were started. Existing D/W/I inputs and source files are unchanged. Root owns the final-fit and native-quality allocations and the decision based on W plus its seed-four confirmation.

Copy `finish-config.template.json` to a fresh `finish-config.json` and bind the exact final training/native allocation hashes, an absolute UTC deadline allowing completion plus the bounded finishing work, and `execution_authorized:true`. The template's runtime CPU8/sibling48 are a prospective pair, not an allocation; root may choose a fresh idle pair before freezing the config. The CP311 package check is fixed CPU8. Its three finishing stages each have1CPU/4GiB/swap0 caps: packaging120s, package check120s, runtime-plan preparation30s, with owned UUID scopes and TERM/KILL. All stages are also bounded by the root config deadline. Disk must remain above3GiB;64MiB extra is required before starting a stage.

Invoke only after root authorization:

```sh
.venv/bin/python research/iterations/2026-10-02-ordinary-warm-delivery/finish_delivery.py \
  --config research/iterations/2026-10-02-ordinary-warm-delivery/finish-config.json \
  --config-sha256 CONFIG_SHA
```

It polls canonical `execution-01/result.json`, `native-quality/supervisor-result.json` and `native-quality/evaluation/report.json`. Failure, timeout, changed source/binding or output collision stops without retry. It checks the native pipeline's four completed stages, exact report/allocation/source identities, final two-fit count and scope release, and the following prospectively fixed point-estimate gate:

- Native warm minus current native incumbent: pooled≥0.0001 and each fold>0.
- Native warm minus GPU warm: pooled≥−0.0001 and each fold≥−0.00015.
- No requirement that a confidence interval exclude zero. These are spending criteria on exposed data, not a generalization claim.

A failed quality gate produces `NATIVE_GATE_FAILED_NO_PACKAGING`. A passing gate runs only the existing packager, fresh CP311 package self-consistency, and unallocated runtime-plan builder. Success is `finish-delivery-01/result.json` status `READY_RUNTIME_PLAN`, containing the exact new ZIP, manifest, package-check and runtime-plan hashes. It does not create a runtime allocation, invoke `start_runtime.py`, start a runtime audit or modify the incumbent. Root must separately authorize the complete callback audit.

Optional local collector, explicitly invoked on the local creative checkout:

```sh
python3 research/iterations/2026-10-02-ordinary-warm-delivery/collect_delivery.py \
  --phase ready --deadline-utc ROOT_UTC_DEADLINE --remote-cpu FREE_CPU
```

Use `--phase runtime` only after the separate runtime has been launched and should complete. `ready` refuses collection if runtime has already started, avoiding a moving runtime tree. The collector waits for terminal phase receipts and exited owned controllers/scopes, then hashes/copies the complete D tree and both registered final-run directories, including every checkpoint, alias, initial-state and history file. No arrays are decoded, inference/metrics invoked, models loaded or files deleted. Large NPYs may be losslessly gzipped locally; the manifest binds original uncompressed bytes/hash and local representation/hash. Differing existing local files remain in place; final remote versions map into `remote-final-snapshot`. Remote trees are rehashed after transfer. Partial downloads remain on failure. Local disk floor3GiB applies throughout; all remote hashing is pinned to the explicitly supplied CPU with180s per inventory and an absolute collector deadline. Successful `preservation-{phase}.json` is written locally and synchronized remotely; failures are preserved locally without claiming successful preservation. No automatic collector retry.

Only source/AST and metadata bindings were checked during preparation. The finisher and collector themselves have not run. Their source receipts are separate from the previously frozen D training/package/runtime source receipt.
