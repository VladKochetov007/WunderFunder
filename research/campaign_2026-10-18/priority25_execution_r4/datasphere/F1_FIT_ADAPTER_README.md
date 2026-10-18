# Rank 1 DataSphere fold-1 pair adapter

This is a follow-on one-cell adapter for E-RR18-01-LABELS, cell F1/seed1001.
It does not alter the admitted fit-only profile packet. The adapter reuses
`profile_staging.py` for archive safety, exact mounted-file discovery, data
hashes and target-index staging, then calls `wnn.training.runner.run_training`
and the frozen source's `scripts/predict_run.py`.

The only accepted pair is cold labels-only GRU: H512×1 (`rr18_r01_wide_f1_s1001`)
and H384×2 (`rr18_r01_deep_f1_s1001`). Both use source a266b036, fold-1 manifest
SHA-256 `d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9`,
seed1001, train+valid membership, B8/C2000, six epochs, AdamW/OneCycle, strict
FP32, zero dropout, seeded order, fixed epoch-5 checkpoint, and no fit-time
evaluation. There is no KD, teacher, propensity, initializer, or profile-step
argument. The models are near-parameter matched: 2,210,306 vs 2,151,554
trainable parameters. Six epochs is a fixed comparison horizon, not a
convergence guarantee.

`f1_fit_spec.template.json` is deliberately not runnable. The admitted profile
must first establish usable runtime; an execution copy must set `state` to
`SEALED_FOR_F1_PAIR` and populate positive `internal_deadline_seconds`,
`collection_reserve_seconds`, and
`external_cancel_deadline_seconds_from_submit`. The adapter keeps a fit alarm
through the internal deadline, then uses the separately declared collection
reserve through `entry start + internal deadline + collection reserve`. It
records SIGTERM during collection and continues bounded finalization; the
collection alarm remains armed, and expiry leaves a partial receipt without
claiming an archive. The submit-relative external deadline must leave the
declared collection reserve plus the provider's 60-second graceful shutdown,
and admission must separately include observed submit/start delay. No runtime
deadline is inferred from the failed profile attempt. The adapter rejects
unset deadlines, changed source/archive/split identities, changed arms, and
recipe drift. `f1_fit_config.template.yaml` is a
portability template: its local absolute archive paths and provider dataset
identifiers still require the already-authorized DataSphere project input
bindings. It has not been submitted. The DataSphere 0.10 SDK rejects an empty
manual-environment `local-paths` list before provider submission, so the
template uses the inert `runtime_marker.py` module, matching the admitted
profile envelope.

Both fits run sequentially in one provider allocation. Canonical
`--save-training-state` writes an epoch-boundary resume state throughout the
fixed six-epoch schedule. On completion, the adapter invokes the canonical
feature-only FP32 fold-1 evaluation exporter and checks its checkpoint hash,
ordered fold membership and sidecar identity. It does not grade the metric. Its
terminal bundle includes run checkpoints,
`training_state.pt`, `history.json`, provenance, logs, predictions, sidecars,
and the fit/split/registry identities. On SIGTERM or the internal deadline it
tries to bundle every artifact already written within the declared collection
reserve. If that deadline expires, it retains a partial local receipt but does
not claim that an archive was collected. If interrupted within an epoch,
the last durable epoch state and fit log are retained; a clean `history.json`
is only written by the canonical trainer at normal finalization. SIGKILL,
provider loss before output collection, or disk failure can still prevent a
complete bundle.

Before packaging, the adapter writes `fit-artifact-manifest.json` with a
relative path, byte count, and SHA-256 for every other file, then verifies the
tree and archive against that manifest. The compressed archive fails closed at
768 MiB to leave room under the SDK's 1 GiB aggregate download limit for
provider logs and diagnostics; this cap alone does not guarantee the aggregate
limit. The controller must explicitly download the `fit-results.tar.gz` output
and verify the provider file's path, size and hash plus this embedded manifest
before marking collection complete. A zero SDK exit status is insufficient.
After extraction, verify the tree as well:

```sh
python3 f1_fit_bundle.py --verify-bundle /path/to/fit-results.tar.gz
python3 f1_fit_bundle.py --verify-tree /path/to/fit-results
```

These checks catch missing, extra, or changed files. A successful SDK download
exit code alone is not evidence of complete collection.

No fit, GPU allocation, provider call, or grading was run to validate this
adapter. `python3 -m unittest -v test_f1_fit` exercises fixed recipe validation,
command invariants, producer-shaped artifact checks, bundle integrity, the
downloaded-tree verifier, and the output-size guard entirely offline.
