# Dictionary-only child admission template

This folder holds the frozen dictionary-only recipe and a local conditional renderer for one next fixed arm. The run identity is `fg19_child_dictionary_f0_s1001_fit_v1`, and its sparse representation is the source-qualified signed top-4-of-16 dictionary code plus the retained 112-wide normalized-raw residual. The final 128 appended columns stay exactly zero; the total inherited input width remains 467. The objective is WMSE with the fixed one-layer mean/variance head.

The renderer transforms the pinned EMA admission body and preserves its CONTROL, parent, dictionary, profile, T3 cache, source, resource, and ownership gates. In the trainer argv, it changes only the run name and `ema` to `sparse`; the output, unit, receipt, admission, and launch identities also change to the dictionary arm. Initialization remains the accepted parent checkpoint at epoch 3 with the same inherited normalization. The shared dictionary artifact must be present at its exact registered path, size, and digest before the trainer can start; a missing or changed cache fails closed before the canonical trainer's cache-missing fallback could fit a replacement.

The EMA terminal acceptance and exact-stop SHA pins remain unbound. The native exit capture is not a terminal acceptance. Until root supplies and validates both actual receipts for the fixed EMA invocation, this folder must not contain an executable `REMOTE_SUBMISSION.py`. The acceptance gate requires the six-epoch counters, exact source and predecessor ancestry, all 10 output-file metadata pins, selected epoch-5/training-state tensor identity proof, inherited normalization, optimizer/RNG/sample-order evidence, and a finite model fingerprint. The exact-stop gate separately binds the accepted bytes, unit, and invocation, then requires the successful exited invocation to be inactive and absent. Serialized `model.pt` and `model_ep5.pt` file hashes may differ; tensor identity comes from the accepted verifier proof. No staging, remote inspection, admission, fit, evaluation, or stop action is performed by the local checker.

Run the focused local-only checks with:

```sh
./.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit.py
./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit.py
./.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit.py --check
```

The default check verifies only local recipe, source-component text pins, argv delta, template identity, and unique body-patch anchors. It does not read model, cache, dictionary, or dataset payloads. The explicit `--render` path requires both actual receipt digests, rereads their fixed local receipt paths, validates the accepted model/state/source/ancestor and exact-stop fields, and creates the body with exclusive file creation. It refuses unbound digests, receipt mismatches, changed templates, and output overwrite.
