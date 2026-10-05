# E482 independent grader v2 source receipt

Version 1 is preserved unchanged. Its actual fit-only run stopped before the test loader because Kaggle normalizes five metadata fields at submission time, so byte equality with the locally submitted metadata was too strict. The posted source bytes themselves matched the reviewed packet. No evaluation rows or predictions were loaded by that failed gate.

Version 2 retains exact local packet/source hashes and requires the downloaded posted metadata file's recorded size and SHA to match its provider proof inventory. It then parses that metadata and allows only the five observed provider normalizations: `code_file` (`kernel.py` to the submitted filename), the exact Kaggle `docker_image` digest, `id_no` 137124210, and empty `keywords` and `model_sources`. Every other metadata field must equal the immutable local packet. The posted code file must still match the exact reviewed kernel-source bytes.

The candidate grader also uses the actual layout `remote_v1/e482-results/admission.json`, with fit/probe artifacts under `e482-results/`. It validates exact remote output paths/hashes, successful private-kernel status, actual T4/probe/fit continuity, receipts, OOM/resource guards, fit-only cohort, ordering, telemetry, checkpoint bytes and tensor shapes before a later explicit scoring path. `--verify-fit-only` returns before evaluation loading. No actual score or evaluation-row read has been performed with v2; it awaits parent source review.

Synthetic tests include a successful normalized metadata fixture and a tampered allowed-key value that must fail. Validation ran under a systemd cap of 1 CPU, 2 GiB memory, and no swap:

```sh
.venv/bin/ruff check research/campaign_2026-10-05/activation_t4_replication_v2/score_e482_v2.py research/campaign_2026-10-05/activation_t4_replication_v2/test_score_e482_v2.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/activation_t4_replication_v2/score_e482_v2.py research/campaign_2026-10-05/activation_t4_replication_v2/test_score_e482_v2.py
.venv/bin/python -m unittest research.campaign_2026-10-05.activation_t4_replication_v2.test_score_e482_v2 -v
```

All five synthetic tests pass. They read no evaluation data and do not score any model.
