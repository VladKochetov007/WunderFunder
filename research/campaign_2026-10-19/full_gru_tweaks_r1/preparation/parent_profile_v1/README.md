# Parent profile preparation

Status: the local profile admission is bound to the accepted and exactly stopped CPU cache V2. The parent profile itself remains unlaunched; root owns the independent final review, fresh preflight, and only launch.

`REMOTE_SUBMISSION_TEMPLATE.py` retains a null cache binding and fails closed. `REMOTE_SUBMISSION.py` is the rendered body pinned to cache CPU V2 acceptance `88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7`, exact stop `286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4`, and metadata `03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3`.

The gate compares canonical train and valid fit groups separately, permits numeric ID overlap across those source namespaces, and rejects valid-fit/outer-valid overlap. It checks cache-output membership without treating fit-list order as target-index order. The canonical trainer validates exact target-index row order inside the bounded profile unit before profile updates. Preflight reads metadata and array stats only; it does not read cache payloads. A post-submit `systemctl show` timeout is saved to `LAUNCH.json` as an unknown outcome, with the existing no-retry policy.

This is an engineering profile: two normalization batches followed by six fit-only profile batches and 60 in-memory AdamW updates, with no evaluation, checkpoint, score, or quality claim. The superseded pre-fix template remains preserved under [pre_namespace_fix_unlaunched_v1](pre_namespace_fix_unlaunched_v1/README.md).

See [CLI.md](CLI.md), [LOCAL_CHECKS.json](LOCAL_CHECKS.json), and [PREPARATION.json](PREPARATION.json).
