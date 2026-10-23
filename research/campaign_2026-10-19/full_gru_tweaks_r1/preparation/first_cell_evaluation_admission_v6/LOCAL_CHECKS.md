# V6 local checks

The exact exporter V4 / scorer V5 source-pair review was verified from its saved JSON and all 12 `reviewed_files` pins. The V6 renderer consumed that actual schema, joined the exporter/binding/manifest pins, and produced one local body candidate:

```text
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v6/render_admission_remote_v6.py
```

It produced `REMOTE_SUBMISSION_V6.py` at 249,896 B / `d963b7df0349fcef3ae4804502d8b4b9430ff667132bd1b91552e73801807d10` and binding `ADMISSION_SOURCE_BINDING_V6.json` at 22,263 B / `1175c095fa21b0cf2d26544ea9817f943fb5270b6539813a7b78353220c645c9`.

The changed V6 review-contract fixture passed 4 focused tests; the new historical-failure metadata fixture passed 1 focused test. The inverse proof passed, establishing exact byte recovery of frozen V5 after replacing only the five payload assignment and reviewed V4 exporter/binding identities, worker unit, and evidence root. Ruff format and check passed for the V6 renderer, tests, and inverse proof using the literal maintenance environment.

These checks read only source text, source review metadata, small accepted receipt metadata, and saved root transport/startup captures. They did not query B or old units, import project code, or read model, checkpoint, cache, registered data, prediction, or score payloads. No unchanged receipt, model, source-tree, or cold suites were replayed. The V6 body remains subject to an independent exact-body source review before root considers any admission.
