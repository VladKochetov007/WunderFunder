# Explicit binder-v2 arithmetic amendment

Original binder, original prefit spread design and original11-case proof are unchanged. `spread_binding_amendment_v2.json` binds their SHA256 identities, the separate v2 source and the new18-rejection proof. Root authorized this critic-only correction before any score and reviews it before binding.

The frozen trainer uses `drop_last=True`; continuation11,940/batch32 has373 batches and3,730 attempts, FT1,405/batch16 has87 batches/870 attempts. V1 incorrectly used ceiling. V2 derives batch count from recorded loader/sampler configuration and validates it against frozen arguments and membership. It rejects a false/nonboolean drop_last for this source, wrong batch size, sampler class/replacement/count, or ceiling counts with true drop_last. A separate synthetic false fixture verifies ceiling arithmetic5/batch2=>3batches/30attempts. The known-answer matched fixture now has a nondivisible dataset.

All original exact source/argv, lineage, cache, selected-epoch, normalization, sampler/RNG, completed/skipped update and EMA parity gates remain. The generic scorer is unchanged. A difference in successful updates between current/proxy invalidates the mechanism comparison even when complete-batch counts match.

After both jobs finish and root accepts the correction, bind using:

```bash
.venv/bin/python research/campaign_2026-09-30/audit/build_screen_specs_v2.py bind \
  --design research/campaign_2026-09-30/screens/spread_v1/design.json \
  --amendment research/campaign_2026-09-30/audit/spread_binding_amendment_v2.json \
  --spec research/campaign_2026-09-30/screens/spread_v1/spec_v2.json
```

Use the campaign CPU scope limits. V2 prohibits new freeze operations and accepts the original design only through this explicit amendment. Generated specs retain original preregistration identity and append amendment, original-builder and actual amended-builder identities. No candidate scoring was performed during the correction.
