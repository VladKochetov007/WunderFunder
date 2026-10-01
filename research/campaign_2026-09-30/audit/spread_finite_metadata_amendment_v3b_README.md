# Finite metadata supplement before scoring

The approved supplemental wrapper preserves frozen v1/v2/v3 binders and their tests. It rejects boolean, nonnumeric, NaN, infinity and negative duration/resource values, checks integer byte counts and stage exits, exact timeout allowances, cumulative stage timing, and the original 720-second combined fragment cap. No statistical design, threshold, update equality or lineage rule changes.

42 synthetic rejection cases and known answers passed. The actual current pipeline and proxy continuation prefix have finite durations and valid resources. Current elapsed is 255.174646216 seconds; proxy prefix is 252.993773831 seconds. Both peaked at 4,183,818,240 observed GPU bytes. The frozen 467-second resume reservation retains the original combined cap. The resume job is absent, so binding remains blocked pending its successful FT and export suffix. No target or prediction values were opened.

Use the supplemental wrapper after the missing fragment lands:

```sh
WNN_SLICE=wnn-campaign.slice .venv/bin/python research/campaign_2026-09-30/audit/strict_fragment_metadata.py bind \
  --design research/campaign_2026-09-30/audit/spread_operational_design_v3.json \
  --amendment research/campaign_2026-09-30/audit/spread_binding_amendment_v3.json \
  --finite-amendment research/campaign_2026-09-30/audit/spread_finite_metadata_amendment_v3b.json \
  --output research/campaign_2026-09-30/screens/spread_v1/spec_v3b.json
```

The two metadata probes used hard 2 GiB address-space limits, affinity to one CPU and one library thread. Observed peak RSS was 68,680 KiB. This fallback does not establish an individual 2 GiB cgroup limit.
