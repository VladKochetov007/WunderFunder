# Input-only additional-feature relations: no simple decoder found

Fixed before data inspection: 36 encoded-value/CDF-coordinate aggregates and causal refresh-clock proxies; one proxy for each of `a0`–`a4` selected by absolute Spearman correlation on16TRAINcomponents, then scored on16differentcomponents (320,000evaluation rows). No targets, masks, VALID or external data were read. Components were selected through the canonical root split manifest; exclusions and their exact scope are in selection.json. This is component separation within this diagnostic, not an untouched campaign/public test.

| Feature | Fixed proxy | Fit rho | Evaluation rho | Median within-component rho | Components meeting0.95 |
|---|---|---:|---:|---:|---:|
| a0 | i1_volume_row_age | -0.4051 | -0.3483 | -0.3501 | 0/16 |
| a1 | i0_trade_price_z_sum | -0.0851 | -0.0736 | -0.0936 | 0/16 |
| a2 | i1_volume_row_age | -0.6269 | -0.3878 | -0.4186 | 0/16 |
| a3 | i1_volume_row_age | -0.5565 | -0.2477 | -0.3578 | 0/16 |
| a4 | i0_price_row_age | 0.2052 | 0.1797 | 0.1871 | 0/16 |

None meets the frozen support criterion of sign-consistent correlation>=0.95 on at least8components. Age/activity correlations remain descriptive. No physical quantity/time units, deterministic feature identity, target mechanism or score gain was recovered. Normal-CDF sums/fractions are proxies in encoded space, not recovered volume quantities. Bid/ask membership uses the published columns; no ordering within a side is assumed.

The fitting/evaluation commands took about2.86s/2.55s and returned0. Peak RSS is recorded perphase (~1.3GiB); each foreground child had hard2GiBaddress-space and60sCPU/wall limits, one thread perlibrary. These are process resource limits, not a systemd scope or durable job. The normal systemd wrapper was unavailable under the changed runtime permissions. No GPU time used.

Important outputs: preregistration.json, selection.json, fit.json, results.json, perphase command/resource JSON. Independent reproduction remains pending; do not promote a correlation as a decoded feature.
