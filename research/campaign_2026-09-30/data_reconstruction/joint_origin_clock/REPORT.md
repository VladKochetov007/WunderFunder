# Conditional age-coordinate relation

Frozen input-only probe on16 fresh TRAIN canonical components, disjoint from the
earlier small EDA studies recorded in selection.json. No targets, masks, VALID,
external data, models or predictions were read. Seven pairs and conditions were
fixed before feature reads; no fitted transform was used.

The primary condition requires steps>=99, previous rawfloat32 a5 and a6 both
equal the published rank-normal lower endpoint, and current a5/a6 both strictly
inside their endpoints. Primary rows per component were:
206,230,198,75,27,105,123,192,30,139,157,245,90,64,58,90.

| Fixed pair | Median component Spearman | Component range | Components with >=100 rows and same-sign rho>=.99 |
|---|---:|---:|---:|
| a5,a6 | approximately1 | approximately1 in all16 | 9/16 |
| a0,a5 | approximately1 | approximately1 in all16 | 9/16 |
| a0,a6 | approximately1 | approximately1 in all16 | 9/16 |
| a1,a5 | .616 | .421–.819 | 0/16 |
| a1,a6 | .616 | .421–.819 | 0/16 |
| a4,a5 | .706 | .424–.848 | 0/16 |
| a4,a6 | .706 | .424–.848 | 0/16 |

The first three pairs pass the fixed strong-order support gate. A floating-point
dot product yielded deviations from1 of <=4.5e-16; raw results are preserved.
The denominator remains all16 components; only9 have the required >=100 primary
rows. These are dependent within-component observations, not2,029 independent
events or16 independently established physical clock identities.

Together with the earlier quiet-growth behavior of a5/a6, the result supports
the hypothesis that a0 is a monotonic coordinate of the interval between rows,
and a5/a6 are two elapsed-age coordinates. It does **not** identify physical
timestamps, units, numerical transforms, additive time increments or source
identity. A clipped lower endpoint may represent a range of small ages rather
than a true zero; joint endpoints need not be simultaneous physical resets.
Alternative conditional generators can also produce identical rank order.

This differs from the earlier failed E274 shared inverse-clock fit: the present
probe has no fitted clock or noncollapse optimization. That failure remains
unchanged. An additive reconstruction using a0 as an increment coordinate would
require a new frozen fit, unseen-component falsifier and noncollapse/coverage
checks before any model use.

All-row fixed-pair controls, endpoint/refresh coincidence, distinct-value counts,
component records and hashes are saved in results.json. Independent raw-Parquet
reproduction is pending. This finding is not a predictive-quality gain.

Freeze/probe outer durations .502/1.567 seconds, exits0. Peak RSS1,101,692KiB;
one thread-pool worker, hard2GiB address-space and60-second CPU/wall limits per
foreground phase. Zero GPU. No durable or cgroup execution claim.

## Independent follow-up

Independent raw Arrow replay passed: membership/exclusions regenerated, counts and
all-row/primary correlations reproduced within3.33e-16, and independently computed
average-tie rank vectors were exactly identical for a0,a5,a6 in all16 components.
See ../../audit/joint_origin_clock_independent_v1.json and E285. This strengthens
the conditional-order observation; numeric time/additivity and physical resets
remain unproved. The proposed extra joint-quiet positive-increment diagnostic was
not included in the frozen primary probe and is not claimed here.
