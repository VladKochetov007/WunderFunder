# Vast campaign-owned useful-time utilization (R1)

As of 2026-10-08T14:39:58Z, this report includes only campaign instances with durable ownership and verified-destruction evidence. Reservations, holds and unproven machines are excluded.

Five owned lifecycles contribute at least **11408.56 seconds** of documented instance time. Several starts are first-observed or creation-response timestamps, so the combined denominator is a lower bound. H230 V6R1 accounts for exactly **0 of 1,806 seconds** in recorded profile/fit/export phases; that startup failure ended before worker launch.

The only timed successful model phases found are the H217 A3 TCN fit (**823.97s**) and valid export (**22.27s**), totaling **846.24s**. Their recorded phase wall-time is **7.42%** of the documented lower-bound lifetime denominator. The TCN profile passed, but its elapsed duration is missing; it is unknown, not zero. This ratio covers only the two timed phases and is not a total useful-work bound.

The fit timer wraps the trainer subprocess. Data loading/setup within it is not separated, so this cannot be reported as pure GPU-active time; no loader subtraction is supported by the current receipt. Startup, upload, hashing, staging, idle and collection time are not added to the numerator.

| Owned instance | Documented interval | Seconds | Recorded model phases | Limitation |
|---|---|---:|---|---|
| 54725432 | 2026-10-07T22:08:13Z → 2026-10-07T22:40:30Z | 1937.00 | none; scientific_outcome=none | Input staging occurred; no fit/evaluation began. |
| 54734753 | 2026-10-08T00:22:30.127205+00:00 → 2026-10-08T00:50:49+00:00 | 1698.87 | no recorded profile/fit/export; terminal result.phases is empty | Provider creation timestamp was not located; interval is a lower bound. A3D ended as execution failure. |
| 54755152 | 2026-10-08T01:56:51.315003+00:00 → 2026-10-08T01:57:31+00:00 | 39.68 | prefetch only; no fit/profile/export recorded | Start is first observed creation status, not provider create timestamp; interval is a lower bound. |
| 54755890 | 2026-10-08T02:01:06+00:00 → 2026-10-08T03:39:53+00:00 | 5927.00 | TCN profile passed (duration missing); fit and valid export passed | Creation response may follow provider creation; interval is a lower bound. Profile timing is not reported. |
| 54852864 | receipt elapsed through 2026-10-08T14:05:51+00:00 | 1806.00 | 0 recorded; worker_started_at_utc=null, worker_launch_status=null | Startup failed before worker launch; no profile/fit/export phase. Exact 1,806-second lifetime is reported by terminal receipt. |

H217 A3D has only a first-observed staging time, not an exact creation timestamp. The H217 A3 TCN profile duration is absent. These evidence gaps remain explicit. SHA-256-bound receipt references are in the JSON companion.
