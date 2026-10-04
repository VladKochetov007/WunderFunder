# E425 durable queue admission

Root admitted the exact queue at 2026-10-04T18:17:30.085289+00:00 after8CPU checks, actual frozen control identities, controller domain/FP16 diagnostic checks, and unchanged sealed worker code closures. E425 source SHA256 is `33208c5bc9f9ad5af6db8f9ee488e6739cb7d52f58a2d24a18439c64aec94097`.

`wnn-e425-sequence-v1.service` requires and starts after the6-hour oneshot `wnn-timing-window-v1.service`. The latter first verifies E426's4successful smoke/full-fit records and terminal controller. Each CPU timing is bounded1CPU/2GiB/1,800s; an unavailable quiet window defers after900s rather than blocking the next science fit. Actual timing worker failure stops the dependency; no blind restart.

The admitted E425 workers are a full20k two-batch smoke and the fixed4epoch EP3 full soft-routing fit, followed by reviewed full1,873-sequence FP32 export. CPU/GPU/RAM caps are enforced by the unchanged campaignrunner. No E425 scientific result is claimed at queue admission. This adds a scheduling dependency; it does not alter model/trainer math, evaluated row sets or checkpoint policy.
