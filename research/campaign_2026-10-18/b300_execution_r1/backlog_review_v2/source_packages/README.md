# Original PR source packages

These are code-only deployment bundles extracted byte-for-byte from the pinned local Git objects. Each tar.xz contains the complete `src/` tree, the listed trainer/runner and data helpers, the exact recipe and proposal documents, and the four committed fold manifests. It contains no datasets, target/cache arrays, checkpoints, or weights. The adjacent JSON manifest records every path's Git blob OID, SHA256, size, source commit, and archive SHA256.

| Original proposal | Commit | Archive | SHA256 | Files |
|---|---|---|---|---:|
| PR15 causal geometry/discrete | `e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e` | [tar.xz](pr15_geometry_e238fc5c82ac.tar.xz) | `29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985` | 172 |
| PR17 Kalman teacher/distillation | `588b93f9c2ad7e53aac7b1e2a39f30c32047ea22` | [tar.xz](pr17_kalman_588b93f9c2ad.tar.xz) | `236ed4277bac52266d0e5c3eb8a842d77d59a1950f36f13e419d61f58a3dc3ce` | 164 |

See [PR15 operator notes](pr15_geometry_README.md), [PR17 operator notes](pr17_kalman_README.md), and the [PR15 manifest](pr15_geometry_manifest.json) / [PR17 manifest](pr17_kalman_manifest.json). The manifests are sidecars and are not inserted into the source archives, so the archives contain only files present in their source Git commits.

The producer verified each archive by reopening it, rejecting unsafe or non-regular entries, and comparing every extracted member against the manifest bytes. Text blobs were LF-only; tar entries have normalized ownership and timestamps. Profiles and fits were not run while preparing these packages.
