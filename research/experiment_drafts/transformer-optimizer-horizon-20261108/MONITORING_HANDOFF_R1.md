# Live execution and publication handoff

Observed 2026-11-08 21:21 UTC. Four fresh fits are advancing on four B300
pairs; C1/203102 and C2/203102 wait for the second frozen wave. No full
four-epoch endpoint is complete at this observation.

The CPU plot service has written actual E1 PNG/SVG evidence for all four
active cells and E2 evidence for C2/203101. ROOT checked the image hashes
against their epoch manifests. Corresponding epoch checkpoint files exist;
the checkpoint hashes in the producer index remain producer declarations
until the final independent byte-binding comparison completes.

| Cell | E1 full-valid WP | E2 full-valid WP |
| --- | ---: | ---: |
| C0/203101 | 0.6675113021 | pending |
| C0/203102 | 0.6642413118 | pending |
| C1/203101 | 0.6418524563 | pending |
| C2/203101 | 0.6674788676 | 0.6717760957 |

These are trajectory diagnostics on exposed full official valid
(1,873 sequences / 3,528,273 scored rows), not the configured E4 grade.
Train-eval has a different population and mask; do not interpret its score
level relative to valid as a comparable generalization gap.

The [actual C2 E2 figure](evidence/plots/C2_muon512_a2_s203101-epoch02.png)
was downloaded without rerendering, hash-checked and visually inspected.
Both target WPs improve and both validation WMSEs decrease from E1 to E2;
two epochs do not establish a durable gain or absence of later overfitting.

## Durable monitoring

- Local dashboard: http://127.0.0.1:8769/ (HTTP/API checked).
- Native receiver: PID1818502, birth158011868; runs independently of SSH.
- Existing ROOT RPC: SSH PID8176, local execution session90678. Reuse it;
  no new connection is necessary for monitoring.
- Local dashboard PID8656, bridge PID12420, readout PID6784, publisher PID9232.
  The Windows processes have hidden windows and are independent of a CLI
  command returning; local mirroring/publication require this workstation
  and its network to remain available. Native training does not.
- Publisher uses immutable reporting source0471f41f40394f3006f2fe07858be06925ac1000.
  It posts hourly and upon a new canonical grade, final byte audit or failure.
  It uses producer/observer freshness rather than stale controller status.
- [PR72](https://github.com/VladKochetov007/WunderFunder/pull/72) and the
  [first automatic update](https://github.com/VladKochetov007/WunderFunder/pull/72#issuecomment-6727245866)
  were read back from GitHub and verified as authored by SemyonEpanov.

The original clock remains first-profile1794168764.8507051. Fit admission and
owned fit execution stop by1794193964.8507051; collection finishes by
1794197564.8507051. No clock reset, extra fit, source/config amendment or old
fit resume is authorized by this note. Initial total ETA remains provisional
3–4 hours from the first full-fit launch at20:37 UTC, including validation,
export and collection. Full-valid evaluation takes about3.5–6.5 minutes per
epoch in the current shared cohort; engineering update timing alone omits it.

## Recoverable evidence

Local root:
`C:/Users/admin/Documents/Codex/artifacts/transformer-optimizer-horizon-20261108`.

`ACTIVE_EXECUTION.json` records process handles; `live-dashboard/` retains
raw mirrors, publication journal, owner receipts and the bounded readout bundle.
`b300-rpc/results/030-current-epoch-health.json` binds the live progress and
actual image checks. Native epoch weights and evidence archives stay under
`/data/epanov_s/tabdata/runtime/transformer-optimizer-horizon-20261108-r1/runs`.
The receiver will export, score, bundle and compare all six exact E4 endpoints;
the publisher marks final byte audit pending until that comparison exists.

[The separate TL3 replay and architecture audit](https://github.com/VladKochetov007/WunderFunder/pull/73)
records the best cached standalone referenceWP0.6914165040 and fixed blend
0.6923774248, with its historical consumed-cache ancestry limitation. These
are exposed cached-artifact scores, not newly trained competitors or public gains.
