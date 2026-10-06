# E-497 independent prototype review

Review date: 2026-10-06. This is an independent review of the isolated CPU
prototype and a non-executable resource plan. No trainer integration,
resource probe, fit, export, score, or provider launch is authorized by this
review.

## Verdict

GO for registration as an isolated prototype only. The exact implementation
and tests support the limited causal/state-storage claims in the design. The
packet correctly marks trainer integration, native export, and resource
profiling unsupported. This is not a GO for integration or an experiment run.

## Reviewed identity and evidence

| File | SHA-256 |
|---|---|
| `residual_memory/DESIGN.md` | `9c4b3e0b74b15b8cb130ba3a29944b7fc4ad0234f03f619c2189bc305455aedd` |
| `residual_memory/model.py` | `e69437dab72caf71b57cffad40efea6096a21e50fca7e5429db12f241f222eca` |
| `residual_memory/test_model.py` | `a859d747a34b892d89fb20c41aa8357b2374127346ff23d97ba253759a9c134f` |
| `residual_memory/resource_probe_plan.json` | `892093b93e796b48f6896f94efcffc952c9c1aab333dd15ecc0fd54bc4052c36` |
| `residual_memory/ledger_packet.json` | `0c0481b502843b75647601b8c9c33b76ae06c9cbcd0dce584109d959799d880b` |

The packet records 12 CPU tests, 0 failures, 0 errors, under a one-CPU,
2,147,483,648-byte host cap with GPU disabled. The receipt SHA-256 is
`4bb06265fea40cef3818a28ac5d59978885258ded78867c576d5279b17c905f7`.

## Findings and limits

The API accepts only base predictions, current-row write features, and optional
carried state. It reads the existing matrix before writing the current row;
the event/quiet gate also comes from a configured current-input column. There
is no label, future-row, or external event-mask input. The delta and fixed
0.999 additive updates match the design equations. State reset via `None` and
explicit detachment are represented in the API; tests cover prefix causality,
uneven chunk value/state parity, initialization identity, and gradient
opening after the zero residual head moves.

The asserted bound is persistent state **storage** of `B*K*V` float32 values
(32,768 bytes at B32/K16/V16). It is not a measured or proved global numeric
range bound, nor a bound on the full autograd activation tape for a long
un-detached sequence. The current design describes the storage claim
precisely; future integration should preserve that distinction. The generic
constructor also accepts `fixed_decay=1.0` for the additive arm, which would
remove forgetting; the registered plan fixes this arm at 0.999. Keep that
unregistered setting out of the experiment, or reject/document it before a
future public integration.

The resource plan is deliberately descriptive and has no executable argv. The
design explicitly requires a reviewed causal wrapper, bound event feature,
reset/detach integration, a passing resource profile, and a separate
controller admission before science fits. No fit/probe/score claim follows
from the CPU tests.
