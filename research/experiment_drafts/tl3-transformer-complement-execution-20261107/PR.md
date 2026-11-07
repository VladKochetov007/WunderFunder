# Receive PR67 as four matched fixed-budget two-B300 fits

The frozen PR67 handoff requires eight GPUs per fit, assumes Git/uv on the
receiver, and has no per-epoch official validation report. This separately
identified receiving source uses four disjoint two-B300 pairs, batch 4/rank and
four accumulated chunks. It preserves 25,820 updates/epoch and each fixed final
30-epoch endpoint while reporting the same 90/10 composition on all official
valid members after every epoch. No validation score selects or stops a fit.

The packet preserves frozen recipes/configs and adds sender-bound source capsule
verification, exact legacy export support without Git, explicit homogeneous
device comparison, and a fix for the composed-export integrity callback returning
`None`. [README](README.md) gives authority, identities, dependency order, commands,
uncertainty and pending checks. Different UUID pairs are an explicit exploratory
hardware confound; this is a new recipe identity rather than measured eight-rank
parity. Four new fits, two per arm, no new teacher/student fits or submission.

The explicit source-r3 extension adds independently authored process/resource
observation and phase/successful-finite-update heartbeats. It preserves source-r2
and requires fresh admission. Observation never selects or interrupts a fit.

Validation: local Ruff, Bash syntax, training structure and 59 focused CPU checks;
independent r2 review passed 53 CPU checks. Receiver acceptance, baseline replay,
native precision/DDP/Muon checks, real
profiles and all scientific endpoints remain pending. Root exclusively owns
remote execution; an independent reviewer must approve the actual production
profile copy for each cell before fitting. No performance result is claimed.
