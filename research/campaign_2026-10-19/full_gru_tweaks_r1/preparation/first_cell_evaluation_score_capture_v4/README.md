# First-cell score result capture V4

This packet preserves the score-result reader's V3 identity, receipt, unit, and
strict-terminal gates while bounding metadata observation before accumulation.
The V4 template caps subprocess stdout/stderr as bytes arrive, enforces the
existing per-command deadlines, reads cgroup and `/proc` metadata with a
no-follow byte ceiling, and limits cgroup directory traversal, per-directory
entries, per-file contents, and total process-list bytes. An empty or root
unit `ControlGroup` still produces no unit-cgroup snapshot; it never falls back
to reading host-root cgroups.

The source is a conditional reader template, not an actual score capture. Its
five future score-admission literals remain unbound, and its score unit,
export unit, result path, scorer pins, receipt schemas, and result-read gate
remain those from V3. It cannot establish score completion or quality before a
later independently reviewed binding and exact terminal evidence.

The local fixture compares the rest of V4's AST against frozen V3 and exercises
bounded command output/timeouts, temporary cgroup trees, symlink rejection,
per-file and aggregate byte caps, and empty/root cgroup handling. It uses only
synthetic local metadata and standard-library subprocesses; it does not query
systemd, NVIDIA tools, project source/data, models, or B.

The predecessor remains frozen at 20,781 bytes, SHA-256
`44ba4410a20805e1a468c97a5c4cf4577bece6a4cd8092ab1da9350b86d678ec`. V4 is
26,354 bytes, SHA-256
`88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6`; its
focused fixture is 7,770 bytes, SHA-256
`59da8db634549cef182234b5fc230053e8c7f52013024e83bd1eb36358ef2783`.

See [LOCAL_CHECKS.md](LOCAL_CHECKS.md) for the bounded local verification
record and [the independent source review](INDEPENDENT_SOURCE_REVIEW_V4.json)
when that review is complete.
