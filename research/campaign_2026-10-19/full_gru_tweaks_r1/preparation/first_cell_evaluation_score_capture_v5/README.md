# First-cell score capture V5

V5 is a conditional, bounded metadata reader for the prospective score-v2
unit. It is an exact V4 source fork with three assignment changes only: the
score unit name, its private score directory (`score_v2`), and the scorer-core
pin moved to frozen V6. The export unit, export/result paths, reader behavior,
resource bounds, and all other scorer pins remain unchanged. The five future
score admission/launch byte, digest, and invocation literals are still
unbound, so the template cannot query systemd or read a score result yet.

The V4 predecessor and the first score attempt remain historical. That attempt
failed during the registered-alias metadata gate before reading labels or
prediction arrays; it is not polled, stopped, or retried by this packet. The
new fixture is AST-only and checks the exact three-assignment delta, score-v2
namespace, V6 core pin, and unbound future receipt literals. It does not import
or call the reader's `main`, query a unit, read a receipt/result, or access any
payload.

`SCORE_CAPTURE_V5_REBASE_PROOF.json` records the source-only inverse AST proof.
The reader is pending independent source review and has not been bound to an
actual score admission. It makes no score-quality claim.

See [LOCAL_CHECKS.md](LOCAL_CHECKS.md) for the focused verification record.
