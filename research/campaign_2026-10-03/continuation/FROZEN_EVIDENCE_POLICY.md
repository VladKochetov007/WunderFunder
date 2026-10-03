# Frozen evidence and amendments

Before indexing an agent report, require the agent's terminal completion and a
hash-bound final manifest, then verify the named bytes. A `STATE.json` frozen flag
while the agent is still completing final checks is insufficient. Subsequent work
uses a new scope/version; keep old report/spec bytes and hashes unchanged.

If a report is amended after indexing, retain the old ledger entry and reconstruct
its exact bytes from recorded edit history into a named historical-version folder.
Accept that recovery only if the original SHA256 matches. Record the amendment
as new evidence with both original and final bindings; do not rewrite the ledger
or describe reconstructed unmatched prose as preserved original evidence.

E-385 was indexed while the v2 critic was still finishing a moment-dtype check.
Its original NumPy-only report/spec were subsequently amended to include warm
optimizer-moment coercion. Both verdicts reject v2, so no GPU approval occurred.
The controller is preserving the original hash-bound bytes and adding the final
typed RNG/moment rejection separately. V3 is an isolated correction; default
CUDA reproduction still needs independent final review and an actual controller
receipt binding the approved candidate, runtime and live terminal queue evidence.
