# ADR0015: flexible credit-bounded parallel compute

Status: accepted, 2026-10-08 Kyiv. Supersedes ADR0013's $4 first-cohort and evidence-release spending gates only; ownership, independent training, bounded execution and teardown remain binding.

## Basis

Latest user instruction: “Actually feel free to use so much money as needed to iterate faster. but reasonable amount, not too much of course. make it efficient”. This authorizes faster useful parallel execution without waiting for the original first-cohort evidence release. It does not define an unlimited numerical budget or authorize a credit top-up.

## Decision

Use the existing account credit flexibly, retaining $20 cumulative all-in as a conservative operating ceiling until a further explicit numerical allocation. Do not top up or subscribe. Every cohort has a reviewed finite all-in reservation and hard teardown deadline; ledger committed reservations plus settled spend must stay inside this ceiling and verified available credit must cover admission. Unsettled destroyed workers retain conservative reservations until bill reconciliation; credit decrease is provisional spending evidence, not an invented itemized invoice.

Add an explicitly root-authorized parallel cohort path bound to this ADR's exact file hash. Preserve the original first-cohort/evidence-release paths for historical receipts; do not reinterpret or alter their frozen records. A new scientific result is not required to unlock the new path. A runnable scientifically valid recipe, exact source/inputs, representative resource admission, monitored delivery, creation ownership and cost headroom still are.

Prefer diverse useful approaches in parallel and common verified staging over repeated seeds or independent copies of the same loading bottleneck. A more expensive GPU is eligible when measured cost per useful result and turnaround improve. Terminate idle, bugged or inefficient campaign-owned workers when a diagnosed successor has better expected value. Preserve terminal evidence and avoid restarting merely because an observation timed out.

## Validation

Tests must reject a missing/mismatched authority hash, unbound authority source and aggregate overspend; allow a reviewed parallel cohort without a scientific release receipt; retain all existing ownership, uncertain-create, cost and teardown tests. Scientific actors cannot edit budget authority or authorize their own launch.
