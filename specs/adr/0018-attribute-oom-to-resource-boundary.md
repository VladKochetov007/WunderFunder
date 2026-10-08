# ADR0018: attribute OOM events to the resource boundary

Status: accepted, 2026-10-08. Complements ADR0006 and ADR0017.

Two separately bounded support helpers OOMed inside their own child scopes. The local H230 trainer had no OOM, but the controller treated the parent hierarchical counters as an aggregate-limit violation and terminated it. Both failed attempts remain infrastructure evidence, not scientific negatives.

Read the campaign parent’s `memory.events.local` for events at its own aggregate limit. Continue to read the fit worker’s hierarchical `memory.events` so loader/worker descendant OOMs stop that attempt. Record the parent’s hierarchical counters separately for diagnosis; a sibling-only event does not terminate an unaffected fit. Missing local counters reject admission. Aggregate RAM, CPU, swap, VRAM and worker guards stay unchanged. Linux documents this distinction in [cgroup v2 memory events](https://docs.kernel.org/admin-guide/cgroup-v2.html).

Root wrote the fix; the independent Luna model/data-forensics reviewer inspected baseline, runtime and final guards and passed the focused attribution change. Root focused resource/CPU tests passed 26 cases; reviewer independently ran 23 resource tests. These are simulated observer tests, not a live kernel fault injection. Nine cases cover sibling, worker and aggregate OOM/kill/group-kill events; missing local counters fail closed.

The reviewer identified a pre-existing limitation: persistent unknown runtime/final observations can leave a successful exit unverified. This focused fix does not claim to resolve it. Preserve unknown observation evidence and audit it before result promotion. New attempts bind the corrected observer source and fresh resource admission. Historical frozen attempt receipts remain unchanged.
