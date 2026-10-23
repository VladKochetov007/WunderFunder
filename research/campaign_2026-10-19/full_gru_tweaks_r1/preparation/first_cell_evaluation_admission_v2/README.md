# First-cell export admission V2

V2 makes one narrow correction to the frozen V1 admission body: GPU process command lines are inspected only for processes whose observed UID equals the current UID. The campaign-cgroup ownership check still applies to every GPU process, and the admission continues to record UID/cgroup identity and take no action on foreign processes.

V1 remains preserved at `../first_cell_evaluation_admission_v1/`. Its independent source review `REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json` rejected that exact body because it read foreign `/proc/<pid>/cmdline`; V2's source binding pins that body, binding, and negative review. `V2_INVERSE_BYTE_PROOF.json` shows that replacing only V2's `gpu_owners` function with the V1 function reconstructs the exact frozen V1 rendered body byte-for-byte.

`test_gpu_owner_boundary_v2.py` executes only the three source statements that make and enforce the GPU ownership decision, with a spy for `/proc/<pid>/cmdline`. It checks that foreign UID command text is never read, while campaign-cgroup membership still rejects any UID and a same-UID project command is detected. It does not import or run the full admission body.

The body retains V1's fixed payloads, exact source-review input, original fit receipt handling, resource gates, and one-attempt reply contract. This packet has not been staged or executed on B. An independent review of the exact V2 body and binding is required before any B action.
