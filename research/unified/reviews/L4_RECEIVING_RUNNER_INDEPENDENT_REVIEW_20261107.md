# Independent static review of the bounded L4 receiving runner

Reviewer: PR67 receiving execution agent, 7 November 2026. The compact-model
builder implemented the runner. This reviewer read the six helper files and the
canonical ownership/budget/profile/terminal functions they call, and independently
ran seven fast synthetic gate tests. No staging, SSH, GPU, real process-group
experiment, data access or model execution was performed for this runner review.

Disposition after the two fixes below: **STATIC READY for ROOT-controlled
receiving use**. Two GRUs with measured profiles can advance independently once
ROOT verifies and approves their exact native receipts. This does not admit the
Mamba backend, other profiles or uncompleted scientific endpoints.

The helper scope is the finite eight-cell architecture screen under the actual
first-profile clock 1794071266.717658: 24 hours total with a final one-hour
collection reserve. No automatic profile approval, retry, extra scientific fit,
resumption, rental or source modification is implemented. Existing canonical
attempt ledgers and exact PID/birth/PGID ownership helpers are reused.

Verified static boundaries:

- The operator-supplied panel digest, trusted 620-file source inventory, staged
  source bytes and extra-importable-Python check bind the actual trainer source.
  The panel separately hashes every receiving helper.
- The runtime config derivative permits B16 plus the declared monitoring cadence
  and evaluation batch, preserving the 12-epoch label-only recipe. Native profile
  checks bind normalized config, source, full data digest, exact train/evaluation
  membership, fitting-only normalizer and physical L4 UUID.
- The approved receipt is a separate ROOT-authored copy. Every original field
  except `approved` is preserved, followed by canonical full identity validation.
- Existing ROOT profile ownership receipts are adopted without relaunch. The two
  prior profiles enter the admitted total profile count; later profiles and fits
  reserve independent bounded attempts in the existing ledger.
- Each cell advances independently. Mamba waits for a source/device-bound passing
  actual CUDA oracle, while ready GRUs do not wait for other families.
- Resource/deadline checks retain the collection reserve, output/free-space/RSS
  headroom and worker identity. Termination is limited to receipt-proven commands
  whose PID, kernel birth and process-group identity still match.
- Every complete validation event must cover 1,873 sequences and 3,528,273 rows,
  have finite metrics, and advance the epoch exactly once. Final collection needs
  all twelve reports plus a complete fixed-final terminal/checkpoint identity.

Two concrete problems were corrected before this disposition:

1. The initial relay called `terminate_owned_process` on its own process group
   with the default SIGTERM handler. That could kill the leader before the helper
   reached group SIGKILL. The fixed relay spawns its observer first, then ignores
   SIGTERM only in the receipt-bound leader. Children retain normal handlers; the
   leader stays alive so canonical failure/deadline cleanup can reach exact-group
   KILL. The new mocked ordering test independently passed.
2. ROOT independently found a forced `CUBLAS_WORKSPACE_CONFIG=:4096:8` that differed
   from the initial GRU profiles' absent setting. The final worker environment
   removes workspace/emulation/rank variables and sets both TF32 overrides to zero.
   The environment parity test independently passed. Native fitting still compares
   its complete profile identity rather than relying on this static assertion.

Independent checks: **7/7 selftests passed in 0.084 seconds**; Ruff format/check
passed for all six helper files. Reviewed raw working-file SHA256 values:

| File | SHA256 |
|---|---|
| `build_panel.py` | `2be0d1c856baea42d4fe0fe94f15abd15ca79b6b86cb56a274558e1b9aafb548` |
| `observed_worker.py` | `0032f1d44aec5d2f1d8c4ee3d2c1e91c3cd4c68075d50275febba68dbe568854` |
| `receiving_checks.py` | `184dce34483f3927756d16dab80882b4802f148b049d46cda9ff8ae490d9061e` |
| `receiving_resources.py` | `af40f3169ef9c93178d00dd855f42b86ec21d3059c81579e3392e672a7df99fb` |
| `run_cohort.py` | `bb3763bde93301367871d5eab6588b833663e2d0f597bf106c0660e2af62cb77` |
| `selftest.py` | `b2a3936ad3673ed74d9c26da23d9ebdbbef799a15ce410de45f44312e568062a` |

These files live in the builder worktree's
`research/experiment_drafts/architecture-execution-20261107/l4/receiving/`.
ROOT must freeze and verify the actual transferred helper bytes when creating
the panel. This review does not replace native identity or profile admission.
Triton SSD precision inside the genuine Mamba backend remains a separate
inspection question: Torch TF32 flags alone do not prove IEEE dot arithmetic in
custom kernels. That question does not block the independently ready GRUs.
