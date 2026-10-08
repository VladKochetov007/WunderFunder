# Local v4 autopilot allowlist — independent review R1

**Verdict: PASS for the frozen local finite-queue supervisor and worker allowlist implementation.** This review is bound to commit `d6a94726f8eed8509fb9b22128d794f759ba8af7`. The checked-in contract, policy draft, and root-review draft remain unauthorized; root must produce the final admission before the service can launch.

## Frozen source identity

- `scripts/research_autopilot.py`: `56300f0b9d475362f526286ee8638f934f330626b775431ea8edfc33a28b6e99`
- `scripts/warm_queue_worker.py`: `2f19e3393a9cc8f3f24da52dba5d32a8e5dc607a7addd97b654c50440c72f843`
- `LOCAL_V4_ALLOWED_SPECS.json`: `a1102d871509dc2f075bbecc49d827c54828d8916e30652b0e9acb092675db42`
- finite queue: `6afd1c7f422f257aad772342eb1ed3c091adbb7bc151a5025b8a3ade15124f7c`
- root review draft: `b40d55c6e5a6e0b53e3dd22cd906c40631cb9f6990bbfa26e43a21868a5398af`

The allowlist contains the three exact v4 pending run IDs and hashes. Autopilot load checks require it to equal the queue's `pending_spec_hashes`, require the worker command to pass that exact in-repository manifest, and verify its hash through both review and job source maps. The dispatcher admission rechecks those source hashes. The resource guardian's source manifest also hashes `.json` command arguments, so it pins the allowlist through its pre-start verification. The worker checks every pending spec against the allowlist before selection and rereads/rechecks the claimed bytes before parsing or execution; an unknown or changed spec stays in pending and produces an `admission_hold.json` record. The new option defaults to `None`, preserving the existing v3 worker behavior.

The frozen queue changes only by passing the manifest and updating its source hashes. The four recipe files and science flags are unchanged. The service unit runs the watcher directly in its own bounded systemd cgroup, avoiding the old 900-second support-wrapper lifetime. STOP handling closes dispatcher admission under the same launch lock for CLI stops; signal stops write the barrier synchronously, with a stop that arrives after the dispatch gate treated as an already in-flight admission. Existing tests cover both orderings.

## Validation

- Test command under the shared support lock and 1 CPU / 1 GB / 180-second wrapper: `flock -w 10 /tmp/wnn-codex-support.lock env WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=180s /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -m pytest -q tests/test_research_autopilot.py tests/test_warm_queue_worker.py` — **41 passed**.
- Ruff check and format check passed for the two scripts and their tests.
- Python 3.11 `py_compile` passed for those four files.
- `systemd-analyze --user verify` passed for the frozen service unit.
- The static checked-in-source test confirms manifest, contract, queue, job source hashes, root-review source hashes, and worker argument all agree; policy, review, and contract drafts remain `authorized: false`.
- The campaign slice was at 12.80 GB of its 20 GB maximum before the bounded test run, below the 14 GB deferral threshold. No verification exceeded the 1 GB support cap.

No training fit, warm-queue worker, systemd installation, provider call, or provider mutation was performed during this review. The worker tests use synthetic local specs only.

## Operational limitation

If an unreviewed pending spec appears, the worker safely holds the queue and waits for the configured 300-second idle exit. The hold remains at `runtime/warm_queue_local_v1/admission_hold.json`; the supervisor will fail closed if terminal collection lacks expected run results. That adds review latency but does not permit an unreviewed fit.
