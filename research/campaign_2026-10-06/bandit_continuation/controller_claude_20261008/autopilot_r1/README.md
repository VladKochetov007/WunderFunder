# Local finite-queue autopilot

This release supervises one exact local warm-queue successor after the existing
v3 attempt drains. It does not generate recipes, retry a fit, call an API, rent
capacity, or launch a remote worker. The queue contains only frozen seed99,
seed9, Z4 and Z5 specs. Root authorization is required in the contract, policy,
review receipt and installed service hash. The checked-in `*_DRAFT.json` files
are deliberately unauthorized and cannot start this service.

The user service runs Python directly inside its bounded systemd unit, not
through the short-lived support-command wrapper. Its cgroup limits bound the
long-lived daemon without a wrapper runtime timeout. It has its own lock,
durable heartbeat/state/journal, stop barrier, single-launch dispatcher receipt,
bounded terminal collection retries and artifact hashes. The finite worker
stays under the existing campaign resource guardian. The service remains alive
while a healthy v3/v4 worker drains, then
exits on terminal collection, queue exhaustion, policy expiry, or a review
error. Systemd restart cannot duplicate a launch: the dispatcher attempt receipt
is durable and uncertain launches fail closed. Existing specs declare no
`resume_state_path`; an interrupted fit is not automatically resumed or
transplanted and needs a new root-reviewed attempt if recovery is warranted.

## Root installation and authorization

The root controller reviews the exact source, pinned specs, finite queue,
`LOCAL_V4_ROOT_CONTRACT.json`, and policy. Before enabling the contract, verify
the live v3 service and worker are still healthy and that the three expected
pending hashes have not changed. The autopilot waits for v3's terminal receipt,
no running jobs or live worker, verified seed99 outputs, exact pending recipe
hashes, free disk, current parent quota/RAM and observed GPU headroom before it
mints the dispatcher's short-lived admission. The v4 worker also receives
`LOCAL_V4_ALLOWED_SPECS.json` and hashes each pending spec immediately before
claiming it; any changed or unknown pending file is left untouched and recorded
as `admission_hold.json`. The autopilot rechecks the allowlist and current
pending hashes before dispatch as well.

After reviewing and finalizing the root policy/review files, update the service
unit paths if root renames those files. Create the root-only service environment
file containing the exact authorized review-file digest:

```sh
printf 'AUTOPILOT_REVIEW_SHA256=%s\n' '<sha256-of-reviewed-review-receipt>' > research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/autopilot_r1/LOCAL_V4_SERVICE.env
chmod 600 research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/autopilot_r1/LOCAL_V4_SERVICE.env
install -m 644 research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/autopilot_r1/systemd/wnn-research-autopilot-local-v4.service ~/.config/systemd/user/wnn-research-autopilot-local-v4.service
systemctl --user daemon-reload
systemctl --user enable --now wnn-research-autopilot-local-v4.service
```

Installation and enablement are root actions after review. The unit is not
started by this code change. User lingering must remain enabled for it to
survive a Codex/terminal exit. The unit is bounded to 1 CPU and 1 GB RAM inside
`wnn-campaign.slice`; v4 itself remains bounded to the reviewed 2 CPU, 9 GB RAM,
5 GB observed VRAM, 6-hour fit profile.

## Operations and evidence

Runtime receipts are written under this directory's ignored `runtime/` folder
on the project SSD. `LOCAL_V4_RESULTS.json` records each frozen spec, worker
terminal status, steps, output/checkpoint hashes, queue result/log hashes and
resource peaks. `LOCAL_V4_COST.json` records local elapsed/quota time and zero
provider spend. These are training artifacts, not an evaluation: this
autopilot does not create predictions or claim a metric. The independent grader
owns compatible evaluation when required predictions are available.

Inspect service health and its heartbeat before interpreting a process state:

```sh
systemctl --user status wnn-research-autopilot-local-v4.service
journalctl --user -u wnn-research-autopilot-local-v4.service
```

Use the script's `status` command with the same runtime path to inspect the
durable state file. A graceful stop writes both `STOP.json` and the dispatcher's
`STOP_ADMISSION.json`, then drains a healthy worker. Do not kill the worker to
accelerate a stop. Resuming requires a new root review bound to the exact stop
receipt; use the script's `resume` command with that review digest before
starting the service again. A changed queue, policy, contract, source, or review
must be re-reviewed and re-pinned.
