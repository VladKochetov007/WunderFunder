# F1 DataSphere one-shot controller

`f1_fit_controller.py` is the follow-on controller for the exact paired fit
`computer-vision-homework-rank01-f1-s1001-fit`. It reuses the profile
controller's safe CLI runner, exact-ID state machine, cancellation, and
private atomic receipts without changing the fit profile controller. Its
submission state is separate and fail-closed: an uncertain execute result or
an unknown job ID never causes resubmission or cancellation by name.

`f1-fit-admission.template.json` is deliberately not runnable. A private copy
must be sealed only after checking the exact packet hashes, finite measured
runtime allocations, estimate and maximum cost, and fresh balance/rate
evidence. `max_cost_minor_units` is a recorded authorization ceiling;
DataSphere SDK 0.10.0 does not expose a live job-cost meter here, so the
controller cannot enforce that amount during execution. The admission owner
must verify that the sealed estimate is below the cap and the cap is within
the current spend authority before arming the service. The admission also
binds exact SHA-256 maps for every uploaded Python helper, requirements file,
runtime marker, and the local controller/probe modules; missing or changed
files fail before submission.

The runtime admission must satisfy all of these inequalities:

```text
startup reserve + internal fit deadline + collection reserve + 60s provider grace
    <= cancel_after_seconds_from_submit
cancel_after_seconds_from_submit + 60s provider grace
    <= external_deadline_seconds_from_submit
```

The finite limits come from the sealed fit specification and current admission;
there is no inherited one-hour profile limit. On the first `EXECUTING` poll,
the controller checks the observed submit-to-execution delay plus the internal
deadline, collection reserve, and provider grace against the immutable cancel
deadline. If startup expires or the observed delay leaves too little time, it
cancels only the persisted exact job ID and retries that cancel at most once
per minute. The provider status is the earliest remotely observable execution
proxy; the sealed startup reserve must also cover any gap from that status to
the fit entrypoint's own timer. If the first observed phase is
`UPLOADING_OUTPUT`, the execution start is recorded as unobserved and the
controller allows output upload to continue under the sealed outer deadline.
The fit entrypoint's internal deadline starts before its archive staging and
preparation work.

The controller does not treat CLI `download-files` exit 0 as collection
success. Once the exact job is terminal, it queries the SDK's job record by
persisted job ID and verifies project, immutable name, terminal status, and
the single `fit-results.tar.gz` / `OUT` manifest entry. It records the
provider-declared byte count and SHA-256, downloads the output without logs or
diagnostics, reconciles local size and SHA-256, then verifies every file in
the embedded `fit-artifact-manifest.json` using `verify_bundle_archive()`.
Each durable state records `DOWNLOAD_ATTEMPTED` separately from `VERIFIED` or
`VERIFIED_PARTIAL`. A restart rechecks the provider manifest and local bundle
before returning success. `VERIFIED_PARTIAL` means the bundle is integrity-
verified but the pair receipt is incomplete or the provider job did not end
in `SUCCESS`; a zero controller exit code means collection verification
finished, not that the fit pair is complete or that its quality was graded.
An otherwise valid terminal job with an empty output-file list receives at
most three manifest reads with bounded backoff. A provider error or a wrong
job, path, variable, size, or hash fails closed immediately.

Run offline checks from this directory after the F1 producer modules are
present in the worktree:

```sh
python3 -m unittest -v test_f1_fit_controller
python3 -m unittest -v test_controller
```

No provider call is made by these tests. Real dispatch also requires the
durable host, short-lived credentials, service restart policy, and external
hard-stop timer to be preflighted on the host that will own the job. This
adapter does not configure those host controls or launch a fit.
