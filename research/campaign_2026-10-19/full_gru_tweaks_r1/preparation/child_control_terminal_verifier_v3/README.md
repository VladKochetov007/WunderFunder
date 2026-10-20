# CONTROL terminal verifier V3 compatibility proposal

This folder contains a local-only successor proposal for the completed CONTROL
integrity check. V1 and V2 are preserved as executed history. V2 exited before
writing `VERIFICATION.json`; this packet does not retry either verifier and does
not stop the native training unit.

The V2 capture is pinned at 26,978 bytes, SHA-256
`d7f81c3773ee27c48611623689eda482b41dfc6ddee7776c61f905db8e76f7d2`, invocation
`c890ef75142844a9a8f61ae21e38a97f`, with `MainPID=0`, `ExecMainCode=1`, and
`ExecMainStatus=1`. Root's review of that terminal trace identified the failure:
the restricted `weights_only=True` loader rejected `GLOBAL
pathlib._local.PosixPath`. The fitted checkpoint itself was not opened during
this local preparation.

The cause is source-qualified in commit
`68ff682977b822f38351aac0f621146f5f13b440`. The original CONTROL launch uses
`--representation-cache` with the exact dictionary artifact path. At
`src/wnn/training/cli_features.py:110-115`, argparse converts that argument to a
`Path`. `src/wnn/training/checkpoints.py:120-128` keeps the field in
`resume_contract(...).options`; `src/wnn/training_state.py:21-30` preserves
non-tensor values with `deepcopy`, and lines 145-151 capture that contract.
`src/wnn/training/outputs.py:157-185` writes the continuation state, while
`src/wnn/training_state.py:225-234` serializes it with `torch.save`. The exact
original `LAUNCH.json` is pinned and its `--representation-cache` value matches
the accepted dictionary path. `STATE_LOAD_COMPATIBILITY_PROPOSAL_V3.json`
records the source file digests and these bounded evidence links.

V3 proposes one narrowly scoped compatibility mapping for the exact serialized
field `contract.options.representation_cache`. The added mapping is an explicit
exception to the otherwise unchanged NumPy RNG and `TorchVersion` safe-global
baseline. It maps only `pathlib._local.PosixPath` to an inert constructor that
accepts exactly one string argument equal to the full pinned dictionary path,
performs no filesystem operations, and returns that exact `PosixPath`. Before
loading, the checker scans the same bounded `O_NOFOLLOW` file descriptor and
requires the only unsupported global beyond the unchanged baseline to be this
one name. Loading remains `weights_only=True`, and the mapping is scoped to
`training_state.pt` only. Any other static global, constructor argument,
constructor count, or decoded Path location/type/value fails closed.

After the restricted load, the checker requires exactly one Path at
`contract.options.representation_cache`, with the exact accepted path. The
metadata walk descends through Python containers and NumPy object and structured
arrays, while leaving numeric arrays opaque. For the existing CLI-contract
comparison, it changes only that verified field in a local copy to its
canonical string. The original decoded state remains typed and unmodified.
Model and dictionary loads retain the existing parent checker path and
allowlist.

The runtime exception requires the exact environment value
`FG19_APPROVED_PATH_COMPAT_EXCEPTION=pathlib._local.PosixPath:exact-dictionary-path-only`.
The wrapper checks it before any B-side directory creation or support
submission; the checker checks it before artifact reads. It records the marker
in `VERIFIER_ADMISSION.json`. The marker must only be set after an explicit
human exception. With it absent, the body exits before staging anything.

The prepared support remains a one-shot CPU verifier with the existing
0.5-CPU, 1.4-GB high, 1.5-GB maximum, zero-swap and 600-second limits, hidden
CUDA, same source and receipt gates, and the existing full model, optimizer,
RNG, lineage and numeric checks. The V3 unit and create-only output are
`computer-vision-homework-fg19-child-control-verify-v3.service` and
`research_tmp/fg19_child_control_verify_v3` on B. No V3 body has been staged or
executed; no actual checkpoint, model, dictionary, cache array or training
payload was read or hashed for this proposal.

The focused local tests cover the synthetic exact-path reducer, one-call
constructor, exact path/type/location checks including NumPy object and
structured-array cases, rejection of another pickle global before unpickling,
clearing and rejecting ambient registrations, the approval guards, the pinned
original launch argument, the pinned V2 failure capture, and the rendered
payload byte pins. They do not load the actual training state or run the full
verifier. See `LOCAL_PATH_COMPAT_TESTS_V3.json` for test and byte pins, and
`PREPARATION.json` / `LOCAL_CHECKS.json` for the rendered-body metadata gates.

[Final root payload check](ROOT_FINAL_PAYLOAD_CHECK_V3.json) independently invokes only the final wrapper's bounded payload decoder, with main omitted:12inputs pass size/SHA guards and exact checker bytes match. No exception marker or B action was used. [Earlier root candidate check](ROOT_PAYLOAD_CHECK_V3.json) is retained as superseded history before the object-array location fix and is not current readiness evidence.

The [compatibility proposal](STATE_LOAD_COMPATIBILITY_PROPOSAL_V3.json) binds the exact producer chain, required exception, source hashes and final code/test identities. [Focused test receipt](LOCAL_PATH_COMPAT_TESTS_V3.json) records 11 synthetic/local metadata tests and their limits; [test source](test_child_control_terminal_verifier_v3_path_compat.py) makes the checks reviewable. [Checker source](child_control_terminal_check_v3.py) is the proposed safe loader adaptation, while [rendered admission body](REMOTE_SUBMISSION.py), [template](REMOTE_SUBMISSION_TEMPLATE.py) and [renderer](prepare_child_control_terminal_verifier_v3.py) preserve the one-shot, fail-closed execution path. [Preparation manifest](PREPARATION.json) and [local static checks](LOCAL_CHECKS.json) pin the fixed inputs and resource/identity gates. All are LOCAL proposal artifacts, not actual B execution or CONTROL acceptance.


[One scoped exception question](HUMAN_EXCEPTION_REQUEST_PENDING_V1.json) was sent through async input at approximately 12:44 UTC and is unanswered at this dated snapshot. The exact baseline is fixed by the current monitoring instruction; this request is for one new decoder mapping, not GPU/budget authorization. No exception marker, B staging/admission/execution or scientific successor launch is authorized while pending. A later reply must be appended as new evidence rather than overwriting this snapshot.


Later [direct human authority](HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json) granted the exact exception. The [actual V3 execution](../../records/child_control_fit_v1/terminal_verification_v3/README.md) passed; [CONTROL acceptance](../../records/child_control_fit_v1/TERMINAL_ACCEPTANCE.json) preceded [exact stop](../../records/child_control_fit_v1/exact_stop/README.md). Pending request above is historical and remains unmodified.
