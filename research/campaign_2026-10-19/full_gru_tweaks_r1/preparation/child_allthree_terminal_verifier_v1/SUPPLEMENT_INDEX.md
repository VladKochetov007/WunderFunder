# AllThree terminal verifier V1 supplement

V1 is retained as the conditional source packet and the exact rendered body rejected by independent source review. Its conditional review passed only for the template/renderer boundary; the actual worker body was rejected because its preflight checked the approved `FG19_APPROVED_PATH_COMPAT_EXCEPTION` value but omitted the matching explicit `--setenv` from the support unit's `systemd-run` argv. This is a source blocker, not a remote attempt. V2 contains the narrow argument repair and has its own exact source review. V1 is not an execution authorization and provides no new-fit or quality evidence.

## Files

- [V1 README](README.md) describes the conditional packet and its original readiness gates; use the exact-body review below for the current V1 finding.
- [Conditional source review](CONDITIONAL_SOURCE_REVIEW_V1.json) records `PASS_CONDITIONAL_SOURCE_ONLY` for the wrapper template and renderer, not the subsequently rendered worker body.
- [Rejected exact-body source review](FINAL_BODY_SOURCE_REVIEW_V1.json) records `REJECT_SOURCE_ONLY_NO_B_ATTEMPT` and the missing explicit path-compatibility environment argument.
- [V1 rendered worker body](REMOTE_SUBMISSION.py) is the exact 706,653-byte body bound by that rejection; it is retained unchanged as the bytewise V2 repair base.
- [V1 wrapper template](REMOTE_SUBMISSION_TEMPLATE.py) and [conditional renderer](render_child_allthree_terminal_verifier_v1.py) are the reviewed source inputs; V2 does not rerender them.
- [Static V1 payload check](ROOT_PAYLOAD_CHECK.json) binds 15 embedded payload identities/byte joins, seven private code files, the root preparation, and native evidence. It is a static binding record, not payload-content validation or execution.
- [Renderer-focused test source](test_render_child_allthree_terminal_verifier_v1.py) and [local-check notes](LOCAL_CHECKS.md) preserve the V1 preparation's prior local checks; this navigation task did not run them.
- [Checker root preparation](../child_allthree_terminal_check_v1/ROOT_PREPARATION.json) and its [metadata review](../child_allthree_terminal_check_v1/ROOT_PREPARATION_METADATA_REVIEW_V1.json) bind the shared seven-code closure, root metadata, and local-only status.

