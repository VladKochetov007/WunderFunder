# AllThree terminal verifier V2 supplement

V2 is a byte-preserving repair of V1 with exactly one explicit `--setenv=FG19_APPROVED_PATH_COMPAT_EXCEPTION=pathlib._local.PosixPath:exact-dictionary-path-only` argument in the support unit's `systemd-run` argv. The independent exact-body review is now durable and records `PASS_SOURCE_ONLY`; the root payload check records a static pass for all 15 unchanged payload joins. These reviews establish a source and binding result only. Root's separate outer stdin/transport and fresh admission gates remain necessary; there is no automatic B-execution claim, new fit, or quality result.

The focused byte/AST fixture has three cases and was invoked twice: first before final Ruff formatting cleanup, then again on the final fixture bytes. Both invocations passed. No existing suite was run. The original V2 README predates the durable exact-body review and still says review is pending; the V2 review and root payload check linked here carry the later status.

## Files

- [V2 worker body](REMOTE_SUBMISSION_V2.py) is the frozen V2 body with the one explicit environment argument.
- [V2 binding](V2_BINDING.json) records the exact V1-to-V2 byte relation, V1 negative blocker, root preparation, native evidence, and all 15 inherited payload identities.
- [Independent exact-body review](FINAL_BODY_SOURCE_REVIEW_V2.json) records `PASS_SOURCE_ONLY`, no blockers, and the exact V2 body/binding/native pins; it does not authorize remote staging or execution.
- [Static payload check](ROOT_PAYLOAD_CHECK.json) records `PASS_STATIC_EXACT_BODY_PAYLOAD_BINDING_ONLY`, 15 payload joins, exact native keys, and the single-argv operational AST difference.
- [Focused repair fixtures](test_allthree_terminal_verifier_v2_repair.py) check inverse-byte identity, normalized full-module AST identity, and missing/duplicate insertion failures; see [local checks](LOCAL_CHECKS.json) for the recorded result and scope.
- [Local packet README](README.md) describes the repair and its local-only status; its review-pending sentence predates the final exact-body review.
- [Local checks](LOCAL_CHECKS.json) records the final focused fixture and Ruff result, along with no-import/no-B/no-payload-read limits.

