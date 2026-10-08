# H232 full-export alignment patch: independent read-only review

Review date: 2026-10-08 UTC  
Disposition: pass for the reviewed alignment change; no blocker found.

## Scope

I reviewed the changed `align_full_valid_export` wrapper and its call site in `grade_fixed_reference_blends.py`, together with the shared H218 alignment/scoring helper and the frozen H232 R2 preregistration. The reviewed source hashes are in the paired JSON receipt. This follow-up preserves the earlier source review; it changes neither the grader nor the preregistration.

## Finding

The candidate export contains all 1,873 valid groups, while the frozen comparison evaluates only the 468 excluded fold-0 groups. The old helper expected the candidate array itself to contain exactly the evaluation membership, so it rejected the actual full-valid export. The new wrapper checks that the candidate's full row-group order equals the target cache order, constructs a source-qualified `("valid", row_group)` index over that full export, and uses the shared strict join to obtain reference and target indices for the preregistered evaluation groups. It then substitutes the candidate's actual full-export indices into the aligned rows. The existing caller still validates the full prediction array against all candidate groups before scoring.

I ran a bounded synthetic fixture with candidate/target groups `[91, 42, 12, 7]`, evaluation order `[7, 42]`, and reference order `[42, 7]`. The result was `[(('valid', 7), 3, 1, 3), (('valid', 42), 1, 0, 1)]`, which preserves the candidate and target indices from full-export order while joining reference rows by group identity. A candidate/target row-order mismatch raised the expected `ValueError`. This exercises noncontiguous and reordered IDs without reading actual candidate predictions or targets.

## Conclusion and limits

The alignment patch correctly maps the fixed evaluation subset into the full candidate export and preserves source-qualified identity checks. No scoring or candidate artifact was read; no score was computed; this review does not authorize grading. Root still owns terminal-evidence validation and grade admission.
