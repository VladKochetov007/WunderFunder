# Independent static review

**Verdict: static GO for the draft packet.** This is a source and contract review
of the current uncommitted working tree, stacked on PR72 head
`4021de8ec5130143c207319c73a102a453eb911e`. It does not establish runtime
correctness, score quality, receiver acceptance, or publication readiness.

The implementation places a zero-initialized depthwise residual stem after
fit-only standardization and before the existing input projection. Kernel 16
uses left-only padding and carries a compact cloned tail of 15 standardized
raw rows; the sequence state validates row count, device, and dtype and resets
when the caller starts a sequence with `state=None`. Kernel 1 is an
instantaneous per-channel residual gain. Omitted and explicit-zero settings
normalize to the legacy config shape and add no model parameters. Stem weights
are created after the legacy modules without consuming initialization RNG and
are grouped in AdamW outside the Muon matrices.

The prepared checks cover zero-init identity/backbone initialization, nonzero
causal and chunk/step behavior, reset, history storage, gradients, BF16 and
BF16x9, and a two-rank gradient comparison. The receiver packet freezes six
fresh cells, full-valid scoring, paired same-seed runtime, source/config and
checkpoint byte identities, manual profile approval, optimizer-footprint
comparison, and optional fixed 90/10 blends. Earlier review findings about
retaining full-chunk storage, zero-weight causal tests, config-template
comparison, and approved-profile byte binding are corrected in the current
tree.

ROOT reports Ruff 0.16.7 over 144 files, training-structure 121/0, and
AST/JSON/Bash/hash/symmetry checks. I did not execute those checks or any tests.
No tests, model imports, training, inference, profiles, remote jobs, or
scientific grades were run for this review; all receiver and quality evidence
remains **UNRUN**. Keep `sequence_models.py` within its 300-line structure cap
(it is currently 297 lines).
