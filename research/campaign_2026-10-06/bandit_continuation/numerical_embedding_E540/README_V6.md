# E540 V6 argv and startup-preflight repair

V6 preserves the frozen V5 scientific pair: 512 selected complete components,
the same 468 exposed evaluation groups, AMP initial scale 4096 in both arms,
clip2-target WMSE with chunk normalization, one fixed epoch-0 checkpoint, and
the existing paired profile gate. Its only changes repair command dispatch and
require an explicit root conditional-fit receipt before hashing mounted data.

V5 verified all four mounted inputs against their expected sizes and SHA-256
identities, then failed before profiling with `train_rnn.py: error:
unrecognized arguments: scripts/train_rnn.py`. `call_main` prepended the script
name while its recorded argv already contained the script token. V5 therefore
has no profile or scientific-fit outcome; the PLE arm did not start.

V6 removes exactly the logged `scripts/<entrypoint>.py` token before assigning
`sys.argv`. Before any large mounted-input hashing, its stdlib-only startup
check parses the actual argument declarations from `train_rnn.py` and
`predict_run.py` for both profile commands, both fit commands and both
prediction commands. It does not import Torch, access CUDA, read parquet
footers or load target arrays. Any parser mismatch or absent authorization is
recorded in `e540_cohort/startup_preflight.json` and stops the worker before
the long input-identity pass.

The packet contains
`E540_V6_ROOT_CONDITIONAL_AUTHORIZATION.json`. The committed file is a pending
template with `authorized: false`; it cannot launch a fit. After independent
review, root must issue a receipt binding the V6 code-manifest, recipe and
frozen subset hashes, then rebuild the bundle so the approved receipt itself
is input-hash-bound. That receipt authorizes the fit only after both profile
receipts pass the existing zero-AMP-skip, finite-update, memory and projected
runtime gates. Provider push remains a separate root-owned admission.

The V6 bundle is a prepared review artifact only. No provider upload or push,
profile, fit, prediction or grade was performed for V6. V5's failed execution
and its input identity receipt remain immutable evidence.
