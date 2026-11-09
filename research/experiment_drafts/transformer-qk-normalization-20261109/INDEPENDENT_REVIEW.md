# Independent source and packet review — 9 November 2026

## Scope and verdict

Read-only review of the working tree based on PR72 commit
`4021de8ec5130143c207319c73a102a453eb911e`: Q/K normalization source and config
integration, prepared CPU/GPU/DDP checks, four frozen configs, receiver gates,
comparison, runbook, and submission-status wording. No tests, imports, training,
inference, profiles, or remote jobs were run. **No blocking source or recipe
defect found in this review.** This is not runtime acceptance or model-quality
evidence.

## Review findings

- `qk_norm` accepts only `none` or `rms`; omitted and `none` normalize to the
  legacy config. The RMS helper adds no parameters or buffers. In each block,
  normalization is applied in FP32 to new per-head Q/K before RoPE; normalized,
  rotated K is cached once, while old keys and values are left untouched. The
  new source module is included by the repository's `source_identity`, which
  hashes all `src/**/*.py` files.
- The four configs differ from PR72 C0 only by seed and the optional treatment
  flag. The comparator binds the full source, config, data, optimizer/model
  footprint, terminal checkpoint bytes and profile approval; it reports both
  paired seed deltas and a component bootstrap conditional on those predictors.
  It does not select epochs, seeds, or blend weights.
- Prepared checks cover legacy `none` equivalence, configuration rejection,
  finite normalization gradients, causality, chunk/step/reset behavior, cached
  key handling, checkpoint state, compiled efficient versus eager math attention
  in both declared BF16 modes, and two-rank gradient equivalence with an empty
  rank. They remain **unrun**.
- Receiver hardening binds the data-check status, data-root path, split and
  membership, checks exact production and peak profile identities plus reload
  evidence, and requires manual production-profile approval. `panel-bundle`
  preserves the aggregate comparison, acceptance and per-cell bundle manifests.
- The final compatibility repair restores the Transformer adapter's existing
  `hidden` and `attention_backend` attributes. It moves the unchanged FP32
  forward-contract validator to `sequence_model_helpers.check_model_precision`
  and imports it at the former call site; no behavior change was found by
  inspection.

## Limits and remaining gates

The fixed affine-free RMS treatment is distinct from Henry et al.'s learned-scale
L2 QKNorm and ViT-22B's Q/K LayerNorm; neither paper establishes a gain here.
With the existing SDPA scale, the RMS constraint bounds logits mathematically,
but no attention entropy or collapse was measured. Under default Linear
initialization, the approximately threefold initial logit-standard-deviation
change is a derivation, not an observation; the comparison bundles normalization
with a temperature change.

No B300 acceptance, profile, fit, grade, CPU package, or new submission is
established. The separate submission inventory records released checkpoint
inputs, not an accepted ZIP; the PR51 recipe must first be matched against prior
submissions and pass byte-bound CPU acceptance. `VALIDATION.json` records
root-owned static passes: Ruff checks, formatting on 145 files, structure on
121 files with zero errors, packet AST/JSON/config-hash/symmetry checks, Bash
syntax, and `git diff --check`. The record also confirms tests, imports, model
execution, profiles and remote jobs were not run. I did not execute those
checks. Scientific and receiver evidence remains **UNRUN**.
