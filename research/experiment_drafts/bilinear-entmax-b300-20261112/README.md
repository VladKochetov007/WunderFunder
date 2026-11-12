# Bilinear inputs and entmax on B300

Two implemented, independent architecture screens; **8 prepared fits, 0 executed**.
Publication and static checks do not establish numerical correctness, B300 fit,
quality improvement or submission readiness. Existing pipeline and running jobs
are unchanged. No local model tests, inference, training or profiling are allowed.

| Study | Control | Treatment | New fits |
| --- | --- | --- | --- |
| Input interaction | rank32 additive i0/i1 residual | rank32 bilinear i0×i1 residual | 2 arms × 2 seeds = 4 |
| Attention normalization | explicit FP32 softmax normalization | fixed alpha1.5 entmax normalization | 2 arms × 2 seeds = 4 |

Fresh seeds: **206201, 206202**. Each model uses full supplied fitting data,
H768/L24/12 heads, context1024, 4 epochs, BF16/BF16x9, 2×B300 and the pinned
PR75 Muon/AdamW recipe. No historical controls or additional teacher/student fits.
Official valid is repeatedly exposed; these are exploratory paired comparisons,
not a new clean holdout or evidence of statistical significance.

The bilinear arms retain the native projection of **all112** inputs, including
`a0..a7`. They have identical U/V/W parameterization and initial native function;
only addition versus multiplication differs. The attention arms share an explicit
QK→normalizer→V path, positions, mask, cache and parameters. Alpha is fixed, not
searched. The existing efficient attention remains in the input study.

Entmax uses dense sorting. Its full-shape memory and compilation acceptance are
unknown; if the frozen profile fails, preserve the failure and do not silently
reduce context, batch, layers, precision or change kernels.

## Contents and provenance

- [Scientific contract / PR body](PR.md): mechanism, competing explanations,
  exposure, allocation and decision rules.
- [Frozen recipe](recipe.json) and [configs](configs/): eight exact cell descriptors
  plus two seed-qualified native recipes; their hashes are enforced by admission.
- [B300 commands](B300_COMMANDS.md): prepare → checks → per-cell profile → explicit
  resource approval → fit → fixed-E4 export → full score → two paired comparisons.
- [Runtime](runtime/): `launch.py` owns bounded subprocesses and logs;
  `worker.py` selects canonical stages; `approve_profile.py` records human review.
- [Architecture package](runtime/bilinear_entmax/): `modeling.py`, `input_block.py`,
  `attention_block.py`, `entmax.py` implement only the requested substitutions.
  `bindings.py` freezes source/cells; preprocessing/fitting/admission/endpoint/
  exporting modules adapt the existing PR75 lifecycle. Comparison modules admit
  exact endpoints and paired identities. `receiver_*.py` are server-only numerical,
  mask/cache/reset/reload/gradient/optimizer acceptance checks, not local results.
- [MIDAS handoff](midas/README.md), [actual route preview](midas/route-preview.json),
  `midas_evidence*.py`, `midas_source_admission.py`: metadata routing and strict
  external evidence export, without changing MIDAS or starting an agent/GPU job.
- [Source audit](sources/SOURCE_AUDIT.md), [source pins](sources/provenance.json),
  [native recipe](sources/native_control.json), [runtime hash manifest](sources/runtime_source_files.json)
  and [PR snapshot](sources/pr-audit-20261112.json) explain ancestry and novelty.
- [Third-party notices](THIRD_PARTY_NOTICES.md) retain the entmax MIT license.
- [Static review](sources/STATIC_REVIEW.json) records only checks actually performed.

Canonical trainer source: `8626000a4b965763a70011a0291a891aa6535d7e` (PR75).
Experiment wrappers derive from PR90 `34b2beaf165eb7a224b792c17b99e41b5fd7384a`;
its signed/additive attention and FFN routing mechanisms are not included.
Base branch: `research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`.

No server, data paths, assigned receiver or fresh compute budget are bound yet.
The receiving operator must supply these and pass the documented gates.
