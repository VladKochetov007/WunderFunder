# Exact QAT member in the fixed cross-team package — R1

This code-only delivery extends `scripts/build_cross_team_submission.py` with
an opt-in QAT path while leaving the legacy builder call unchanged. The new path
keeps the existing four-member recipe order and coefficients, and only permits
replacement of member four (`E/weights_0.npz`, coefficient `0.25`). It binds the
replaced inventory member, packed QAT archive, training checkpoint and portable
native manifest by SHA-256. The ZIP includes the package-local NumPy reader,
exact archive and training provenance; existing package archives are untouched.

`scripts/templates/qat_archive.py` validates packed archive version 2 and
restores effective float32 tensors plus the original signed code/scale arrays.
It rejects Q4 `-8`, Q8 `-128`, invalid scale rows, bad odd-code padding,
incomplete GRU matrix sets, head splits away from the hidden width and any
archive field mismatch. It never derives replacement scales. The callback
constructs native-generated Q8 maps for the three unchanged members, replaces
only the QAT member map with the archive's exact rows, and imports all four maps
through the hash-bound `q8vnni` library. It retains the original `.25` blend,
`budget_s=None` and native `predict` sequence reset. The native runtime still
expands W4 to byte-kernel rows; this work makes no INT4 speed claim.

## Bounded validation

The fixture used the actual S4 learned-scale deployment archive at
`artifacts/qat_renewable_r1/lsq_s4_10pct_learned_deployment_r1/recurrent_w4.npz`
(SHA-256 `d205f01063c272bfbd4837c9891a190f60547365a3d4a0feea487deaf4b7c182`)
and the portable exact-import native set in
`final_release_r1/NATIVE_SET_R1.json` (manifest SHA-256
`9d2392da6ca3149a94cd5e1b42b60d2b5381a116cf3e10f08ac3d0888bffadba`; library
SHA-256 `1fa4de2594d1918b4dc11a75b8901dc5e1854b1f00dd6195b8201fed35384052`).
The S4 deployment manifest SHA-256 is
`09f996b7a59ad544da8b31a1c268880ac94ed52c4d9823a73101b80ba5a22c13`. This is
a 10%-subset, exposed-F0 serving fixture only; it is not eligible as a final
submission weight. Its deployment manifest labels the training rule as the
LSQ-inspired learned row-scale derivative, not a full LSQ reproduction. The
replaced team source is inventory-bound as `E/weights_0.npz`, SHA-256
`62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6`.

Under the initial one-CPU bound, the focused source-format, malformed archive,
manifest-binding, callback and existing cross-team builder tests passed:
**17 passed**. After the CLI correction, the same suite passed again in 20.03
seconds under 25% CPU, 500 MB and 180 seconds, with `PYTHONPATH` removed in
both direct-script and module-mode builder subprocesses. The actual source fixture was decoded both by the
package-local reader and `wnn.quantization_export`; all exact code and scale
arrays matched. Native `EnsemblePredictor` constructor-generated maps were
exported and re-imported for the original four weights, and callback outputs
were bitwise-identical across that roundtrip, including a sequence reset. A
second exact-import check replaced the last member's map with the QAT archive
rows and verified every imported code/scale against the archive. The transient
builder integration also produced and loaded a CRC-checked four-member package
under the same 500 MB cap; its prediction replay after changing `seq_ix` was
bitwise-identical. The temporary ZIP and package directory were removed after
the test.

Exact focused command after the CLI correction (the test asserts the fixture
SHA and removes `PYTHONPATH` from both builder subprocesses):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=25% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=180s \
WNN_QAT_SOURCE_FIXTURE=/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/qat_renewable_r1/lsq_s4_10pct_learned_deployment_r1/recurrent_w4.npz \
WNN_QAT_SOURCE_FIXTURE_SHA256=d205f01063c272bfbd4837c9891a190f60547365a3d4a0feea487deaf4b7c182 \
WNN_QAT_NATIVE_MANIFEST=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-13/qat_renewable_r1/final_release_r1/NATIVE_SET_R1.json \
WNN_NATIVE_EXACT_LIBRARY=/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/native_qat_release_r1/wnn_native.abi3.so \
WNN_XTEAM_H4_PACKAGE=/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/team_models_20261013/xteam_h4 \
WNN_QAT_PACKAGE_TEST_ROOT="$PWD" PYTHONPATH="$PWD/src" \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
-m pytest -q tests/test_qat_submission_archive.py tests/test_build_cross_team_submission.py
```

The production builder interface is:

```sh
python scripts/build_cross_team_submission.py \
  --inventory research/campaign_2026-10-13/team_models_r1/INVENTORY.json \
  --recipe <controller-selected-four-member-recipe.json> \
  --qat-release-spec <hash-bound-final-QAT-release.json> \
  --native-manifest research/campaign_2026-10-13/qat_renewable_r1/final_release_r1/NATIVE_SET_R1.json \
  --out-dir <new-durable-package-directory>
```

## Direct-script import correction

The first production-style direct-script invocation failed before creating an
output directory: `python scripts/build_cross_team_submission.py ...
--qat-release-spec ...` raised `ModuleNotFoundError: No module named
'scripts'` while importing `scripts.templates.qat_archive`. The original
pytest fixture had imported the builder as `scripts.build_cross_team_submission`
with the repository root already on `sys.path`, so it did not exercise the
direct-script import context. The builder now uses a relative helper import in
module mode and `templates.qat_archive` when run directly, where Python places
the `scripts/` directory on `sys.path`. The actual S4 fixture subprocess
regression runs both CLI forms with `PYTHONPATH` removed and verifies that they
produce the same ZIP hash; the callback replay remains covered by the direct
build's package fixture. This is an integration repair only and makes no change
to package contents or model behavior.

The release spec must bind the exact all-data final checkpoint/archive selected
by the controller and state its seed, update count, parent checkpoint hash,
training rule, scale-learning semantics and training membership. No final
all-data QAT archive was consumed here, and this builder work makes no model
quality, submission-rank, public-score or target-host runtime claim. Root owns
final-weight selection, complete callback grading and any public test.
