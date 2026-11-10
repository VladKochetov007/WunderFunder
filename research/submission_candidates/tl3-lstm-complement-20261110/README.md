# Canonical TL3 + 10% LSTM CPU candidate — 10 November 2026

Published as [draft PR76](https://github.com/VladKochetov007/WunderFunder/pull/76) into `research-team`.

Receiver revision2 removes redundant per-row blend recomputation during full
validation and stores scored rows only; the initial profile defaults to two
complete sequences. See [the receiver revision](RECEIVER_R2.md).
The ZIP itself is unchanged.

One fixed candidate, zero new fits, no local tests or inference. Final ZIP:
**14,011,657 bytes**, SHA-256
`766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e`. The user asks
for actual CPU-ZIP bytes and commands to validate them on a server. This packet
performs data-only checkpoint export and packaging locally; model execution is
confined to the separately invoked Linux receiver commands. It does not upload
to the competition or alter any GPU job.

The composition is **0.90 × canonical fresh TL3 + 0.10 × PR53 LSTM**. Within TL3,
the coefficients are 0.34 W18 train-only student, 0.33 tiedlead seed5 and 0.33
tiedlead seed6. The LSTM is the fixed eighth epoch of
`primary-refit-03-s1t000000-lstm`, raw112/H192/one layer, seed1302. Saved training
normalizers are preserved; the first two output rows of each GRU head are kept.
The model checkpoint bytes are exactly the four identities recorded in PR53's
fixed-mixture report, not the earlier fbc1k replacement blend or full-corpus W18.

## Evidence and selection

[PR53's report](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545163066)
measured WP **0.6925020625522238**, versus the matched canonical TL3
**0.6923774247837755**, delta **+0.000124637768448288**, on all1873 official valid
sequences/3528273 scored rows. The six mixture coefficients were predeclared;
the choice among their reported outcomes is validation-exposed selection. The
small gain is neither statistical significance nor a public-score forecast.
[The exact archived report](PR53_FIXED_MIXTURES_REPORT.json) is pinned in
[INPUTS.json](INPUTS.json). It uses FP64 raw mixture then one FP32 cast.

This ZIP changes serving arithmetic: the GRUs use the existing native dynamic
INT8 runtime/fast activations, and the LSTM uses NumPy/SciPy FP32 streaming (GELU uses the analytic erf expression with
FP64 intermediate values and an FP32 result).
Research Torch tiedlead encoding resets row-difference features every4000-row
evaluation chunk; the historical native runtime streams continuously. Therefore
the research score above **is not a measured score of this ZIP**. Full callback
validation must establish the delivered predictor's quality and incremental
benefit. Legacy teacher-cache consumption ancestry remains incompletely recorded;
published checkpoint/source identities do not resolve that historical limitation.

## Duplicate check

[DEDUP.json](DEDUP.json) records the signed-in portal observations and downloaded
archive identities. All108 submissions were unchanged as of10November; the only
submission after the LSTM's31October release was4November's full-corpus W18
singleton. The downloaded29October PR49 blend is exactly0.34fbc1k+0.33tlead5+
0.33tlead6, dynamic INT8. It is excluded, as are the old TL3 and already-submitted
W18 singleton. The new recipe contains an additional separately trained LSTM
and different baseline student. Novelty is established against these recipes
and the visible subsequent history, not by a renamed ZIP or local prediction
execution. There is no exhaustive semantic comparison against every historical
model's outputs. Recheck portal history immediately before a later upload.

## Files and execution

- [recipe.json](recipe.json): fixed composition and serving arithmetic.
- [INPUTS.json](INPUTS.json): release/checkpoint/native/split/data identities.
- [build_zip.py](build_zip.py): restricted tensor-only export and deterministic
  archive creation. It never constructs a model or loads a native library.
- [solution.py](solution.py), [lstm_cpu.py](lstm_cpu.py): shipped callback and
  chronological LSTM implementation. Both recurrent branches update on every
  row and reset together when the sequence changes.
- [RUNBOOK.md](RUNBOOK.md): transfer, profile and full-validation commands.
- [receiver.sh](receiver.sh), [Dockerfile.receiver](Dockerfile.receiver),
  [validate_cpu.py](validate_cpu.py): explicit offline Linux x86-64 CPython3.11
  receiver path. No automatic fit, upload or GPU use.
- `BUILD_RECEIPT.json` and `SHA256SUMS` in the delivery directory bind the actual
  ZIP and handoff bytes; `STATIC_REVIEW.md` records independent source review.

The official [submission guide](https://wundernn.io/connectome/docs/submission_guide)
was checked on10November: one vCPU,16GB RAM,60minutes,20MB ZIP, no GPU/network;
the reference CPU is AMD EPYC9224. NumPy>=1.26,<3 and SciPy>=1.13,<2 are installed. The ZIP bundles hash-pinned `threadpoolctl`3.6.0 and its BSD license to keep
already initialized BLAS pools at one thread, without relying on an unlisted
transitive dependency.
The [data overview](https://wundernn.io/connectome/docs/data_overview) confirms
20000rows/sequence and1873validation sequences. It does **not** publish a whole
test-row count. Receiver timing reports therefore include a conditional
projection only when a separately justified test-row bound is supplied.

Until receiving ABI/state/parity/full-valid/resource checks pass for the exact
ZIP, status remains **PACKAGED_NOT_EXECUTED**, not ready-to-submit. Runtime
measured on another CPU is a proxy for the portal. There is no new leaderboard
score, and a negative quality or latency result does not trigger weight tuning.
