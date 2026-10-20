# Seed-5 teacher preparation

This note records a conditional seed-5 successor binding. [TEACHER_S5_FIT_BINDING.json](TEACHER_S5_FIT_BINDING.json) keeps the reviewed H384x2 recipe, F0 membership, source, propensity dependency, and one-GPU resource envelope. Its argv differs from S4 only by run name and seed. It binds the S4 predecessor to recipe SHA `c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88`, native invocation `b6d71599e0044970b8641e3e241c40a4`, and the required root-acceptance schema.

[admit_teacher_s5_fit.py](admit_teacher_s5_fit.py) reuses the S4 profile gates and requires caller-supplied hashes of the actual root-accepted S4 terminal acceptance and exact-stop capture. Both raw byte sequences are hash-checked before launch and are carried as Base64 into the remote preflight. The script remains conditional until root acceptance and exact stop are complete; this preparation records no S5 admission, fit, or quality result.

After root records the actual acceptance and stop artifacts and hashes, the required invocation is:

```sh
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/admit_teacher_s5_fit.py \
  --s4-terminal-acceptance-sha256 <actual-root-accepted-S4-acceptance-SHA-256> \
  --s4-stop-sha256 <actual-S4-exact-stop-SHA-256>
```

The fixed invocation identities and raw-byte digest checks are covered by [test_teacher_s5_fit_admission.py](test_teacher_s5_fit_admission.py). Synthetic test results and local source hashes are recorded below after validation.

## Local validation receipt

- `./.venv/bin/ruff format` and `./.venv/bin/ruff check` passed on the admission script and its focused test module.
- From this `preparation` directory, `../../../../.venv/bin/python -m unittest test_teacher_s5_fit_admission -v` passed all four synthetic tests: fixed argv/lineage, raw acceptance digest preservation, rejection of a wrong digest/native invocation/update count, and exact-stop unit/population gates.
- A fresh `python -B` cold import passed with non-stdlib imports blocked and file-read/open plus subprocess sentinels installed. Running the CLI without hashes exited `2` at argparse and listed both required SHA-256 options.
- No S4 terminal acceptance or exact-stop artifact was read by this preparation. No SSH, fit, model/data payload, or install was attempted.

Source hashes:

| File | SHA-256 |
| --- | --- |
| `TEACHER_S5_FIT_BINDING.json` | `6d9d91b9d19b4d2cd6c957433e534f74880f1bf8414a06e3434adf8eb56d734a` |
| `admit_teacher_s5_fit.py` | `45633a65e16fa0e9b81d68c49bf29a8884ab1d14122e3585bec84ad888cce8ed` |
| `test_teacher_s5_fit_admission.py` | `706bb2a2d41263659686a5cf05602b4368b54f43454f45c46e74cea597973b0d` |

A preliminary `uv run ruff` command stopped during dependency resolution before Ruff started on macOS ARM; no dependency was installed. Validation then used the existing literal `.venv` tools. This is local preparation history, not a native model or remote-attempt failure.
