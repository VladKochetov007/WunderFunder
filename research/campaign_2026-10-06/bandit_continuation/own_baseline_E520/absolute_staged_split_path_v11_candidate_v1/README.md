# E520 profile V11: absolute staged split path

V11 is one profile-only successor to V10 under the root allocation in
[E520_PROFILE_V11_ROOT_SUCCESSOR_ALLOCATION_20261007_R1.json](../E520_PROFILE_V11_ROOT_SUCCESSOR_ALLOCATION_20261007_R1.json). It allows at
most 1,450 worker seconds and 2,900 provisional Kaggle GPU seconds, within the
12,000-second cumulative E520 recovery ceiling. It authorizes no fit, export,
grade, or score.

V10 verified the mounted train and validation identities, fold identity and
membership, and the live CUDA-to-NVML GPU UUID binding. It stopped before any
profile batches because the worker passed
`research/campaign_2026-09-29/artifacts/splits/fold0.json` to the trainer as a
relative path. The pinned `load_campaign_split()` resolves that path against
the Kaggle process working directory, which V10's error records as
`/kaggle/working`; the resolved but absent file was therefore
`/kaggle/working/research/campaign_2026-09-29/artifacts/splits/fold0.json`.
The verified manifest was staged below the extracted source tree instead.
V10 used 909.824 worker seconds and 920.099 quota seconds, and its complete
terminal files are retained in the
[V10 terminal download manifest](../E520_PROFILE_V10_TERMINAL_DOWNLOAD_20261007T1659Z.json).

The trainer is not launched as a subprocess by this worker. The worker imports
the pinned `scripts/train_rnn.py`, installs the profile hooks, assigns
`sys.argv`, and calls `module.main()` in the same process. Its working
directory is inherited from the Kaggle kernel. The trainer's output root is
anchored separately to its own `__file__` path.

V11 changes the only nonempty filesystem path argument in the profile argv:
`--split-manifest` now receives the absolute staged fold path, after the worker
checks that the file exists. The pinned trainer source, split contents,
training recipe, membership checks, CUDA/NVML identity checks, memory limits,
and digest handoff are unchanged. Other path-valued trainer options are absent,
unset, or empty for this profile; `--quote-cache-root` and
`--spread-cache-root` remain at empty defaults and are not used by the recipe.

`tests/test_split_manifest_cwd.py` writes a real manifest fixture under a
staged source root and launches a child Python process with its working
directory set to the Kaggle working-root fixture. The former relative path
fails to open the file; the checked absolute staged path opens it and returns
the fixture contents. The focused test passed locally under one CPU affinity
and a 1.5 GB address-space limit. This test exercises the same `Path.resolve()`
boundary as the pinned loader while keeping the actual trainer invocation
model clear: the trainer runs in-process.

No V11 packet has been built or pushed. Exact source and transport review,
fresh quota/resource admission, and a single guarded private profile push are
still required. No fit is authorized by this candidate.
