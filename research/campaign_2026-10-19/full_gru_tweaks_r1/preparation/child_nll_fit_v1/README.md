# F0 seed-1001 NLL child — local conditional preparation

This packet prepares one concrete next fixed arm, `fg19_child_nll_f0_s1001_fit_v1`, on the canonical source68 trainer. It is local preparation only. The current local template is deliberately unbound to the dictionary child's actual terminal acceptance and exact-stop receipts; there is no `REMOTE_SUBMISSION.py`, launch, or admission record in this directory.

The arm is the frozen `mean_variance` design: representation mode `base`, mean/logvariance head enabled, and the source-qualified NLL objective. It keeps the common 467-input / 25-output, H256×2 GRU, parent `fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3`, inherited normalization, seeded order, B16/C2000, six fixed epochs/checkpoint 5, T3 cache/target/weights, and AdamW recipe. It does not initialize from the dictionary child. The exact accepted 22,557-byte dictionary artifact remains a required, preflight-only shared input; missing or mismatched artifact identity fails closed before trainer invocation.

The only trainer-argv value changes from the actual dictionary child are the run name, `--representation-mode sparse` to `base`, and `--mean-variance-objective wmse` to `nll` (indices 1, 48, and 54). `--loss wmse` and `--loss-norm chunk` remain unchanged because source68's mean/logvariance configuration guard requires chunk-normalized wMSE as the compatibility loss even when the selected mean-variance objective is NLL. Source68's `base` representation keeps the shared 256-column suffix as exact zeros; it does not consume the learned sparse transform, but the exact dictionary artifact is still validated and its path is retained.

The immediate sequential predecessor is the actual dictionary child fit, distinct from the frozen learned-dictionary artifact receipt. Rendering requires both `records/child_dictionary_fit_v1/TERMINAL_ACCEPTANCE.json` and `records/child_dictionary_fit_v1/exact_stop/EXACT_STOP.json`. The renderer checks their exact bytes, terminal schema/status, run/unit/invocation, strict integer counters, finite selected epoch-5/state identity, source pins, and exact exited-then-inactive stop state with empty populated descendants. It then embeds them as `DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json` and `DICTIONARY_CHILD_EXACT_STOP.json` in the new run's evidence. Existing parent, CONTROL, EMA, all-three profile, T3 cache, and learned-dictionary-artifact receipt gates remain in the inherited admission body.

`NLL_LAUNCH_TEMPLATE.py` is a real source-derived admission template carrying the reviewed dictionary V2 resource, data, lineage, ownership, path, cache, GPU, and scratch gates. Its first `main()` action requires actual dictionary-child acceptance and exact-stop pins. All four new receipt slots still contain explicit unbound markers, so running that template fails before any B-side action. `bind_nll_fit.py --check` re-derives it from the pinned dictionary body in memory, proves the three exact argv value changes and source-manifest pins, and asserts that final body and binding files are absent. `bind_nll_fit.py --render` is create-only and refuses to proceed unless the actual receipt files exist and pass the concrete checks.

The unchanged all-three shape profile is reused; this packet does not replay a geometry profile, retune, evaluate, or fit. Resource gates retain one owned selected GPU, worker CPU100%, 7.5GB high / 8GB max / no swap, 4GiB CUDA allocator with 1GB reserve, 24-hour runtime, parent two CPUs and 17GB target, aggregate two CPU cores, loader0, BLAS1, and exact physical `research_tmp` mount/device/ext4/25GiB-free checks. Fresh all-populated, ownership, source167, split, cache/dictionary, output/unit absence, disk, GPU, and headroom checks are preserved for any later actual admission.

At the latest dated dictionary observation recorded in `CONTINUATION.md` (21:14UTC), the native dictionary fit was still active with five complete epochs. This packet does not infer a later state or read its changing checkpoint/state payloads. Root must supply the actual accepted terminal and exact-stop receipts before anyone runs the renderer's `--render`; the resulting body must receive independent source review before root's separately owned admission.

## Local checks

Run the focused local identity/template check with the already provisioned project interpreter:

```bash
./.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_nll_fit_v1/bind_nll_fit.py --check
```

The recorded result is in [NLL_LOCAL_CHECKS_V1.json](NLL_LOCAL_CHECKS_V1.json). Ruff format and lint were run only on the new binder. The unchanged source/model/profile suites were not rerun; no B path, SSH, trainer, model/state/cache/dictionary/data payload, inference, or fit was accessed.

## Packet files

- [NLL_RECIPE_BINDING_V1.json](NLL_RECIPE_BINDING_V1.json) freezes the source-qualified NLL arm, exact common recipe, unchanged ancestry, argv delta, and resources.
- [NLL_ADMISSION_TEMPLATE_V1.json](NLL_ADMISSION_TEMPLATE_V1.json) records the distinct learned-dictionary artifact gate and the still-unbound dictionary-child terminal predecessor.
- [NLL_LAUNCH_TEMPLATE.py](NLL_LAUNCH_TEMPLATE.py) is the fail-closed body template; it is not an admitted submission body.
- [bind_nll_fit.py](bind_nll_fit.py) conditionally renders only after exact actual dictionary-child receipts validate.
