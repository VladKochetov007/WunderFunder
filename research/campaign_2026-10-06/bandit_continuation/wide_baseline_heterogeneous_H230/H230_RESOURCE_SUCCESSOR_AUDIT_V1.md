# H230 resource successor and family-off control audit

## Findings

The seed-3 H230 candidate remains the frozen science recipe. Its warm start is our own `c20261003_combo_s3_f0@ep3` checkpoint (SHA-256 `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`), explicitly not a C1 student. The successor changes the worker reservation from 7GB to 9GB; the science argv is preserved exactly from the prior seed-3 fit-stage contract. This is a preparation contract only. No GPU fit was launched and no fresh host-side lock, cgroup or GPU admission was available in this shell.

A matched family-off comparison cannot be produced with `--distill-family-weight 0` alone at the pinned source commit. The positive-weight predicate controls family-cache validation/loading, the additional readout rows, warm-start extension, family-target parsing and the family-loss term. The actual candidate profile is GRU 256×2 with 31 outputs. At weight zero, this implementation omits the eight appended family outputs and family cache channels, yielding 23 outputs for the otherwise identical model, and rejects `raw` family target mode. That would change architecture and input path along with the auxiliary loss.

The prepared control therefore specifies a 31-output model with the same family cache channels but zero family-loss coefficient. It records a proposed separate `--distill-family-head` switch. That switch does not exist in the pinned source, so the control is explicitly not runnable until an independently written source revision implements and passes the listed compatibility checks.

## Resource-only successor

| Resource | Requested ceiling |
|---|---:|
| Fit worker | 9,000,000,000 bytes RAM, 3 CPU cores, 2,400s per seed |
| Aggregate | 6 CPU cores, 20,000,000,000 bytes RAM, 5,000,000,000 bytes observed VRAM |
| CUDA allocator | 4,300,000,000 bytes |
| Memory-high reference | 18,000,000,000 bytes, with a fresh ≥750MB projected margin required |

The old 7GB preflight saw projected usage 16,279,917,568 bytes against a page-rounded 17,999,998,976-byte high limit, leaving 1,720,081,408 bytes. If those same conditions persisted, 9GB adds 2,000,000,000 bytes and leaves **-279,918,592 bytes** projected headroom. This exceeds the old high limit by 279,918,592 bytes and is 1,029,918,592 bytes below the minimum 750,000,000-byte margin. This is historical arithmetic only; fresh host state can differ, so root must re-read current cgroup usage/events, support reservations, GPU use and durable attempt/lock state before any launch.

The full-shape profile had reached the previous 6.5GB worker reservation at 6,499,999,744 bytes, measured 2,226,126,848 bytes peak VRAM, and recorded no OOM event increase. The independent profile decision was `PASS_FOR_SERIAL_FIT_ADMISSION_WITH_FRESH_LIVE_MEMORY_CHECK` and still requires fresh admission. The cache build is already present and independently verified at 13,372,801,024 bytes; the fit/export/grade are not part of this contract.

## Recipe and compatibility audit

The eight source coefficients still sum to 0.8125. Main `source` targets are float32 with two target channels; the separate family cache has eight channels in order `strong`, `full_finetuned`, `raw_wmse`, `samefold_baseline`. Verified train shapes are `[10535, 20000, 2]` and `[10535, 20000, 8]`; valid shapes are `[1405, 20000, 2]` and `[1405, 20000, 8]`. The candidate reads the same `source` main blend and raw family targets. The proposed control preserves both cache paths, all 31 model outputs and the seed-3 initialization, while changing only the family-loss multiplier from 0.1 to 0.

| Code input | SHA-256 | Bytes |
|---|---|---:|
| `scripts/train_rnn.py` | `fe283da2d70f370bfc82f0faf9f48a75ee4419cb1615024aa6911ec5f0452677` | 143,763 |
| `scripts/build_distill_cache.py` | `09f7110f21507195199d1f50fad6d1074fef1cd4d670a91093935be333f9dd26` | 22,454 |
| `src/wnn/distillation.py` | `352a0c5b782f9e68fc7153783d20b04825e0c764b94f589cd71993580b7857c7` | 8,577 |
| `src/wnn/distillation_cache.py` | `403fd74db91f08bb255db147fb50f7f6b6b9702280350aaa2479c01a7f972f19` | 10,084 |
| `src/wnn/bandit_campaign_recipe.py` | `883bc6415345ec5e4deff56be13fc0ae0be8791702e999898dfdb3557a5a6374` | 34,735 |

| Frozen evidence | SHA-256 |
|---|---|
| Original H230 execution contract V2 | `d5ade4ab645f06640a5f79e097ef00f428215f31b8941b97228357c5c51d0ddb` |
| Seed-3 fit-stage contract V1 | `8825d2c49b933586acdf47699127154f9ebc5affb595115b6e10727bedf286cc` |
| Source bundle V2 | `c9f4235994e9fc85f0f732fd74852e1c28b1d5c7efc8a1ebefbc71f6dc673340` |
| Cache metadata | `398d7753a127054f3e38cd34c4730099843d60663541787e09ab38c9f45e47ad` |
| Cache verification | `e04da68bad9ddbb0ec64baf1ad43c19381c0606d47dbed066416f84d3a03c3d8` |
| Cache independent review | `91af823661cc93b3c7cccf10ed805d0ec918985695a5c026b5a7cc5dc2f68651` |
| Profile result | `99cbd6fb8f236f8e67e5a5bcf4fa69034a227e5eb3c6864763231fda019f2cb5` |
| Independent profile review | `f103b43ce4048272bf31643dbfb0bd27f12ef1e4930c85c1df059db7b68195ec` |
| Split manifest | `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d` |
| Seed-3 initializer checkpoint | `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e` |
| Pinned source worktree | `dde32d0f5e39efe65c0497fec06449d38e18b369` |

The pinned source worktree has no tracked edits; its status contains only the four expected shared symlink placeholders `.venv`, `artifacts`, `cache` and `datasets`. The exact argv, resource successor, full source map, and control requirements are in [the seed-3 resource successor JSON](H230_SEED3_FIT_RESOURCE_SUCCESSOR_V2.json) and [the matched family-off control JSON](H230_MATCHED_FAMILY_OFF_CONTROL_V1.json).

## Execution state

No seed-3 fit, export, grade, or family-off control was launched here. The recovery handoff last recorded no fit receipt or run directory but could not inspect live host service or GPU lock ownership. A future run needs a fresh root admission under attempt ID `h230-w384-hetero-s3-fit-r9g-v2-20261007`. The future control uses separate ID `h230-w384-hetero-familyoff-s3-v1-20261007` after source implementation and review. No seed 4, export, grade or paid compute is included. No root ledger/specification or external worktree was modified.
