# TL3 source and ancestry closure (metadata-only)

This packet binds what could be recovered from the exact named production runs and source paths without reading checkpoint, prediction-array, or cache payloads. It closes a key command-line ambiguity, but **does not establish a clean F0 counterpart** or complete the source/provenance graph for the incumbent.

## Exact command-parent distinction

The `rs1013/fatl.sh` wrapper launches both tied-lead runs with
`ARGS_FROM=$HOME/WunderFunder-creative/artifacts/runs/fttall_s5/model_ep3.pt`,
`RUN_SEED=5/6`, and `OVERRIDES={"enc":"tiedlead"}`. The copied live trainer
reads `torch.load(ARGS_FROM)["args"]`, applies those environment values and the
explicit override, and prints the fully resolved arguments. Both exact logs
show `init: ""`.

The later model initialization code only calls `torch.load(...)["model"]` and
`model.load_state_dict(...)` inside `if args.init`; when the resolved value is
empty it instead calls `fit_normalization(model, train_loader, device)`. Thus
`fttall_s5@ep3` is the configuration/template donor, **not a tied-lead loaded-
weight ancestor**. The child models start with fresh parameters and fit their
normalization over the production loader. The checkpoint is used as an argument
source; the recipe does not initialize child weights from it. Do not describe
this edge as a warm start.

The two actual named teacher dependencies are `fin256ft_s3` and `fin256ft_s4`.
Each exact resolved log gives the same training scope (`train+valid`,
`include_heldout_valid=true`, `valid_fold=-1`), tied-lead encoder, distillation
alpha 0.5, `prop_f0` weighting/cache, fp16 autocast, four epochs, batch 32,
chunk 1000, and Muon optimizer. `PLE_KNOTS` is empty. The s6 log has two
allocator OOM warnings and the s5 log has two as well.

The `w18final_distilled_s5@ep1` member is different: the recorded resolved args
name `fbc1k_s5@ep3` as its initializer and `w18final_teacher` plus four family
teachers as distillation sources. Under the trainer's `args.init` branch, that
is a true loaded-weight initializer edge. Its checkpoint and ancestors are not
closed.

## Recovered graph and remaining blockers

- **Incumbent blend:** 0.34 `w18final_distilled_s5@ep1`, 0.33
  `fttall_tlead_s5@ep3`, 0.33 `fttall_tlead_s6@ep3`. The user transcript binds
  `tl3.zip`, submission ID `1lmehfi3`, and score 0.66832; the uploaded portal
  bytes are not authenticated by hash.
- **Tied-lead pair:** source wrapper, exact resolved logs and live trainer
  snapshot are present. `fttall_s5@ep3` is a config donor only. `fin256ft_s3`
  and `fin256ft_s4` are direct teachers, but target arrays do not have sidecar
  metadata, so teacher endpoint, source, preprocessing, row order and fitting
  membership remain unknown.
- **Distilled member:** `fbc1k_s5@ep3` is the true model initializer.
  `w18final_teacher`, `w18final_fbc1k_s5`, `w18final_family_full`,
  `w18final_family_raw`, and `orgbase` are named distillation targets. The
  available exact run directories do not contain run provenance or resolved
  configuration files; several teacher run directories are absent. None of
  these ancestor memberships is certified clean-F0.
- **Learned weighting/preprocessing:** both tied-lead logs use `prop_f0`.
  Captured cache sidecars pin model SHA
  `9e92f26d1704f84f9170d746b372c2ba67f7fb611831865a1fc689d6b813b21c` and
  report 10,607 train rows / 10,606 filled and 1,873 valid rows / 1,873 filled,
  but do not prove how that model was fitted or which groups it used. The
  tied-lead model normalization is refit from its full train+valid loader.
- **Historical source identity:** the current live trainer SHA and the live
  modified `src/wnn/nn.py` are captured, including the diff from repository
  HEAD. They do not establish the exact source version used when the runs were
  trained. Checkpoint and prediction payloads were stat-only, never loaded or
  hashed.

All three blended production members use `train+valid`, include the heldout
validation data, and record `valid_fold=-1`. Selecting F0 rows from their output
cannot make them clean held-out controls. The separate canonical F0 contract is
10,535 fit-train plus 1,405 fit-valid groups, with 468 heldout evaluation groups;
the old and canonical split receipts prove equal ordered memberships/component
mapping while retaining different manifest/cache identities. Every initializer,
teacher, teacher-target cache, propensity object, and learned preprocessing
step must be certified against that contract before the new reconstruction can
be called a clean F0 counterpart.

## OOM interpretation

The captured logs contain allocator-level “memory allocation failed with OOM”
warnings followed by later batch progress and full `VALID` records. The copied
trainer's batch loop has no Python OOM catch, batch-skip, or retry branch; an
uncaught batch error would terminate the process. These progress records are
only emitted every 20 batches, and no per-batch/update receipt exists. The
warnings therefore did not terminate the overall runs, but they do not identify
the affected batch or allocator-internal retry outcome, and do not prove every
planned optimizer update completed exactly once. Preserve that limitation.

## Packet contents and method

`SOURCE_CLOSURE.json` is the machine-readable lineage record. `REMOTE_PATH_PROBE.json`
contains exact-path remote `stat`/presence results; `RAW_FILE_IDENTITIES.json`
pins each small copied text/source file to its remote size and SHA-256; and
`CANONICAL_SRC_IDENTITY.json` records the repository commit and dirty-file state.
Raw logs, histories, sidecars, wrapper and relevant source snapshots are under
`raw_metadata/` and `source_snapshot/`. `PACKET_MANIFEST.json` pins every packet
file except itself and records the total bytes.

Collection was read-only. It did not load models, read/hash NPY/NPZ arrays,
recompute predictions, run tests/training, or change remote state. The next
concrete action is to recover exact provenance/config/source and ordered fit
membership for every missing initializer, teacher, target-cache producer and
`prop_f0`; if any ancestor cannot be shown to exclude all 468 F0 evaluation
groups, the exact clean counterpart remains blocked or must be explicitly
renamed as a modified recipe.
