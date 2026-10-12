# Private Hugging Face model archive

The user authorized uploading all our available Challenge 2 models to `hft-kagglers` on October 12 and explicitly required **private model repositories only**. The destination is [hft-kagglers/wnn-challenge2-model-archive](https://huggingface.co/hft-kagglers/wnn-challenge2-model-archive), created as a private `model` repository. Authenticated members can open it; it is not anonymously accessible.

**Completed:** 21,836,413,179 bytes (21.84 decimal GB) across 8,066 staged payload, metadata and model-card files, verified at revision `167452c72a8e994b280446d2fca93f7c08e16f66`. Every staged file matched remote size and hash; authenticated privacy was true and anonymous access absent. Counts include documentation and are not counts of independently trained models. Existing weights only were transferred; no training or missing-weight reconstruction ran. GNOME accepted completion notification 44. The original inventory remains the frozen discovery snapshot; the final receipt below records delivery completion.

## Inventory and delivery evidence

- [Model inventory](MODEL_INVENTORY_20261012.json) — maps local checkpoint paths and SHA-256 hashes to Hub paths, preserves aliases for identical files, and gives each run a description, provenance/configuration paths, related ledger records and its Hub model card. It includes historical models and continuation states as archival artifacts; inclusion does not endorse their scientific quality.
- [Upload receipt](UPLOAD_RECEIPT_20261012.json) — records each best/local/archive batch, its actual committed revision, verified byte count and privacy check. Only `status: complete` establishes completion of this inventory. Failed or incomplete transfers remain explicit.
- [Preserved first transfer attempt](UPLOAD_RECEIPT_ATTEMPT1_20261012.json) and [resume receipt](RESUME_UPLOAD_RECEIPT_20261012.json) — the first attempt stopped at HF's observed 128 commits/hour action limit after 13.5 GB was verified. The resumed main transfer skips exact files already on the Hub and streams the remaining existing disk weights in fewer commits, retaining failed-attempt evidence and rate-limit backoff. Completion is determined by the resume and final integrity receipts rather than the incomplete first attempt.
- [Worktree candidate inventory](WORKTREE_CANDIDATES_20261012.json) and [payload audit](WORKTREE_AUDIT_20261012.json) — extend discovery to registered project worktrees and local Claude worktrees; compare model hashes to the main archive and identify additional unique model artifacts for transfer. The [superseded first scope audit](WORKTREE_SCOPE_ERROR_20261012.json) incorrectly applied exclusions to the machine's parent directory named `wunder`; it is preserved as invalid execution evidence, and the corrected audit applies exclusions relative to each worktree.
- [Additional worktree model manifest](WORKTREE_UPLOAD_MANIFEST_20261012.json) — records four unique exported model/package artifacts (8,498,526 bytes), their exact local/Hub paths and short source-run descriptions. These preserve historical own online-normalization and gradient-clipping/EMA models; the recorded status distinguishes staging from completed upload.
- [Additional model-state audit](ADDITIONAL_MODEL_AUDIT_20261012.json) — examines initially excluded Torch files through bounded pickle metadata, without loading tensors. It distinguishes standalone saved research/control model parameters, duplicate payloads and mixed input/output captures; eligible additional states retain local paths, hashes and descriptions for archival transfer. Diagnostic inclusion supplies no competition-quality evidence.
- [Supplemental transfer receipt](SUPPLEMENT_UPLOAD_RECEIPT_20261012.json) — records the separate upload of existing worktree packages and standalone saved model states, with file hashes and actual committed revisions. The completed main resume covered its 7,771 main staged files; its supplemental-directory lookup used the wrong parent path. The separate finalizer delivers those staged supplements without repeating the main upload, and the resume implementation now rejects missing supplemental directories.
- [Final privacy and integrity receipt](FINAL_PRIVACY_AND_INTEGRITY_20261012.json) — checks every main and supplemental staged file against one authenticated Hub revision using size and LFS SHA-256 or Git blob identity, and confirms private status and lack of anonymous access. This receipt establishes completion of the whole task.
- [System notification receipt](NOTIFICATION_RECEIPT_20261012.json) — records GNOME notification-service acceptance of critical/completion notices; acceptance does not establish that the user read the notice.
- [Supplemental delivery and final verification implementation](../../../../scripts/hf_model_finalize.py) — transfers the already staged supplemental payloads in bounded batches, verifies the entire archive and records the final revision. No model tensors are constructed or evaluated.
- [Local HF CLI transfer logs](../../../../artifacts/hf_model_upload_20261012/operations/) — preserved stdout/stderr for each numbered upload batch, separate from the hash and privacy receipts. This ignored physical-disk directory contains operational logs, not research grades.
- [Inventory implementation](../../../../scripts/hf_model_inventory.py) and [transfer implementation](../../../../scripts/hf_model_upload.py) — select model artifacts, hash without unpickling, deduplicate exact bytes, stage with physical-disk hardlinks, and use `hf upload` in bounded batches. Model-repository privacy is checked before and after each batch; committed size and SHA-256/Git blob identity are compared to the inventory.
- [Quota-aware resume implementation](../../../../scripts/hf_model_resume.py) — streams already existing files through the Hub SDK, keeps at most 100 files per commit, skips verified remote objects and honors recorded commit backoff. No model construction, fitting or missing-weight recovery is performed. The action limit is an observed provider response; [official rate-limit documentation](https://huggingface.co/docs/hub/rate-limits) explains that granular action limits are distinct from general API quotas.

The installed HF CLI is version 2.2.0. Its first access check found no login; after the user's device connection, `hf auth whoami` returned `vladkoch` with `hft-kagglers` membership. No token is stored in these research notes. The initial exact public-best C ZIP was uploaded at revision `450777ce2ecad64f611586f85b09d7e6eeb9a1c0` and its 12,680,868 bytes and SHA-256 `6013d20a7ff26ce6ef46821f80ed01da848ef4da565e2cd91656417ddd5186ea` matched remote metadata. The complete batch receipt supersedes this initial transfer observation.

## Start with these models

| Purpose | Run / local path | Description and evidence |
|---|---|---|
| Public-best exact endpoint | `submissions/10-12-best-r1/h390_s3_s4_q8_z4half_20261012.zip` | C: user-reported public 0.6671 / 36m36s; exact 0.25 H390 s3 + 0.25 H390 s4 + 0.50 merged Z4 Q8 package. [Package provenance](../../../../submissions/10-12-best-r1/README_R1.md). |
| Public-best source students | `vt_all_h390_ds_homogeneous_s3_r3@ep5`, `vt_all_h390_ds_homogeneous_s4_r3@ep5` | Independently seeded all-data width-256 GRU students; no held-out claim on their fitting data. |
| Public-best merged Z4 member | `bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1` + `z4_seed99_student_hft_alldata_v2@ep0` | Own trunk and fine-tuned head; both are needed to reconstruct the member. |
| Strong local Z4 reference | `oldz4_h230_student_f0_s3_r3@ep1` | Exposed F0 WP 0.6831418134; ancestry and forward precision differ from clean-T3. [Matched contribution screen](../../../campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/README_R1.md). |
| Clean-T3 local students | `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5`, `clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1@ep5` | Exposed F0 0.6811553603 / 0.6812584217; two student/initializer seeds with a shared teacher cohort. [Model evidence](../../archive/2026-10-12/pages/02-models-and-training.md). |
| Clean-F0 teachers | `clean_t3_f0_muonall_h384_e3_s{3,4,5}_r5@ep2` | Independently trained H384 Muon teachers; exposed scores 0.67898918 / 0.67836691 / 0.67969849; equal trio 0.68090199. |
| Cross-fold and capacity ideas | H382 F2/F3, H384/H512/H768 runs, heterogeneous students and historical representation arms | Preserve valuable research directions as well as winning endpoints. Per-run metadata and original H/E records determine what actually ran; a high score on a different fold is not directly comparable. |

## Find a run or download a checkpoint

In the JSON inventory, search `runs` by `run_id`, or search `files` by `local_path`, `sha256` or `hub_path`. Each run has a short `description`, `metadata_paths`, optional `research_records` with H/E IDs and line numbers, and a `hub_card_path`. The canonical remote payload is `files/<original repository path>` unless the entry's `duplicate_of` names an identical copy. Use the stored `hub_path`, which remains correct for duplicates.

```sh
hf auth whoami
hf download hft-kagglers/wnn-challenge2-model-archive \
  files/artifacts/runs/vt_all_h390_ds_homogeneous_s3_r3/model_ep5.pt
rg -n 'vt_all_h390_ds_homogeneous_s3_r3' \
  research/hypotheses.jsonl research/experiments.jsonl research/unified
```

Keep matching provenance and normalization with weights. These custom models load through this repository's code; they are not standard Transformers `AutoModel` exports. Check checkpoint selection and teacher/initializer ancestry before reusing a historical run.

## Scope and limitations

Transfer scope is archival: no new fits, inference, competition submission or paid infrastructure was authorized. The experimental stop remains in force. Supplied data, prediction/target caches, credentials, teammate-weight namespaces, runtime installations and Challenge 3 are excluded. The inventory records considered nonmodel candidates and their exclusion reasons. Large model bytes remain ignored in Git; the inventory and transfer evidence are tracked.

The user also requested a system notification when the entire transfer finishes or a critical problem blocks delivery. Local `notify-send` is installed and the session notification service responded as GNOME Shell. Completion means the main archive, eligible supplemental models and final privacy/integrity checks have finished; the notification receipt will record the actual delivery response.

Local folds remain exposed development evidence. Public results are user-reported portal observations. The archive retains teacher/student and public/local distinctions rather than creating new quality claims. The HF upload behavior and file APIs were checked against [official upload documentation](https://huggingface.co/docs/huggingface_hub/en/guides/upload) and [CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

Return to [model families and training](../README.md) or [ideas and hypotheses](../../ideas/README.md).
