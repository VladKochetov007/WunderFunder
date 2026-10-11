# Clean T3 F0 profile-only terminal capture R1

## Outcome

The root-admitted profile kernel
`vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner` reached
`KernelWorkerStatus.ERROR`. The exact unsuffixed read-only status/quota snapshot
at `2026-10-11T19:08:50+00:00` is
`raw_terminal_capture_r1/RAW_TERMINAL_CAPTURE_STATUS_R2.json`. It reports GPU
usage 62,182.058 of 108,000 seconds, zero reserved, and campaign headroom
31,645.779 seconds. An earlier snapshot reported a larger used counter; these
provider observations are not treated as monotonic. The versioned `/1` status
404 is retained separately and was not substituted for the successful
unsuffixed status.

The profile worker ran two seed-3 T4 profile commands concurrently. Both use
GRU, hidden size 384, two layers, FP16, Muon over all parameters, the
fold-0 propensity input, the recorded weighted-MSE and auxiliary terms, and
`--profile-warmup 5 --profile-steps 25`. Their frozen runtime-spec SHA-256s
are `cee04e56b6e4bf345b321d8e1ad6c0abc413b0cba26212df385c8453f1e6ff7e`
(`...profile1...json`) and
`26d58c837d3993d218650b280c017cb8bb0c36975709bace09dd1e8fe0f3cf18`
(`...profile2...json`). The only argv difference is the run name.

Each exact `argv` is stored in its runtime spec. For clarity, the command is
the following, with `<profile-name>` replaced by either
`clean_t3_f0_muonall_h384_e3_s3_t4_profile1_r1` or
`clean_t3_f0_muonall_h384_e3_s3_t4_profile2_r1`:

```text
{python} research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r1/source_snapshot/scripts/cleanfold_exec.py scripts/train_rnn.py --name <profile-name> --seed 3 --device cuda --arch gru --cell gru --hidden 384 --layers 2 --dropout 0 --train-source train+valid --train-seqs 10607 --valid-seqs 600 --valid-fold 0 --no-eval-during-fit --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json --epochs 3 --checkpoint-policy fixed --checkpoint-epoch 2 --batch 32 --chunk 1000 --loss wmse --loss-norm chunk --lr 0.001 --wd 0.01 --optim muon --muon-scope all --sched onecycle --precision fp16 --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --prop-cache --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 32 --since-floor 0.25 --fast --sync-free --fused-adam --pin-memory --profile-warmup 5 --profile-steps 25
```

At 277.906 seconds, the worker raised `RuntimeError: Timeout for
clean_t3_f0_muonall_h384_e3_s3_t4_profile1_r1` against its 240-second worker
timeout. The outer runner exited with `CalledProcessError`. The result summary
has `finished=[]`; both per-run failure receipts list no artifacts. The last
captured heartbeat shows profile 1 at 210.661 seconds and profile 2 at 180.412
seconds, with measured peak host RSS of 581,152,768 and 587,087,872 bytes,
respectively, and sampled peak GPU allocation of zero bytes. Device inventory
lists two Tesla T4s, each 16,106,127,360 bytes, and host memory with
32,155,762,688 bytes available of 32,212,254,720. Phase progression inside
either training process is unobserved. No fit result or checkpoint was
produced. This is a bounded execution failure, not a model-quality result.

The profile command does not emit a finite-gradient counter or checkpoint
reload evidence; neither is inferred. The run does not establish that a full
three-epoch teacher fits the time budget. Root owns interpretation and any
successor admission.

## Preserved terminal artifacts and bindings

The copied root receipts bind the one-time push and admitted profile-only
scope: dispatch admission SHA-256
`1368c5283b532dbee61567edcf50b595db3dca0f2d9a238d6cc2afdb10a33948`, push
receipt SHA-256
`19eb640c6e8425398a7244c0244154ca23c338dc6151c30f47a1d8c4ce9d82f1`, READY
readback SHA-256
`8c2e4c846584bdb228da40a9c843e9aade4a6b8b354b30a3f3c5fea7484f9f81`, and
source archive SHA-256
`0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f`.

The low-level terminal collection receipt is
`raw_terminal_capture_r1/TERMINAL_COLLECTION_RECEIPT_R1.json`, SHA-256
`46f6a1a9ef1bcfe61c07cc0a8c412812cec3abf5d6bad33e29df1d3be5bfb63c`. It
records successful `kernels output` and `kernels logs` commands and 84 local
output-directory files. That count includes the SDK-generated
`wnn-clean-t3-f0-teachers-20261011-r1-runner.log` sidecar; it is not a count of
provider session-output descriptors. The terminal output and log receipts,
per-file local size/hash manifest, provider logs and all outputs remain under
`raw_terminal_capture_r1/provider_output/attempt_001/`.

The first temporary local target passed to the shared low-level collector
omitted its required `resource` field. The two provider collection commands
had already succeeded and their files were retained, but local receipt
assembly then failed. I corrected only the local target metadata for receipt
serialization and used the existing `_terminal_receipt` helper over the same
preserved attempt; no provider command was reissued for that repair. The
receipt's `collection_complete=true` therefore means only that the shared
collector's logs/output checks passed. It does not certify the independent
session-output inventory gate or profile success.

The earlier read-only session-output inventory is preserved as
`raw_terminal_capture_r1/RAW_SESSION_OUTPUT_API_INVENTORY_R1.json`, SHA-256
`3ff70884e9fecc01b3d4389b28448856f7f691826a153e45a72097af8ed33f84`. It has
83 unique safe relative filenames; all 83 corresponding payload names exist
locally. The only extra local basename is the SDK-generated kernel log sidecar.
This inventory did not retain first-page `response.log` size/SHA, so it cannot
verify that sidecar's bytes against the API response and does not satisfy the
stricter R2 completion rule. The provider API returns no per-file size or
checksum, so even R2 can make only a filename-set comparison plus local
per-file size/SHA recording, not a remote-byte identity claim. The output
request also leaves `version_label` unset; the packet binds current-slug
results to the unique slug's one-time root push v1 and actual returned
spec/source pins, without claiming endpoint version pinning.

Both `kernels files` diagnostics returned an empty list and are preserved as
`RAW_PROVIDER_FILES_INVENTORY_R1.json` (versioned request) and
`RAW_PROVIDER_FILES_INVENTORY_UNSUFFIXED_R1.json`. The installed CLI source
SHA-256 is
`210e589ab1f7359ffb560aacfe1c003ac1bc844c366f4ff709bf3d13de4d5458`; it
shows `kernels output` calls the distinct
`list_kernel_session_output` endpoint. Those empty `kernels files` responses
are therefore not a valid session-output inventory.

Useful raw pins for review: provider logs SHA-256
`14ffc00051ca64e2fb392d1de47b8007cb58b548d4b64d1c515059a5582ef5f5`, result
summary `2686e7743fdc4a82a298a9e20fc7862edcf425d69fc022bd3049c67dfe094e55`,
device/host inventory
`dd025a6e157d073fbe6d56f3f06b5a64583dcd06d2c3b110d409f75164c9bd7f`,
heartbeat `e8a81aba7f74f16edf9595f6c31986165d964bf514121dd2a4560a000dbe8c1d`,
and provider status
`4a92cfaa48da0d7b95c55bfb25c5021a580d656ea581447299b1b4ebad8d121a`.

R2 code must create a fresh `collection_run_NNN/` receipt and verify the actual
session-output endpoint plus response-log sidecar before setting
`collection_complete=true`. Until that fresh receipt exists and passes its
checks, R2 completion remains false. This packet has no authority to rerun,
repair or change the failed profile kernel.
