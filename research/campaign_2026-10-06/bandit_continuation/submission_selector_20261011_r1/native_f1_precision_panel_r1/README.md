# Native F1 precision panel: frozen preparation

This packet prepares, but does not run, a native Q8 comparison on a selected
F1 panel. Root owns admission for native inference and grading. No model fit,
provider action, upload, or native callback has been run by this packet.

The panel is the root-owned hash-ranked selection with seed `572911`: 41
complete valid row groups from 14 components (457 scored-row groups in total
across the full fold1 evaluation partition). It is a subset of historically
exposed fold1 data, so results are serving/whole-endpoint discovery, not fresh
CV confirmation. Each candidate's F1 source run records zero fit-membership
overlap with the selected groups.

## Frozen inputs and packages

The source selection receipt is
`references/NATIVE_PRECISION_COMPONENT_PANEL_R1.json` (SHA-256
`4d68036453ca83545efc35be66936ab4f9f821502d2036b8d3b41f28fd194e4f`). The
fold1 split is
`/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-29/artifacts/splits/fold1.json`
(SHA-256
`31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae`). The
feature-only fixture is
`artifacts/features/native_f1_component_panel_41.parquet` (86,429,639 bytes,
SHA-256 `e44fd415caf13d60a09f572c12bd7e94292b6d56234ab53a8880f43e155f83a2`).
Its manifest is `artifacts/NATIVE_F1_FEATURE_PANEL_MANIFEST_R1.json` (SHA-256
`733c7368506b42e7893f37dc07e5841e1c4d152faa03d7a0c5405454a85ba109`). It
contains the 112 model features and causal sequence metadata; it contains no
labels or `is_scored` mask. Fixture creation read only the selected source
groups and used a 1-CPU, 2-GB, 180-second limited scope; observed cgroup peak
was 688,640,000 bytes, elapsed 3.93 seconds, and OOM counters were zero.

The candidate source bindings and package receipts are in
`artifacts/NATIVE_F1_CANDIDATE_CATALOG_R3.json` (SHA-256
`c2a0dd18526f98e23bce64937336125fc12be6680c8640815879d8f64d5c6105`). Its
producer is `freeze_candidate_catalog.py`, SHA-256
`f44c60ed0d91a754de99be46486c34d46e8bd624839d8349bf7e38526da33998`. It pins
each model checkpoint, provenance, history, package, and fixed coefficient.
The three H384 sources are `vt_f1_muonall_h384_e3@ep2` (seed 3; checkpoint
`9c7d5e55fde7112dc691c80670fb17b62f24b9d759d2c90312a9ec89334843c6`),
`vt_f1_muonall_h384_e3_s4@ep2` (seed 4; checkpoint
`752f462036a8e340db00de98dcb141795f3a0ae54cfffce3e95ce7ceed7f5e9e`), and
`vt_f1_muonall_h384_e3_s5@ep2` (seed 5; checkpoint
`ee154f035720069ef26bf0fd328c61887cdbb426c49d51fb25172e5bb52b1f80`). The
H256 F1 prototype is `vt_f1_dsq_dm_t3_a9_e6@ep5`, seed 3, checkpoint
`c8e08b33a316f3539d4381429233b90403b76de83bda6dbecc84d7952f8212ad`; its
actual teacher-cache metadata is pinned to
`a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779`. Its
F1 initializer is `c20261003_combo_s3_f1@ep3`. The student target is the
homogeneous mean of H384 seeds 3/4/5, then scaled by 0.8125; the six-source
cache bookkeeping is not the operation used for this selected homogeneous
target. This F1 prototype is not a matched parent counterpart of the H390
all-data final, whose parent is `vt_all_fin_mu256p0e4_s3@ep3`.

| Fixed candidate | Members and weights | ZIP bytes | ZIP SHA-256 | Role |
|---|---|---:|---|---|
| `f1_teacher_s4_s5_q8_fp32` | H384 s4/s5, 0.5 each | 17,270,998 | `82ec7c90cd984b08f49e4cd5468c3eea828ffb3ac908429bd5eb6d54fc93cff1` | Under-20-MB two-seed reference |
| `f1_teacher_s3_s4_s5_q8_fp32_control` | H384 s3/s4/s5, 1/3 each | 25,255,728 | `28fdea27198457865e465e6e7a17ec7abf0f271159e900821c9c36312e753c5e` | Research-only uncompressed-storage transport control; above 20 MB |
| `f1_teacher_s3_s4_s5_q8_fp16transport` | Same H384 trio, 1/3 each | 13,734,593 | `2140aa860238ba06605b3ceba34e284422b4f34966ed01055b2cfd9a10668bc8` | Under-20-MB selective-FP16-storage trio |
| `f1_h256prototype_half_teachertrio_half_q8_fp16transport` | H256 prototype 0.5; H384 s3/s4/s5 1/6 each | 15,651,823 | `82b52dccabbdb2e87d819980bd5beb6c3272e707cac54016b54dad7905eba1e2` | Prespecified family-mixture prototype; parent mismatch disclosed |

All candidate ZIPs and their build receipts are under this packet's
`artifacts/packages/` directory. Exact basenames, in the same order as the
table, are:

- `f1_teacher_s4_s5_q8_fp32.zip` and `f1_teacher_s4_s5_q8_fp32_BUILD_RECEIPT.json`
- `f1_teacher_s3_s4_s5_q8_fp32_control.zip` and `f1_teacher_s3_s4_s5_q8_fp32_control_BUILD_RECEIPT.json`
- `f1_teacher_s3_s4_s5_q8_fp16transport.zip` and `f1_teacher_s3_s4_s5_q8_fp16transport_BUILD_RECEIPT.json`
- `f1_h256prototype_half_teachertrio_half_q8_fp16transport.zip` and `f1_h256prototype_half_teachertrio_half_q8_fp16transport_BUILD_RECEIPT.json`

For the selective transport children, only
`gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`,
`gru.weight_hh_l1`, and `head.0.weight` use IEEE binary16 storage. The
loader expands them to float32 before the unchanged Q8VNNI runtime. Normalizers,
input/head2 layers, biases, `solution.py`, precision file, and all three native
libraries remain byte-identical to their float32-source package. This is a
disk-transport rounding comparison, not native FP16 inference. The uncompressed
selective-transport H384 trio is 45.6% smaller than its FP32-storage control; that size benefit
has not yet been weighed against prediction fidelity.

Every ZIP was created by the project's `scripts/package_submission.py` with
`--require-provenance --normalized-coefficients --precision q8vnni`. The source
solution is `submission/solution_native.py`, SHA-256
`f025d65abb4990c05ca1677cefd2502340f9eb8cd80180cd38c14044a2c9bf54`. Native
library source is the preserved package
`submissions/upload_2026-10-11/z1c_z4_pair_q8vnni/pkg`, pinned as:

| Library | SHA-256 |
|---|---|
| `v3/wnn_native.abi3.so` | `ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e` |
| `avx2/wnn_native.abi3.so` | `e544ed55538c2c08a7b28389fb2a1cd10a91bf272b7dc4f7cbb7bc3b01ebdaf2` |
| `v2/wnn_native.abi3.so` | `0ea36be1ff45556b157f6295892f44adaa89fcfb8242329bbe530dfae07a5fed` |

## Deferred runner and grade commands

The following are frozen command shapes only. Do not start them until root
records the actual native-panel admission. Each inference is one candidate,
with one CPU, 1.2-GB command memory maximum, and a 600-second runtime ceiling;
root may sequence or pair commands after checking the live aggregate. The
runner asserts the loaded `q8vnni` mode, the selected native library hash, exact
package membership, fixture digests, complete sequence order, warmup behavior,
finite predictions, and writes float32 `[41, 20000, 2]` output plus a receipt.

From `/home/vlad/development/wunder/wnn_native_f1_precision_panel_20261011_r1`:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=600s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/run_native_f1_panel.py \
  --package <exact-candidate-zip> \
  --package-receipt <exact-build-receipt> \
  --panel research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/artifacts/features/native_f1_component_panel_41.parquet \
  --panel-manifest research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/artifacts/NATIVE_F1_FEATURE_PANEL_MANIFEST_R1.json \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --predictions <new-candidate-predictions.npy> \
  --receipt <new-candidate-inference-receipt.json> \
  --level v3
```

After all four predictions have independent receipts, the frozen grader streams
only `t0`, `t1`, `is_scored`, `need_prediction`, and `seq_ix` from the 41 exact
source row groups. It uses `GlobalAccumulator` from the pinned project metric,
`need_prediction AND is_scored`, clipping `[-2, 2]`, absolute clipped-target
magnitude weights, and pooled weighted Pearson per target followed by their
equal mean. It verifies selected rows belong to fold1 evaluation and repeats
the exposed-panel qualification in its output.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=600s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/grade_native_f1_panel.py \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --candidate-catalog research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/artifacts/NATIVE_F1_CANDIDATE_CATALOG_R3.json \
  --feature-manifest research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/native_f1_precision_panel_r1/artifacts/NATIVE_F1_FEATURE_PANEL_MANIFEST_R1.json \
  --split /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-29/artifacts/splits/fold1.json \
  --valid-parquet /home/vlad/development/wunder/wnn_connectome_starterpack/datasets/valid.parquet \
  --candidate-id f1_teacher_s4_s5_q8_fp32 \
  --candidate-id f1_teacher_s3_s4_s5_q8_fp32_control \
  --candidate-id f1_teacher_s3_s4_s5_q8_fp16transport \
  --candidate-id f1_h256prototype_half_teachertrio_half_q8_fp16transport \
  --output <new-grade-receipt.json>
```

## Provenance correction retained

The separate preceding full-data four-member FP16-transport ZIP was rechecked
from disk. Its exact SHA-256 is
`85ccc0cbd3c6830cb6aea84caa57202ec783c8f486b236396134bf970c050486` at
`/home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1/submissions/10-11-exploit-now_r3_fp16transport_q8/artifacts/h390_s3_half_h388_s3_s4_s5_sixth_q8_fp16transport.zip`.
The earlier message spelling ending `...236134...` is incorrect; retain this
corrected digest. This prior artifact is separate from the four F1 ZIPs above.
