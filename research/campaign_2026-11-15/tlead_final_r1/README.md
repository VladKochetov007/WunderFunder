# Full-data TLead seeds for the TL3 family (2026-11-15)

User: "utilize gpu, make experiments" — releases the PHCapital GPU hold
([release receipt](../../campaign_2026-11-14/quality_r2/LOCAL_GPU_RELEASE_R1.json)). Deadline Nov 15 23:59 UTC.

## Hypothesis

Public feedback ranks the original team `tl3.zip` (.34 distilled S5 + .33 `fttall_tlead_s5` + .33
`fttall_tlead_s6`, all-data parents per its description) first at .66832, while every newer student
line scores lower. Its TLead members carry most of its weight. Adding further all-data TLead seeds should
diversify the strongest public family. Its tensors differ from the PR49 train-only `tl3c_fttall_tlead_s5`
reproduction (normalization max diff .23), consistent with different fit data.

## Recipe

[queue.sh](queue.sh): exports full-valid caches of train-only teachers `tt29rep_fin256_s3/s4` (ep3), then
fits `q15f_tlead_s{7,8,9}` with the PR49 `q14t_tlead_s5` recipe (tiedlead H256×2, tl3-muon rnn ×3, onecycle
lr .001, EMA .999 eval, KD α .5 boot-h 8, since32, WMSE + aux, 4 epochs fixed ep3, strict FP32) changed
only to train+valid (10,607 + 1,873 sequences, no evaluation) via the final-fit launcher. Deviation: the
original distilled from `fin256ft`; the local reproduced `fin256` teachers are used. No held-out score is
possible; fitted-valid diagnostics only detect broken training.

## Packaging tools

- [export_npz.py](export_npz.py): tiedlead checkpoint → TL3 `wnn_cpp` member NPZ (keys/shapes asserted
  against the original `weights_tl5.npz`).
- [build_tl3.py](build_tl3.py): TL3 runtime bytes from the original team ZIP + chosen members.
- [parity_torch.py](parity_torch.py): native callback vs torch FP32 predictions on valid sequences.
  Exporter check on PR49 `q14t_tlead_s5`: 4 sequences, 79,604 rows, corr .99993/.99991, mean abs .0052
  (dynamic int8), finite — [PARITY_TORCH_q14t_tlead_s5.json](PARITY_TORCH_q14t_tlead_s5.json).

## Status

Running as `wnn-q15-tlead` since 22:16 UTC Nov 14; s7 at 12.8 seq/s ≈ 65 min per fit. GPU 3.7 GB/75°C,
CPU package 89°C at start.

## Recipe audit (independent, read-only)

Resolved args match `q14t_tlead_s5` except the intended all-data membership, no-eval and teacher swap;
valid-row KD targets load from aligned `pred_*_valid.npy` (`datasets.py:140-160`, `data_loaders.py:82-102`);
`model_ep3.pt` holds EMA weights; strict FP32 confirmed. Low risks: fold-bound teacher sidecar checks are
skipped in the all-data path (teacher valid caches hash-checked manually: both match), teacher fit is in-sample
on train rows only. Trainer is the PR49 source (legacy BF16 Newton–Schulz inside tl3-muon), faithful to how
the original team TLeads were produced.

## CPU evidence on train-only twins (exposed full valid, 1,873 seq / 3,528,273 rows)

[seed_law.py](seed_law.py) → [SEED_LAW_R1.json](SEED_LAW_R1.json): TLead s5 .691415, s6 .690949, equal pair
.691857 (ρ .994), TL3C-torch .692340 (matches native .692336). Pair gain .000675 fits c=.222 in the
k-member law: 3rd seed +.00022, 4th +.00011 on the TLead-only part. Heterogeneity (FBC) adds +.00048 to the pair.

[fourth_member.py](fourth_member.py) → [FOURTH_MEMBER_R1.json](FOURTH_MEMBER_R1.json): fixed .25×4 = TL3C
members + X, all 8 X reported (selection exposure 8): KD90 strict +.000165, KD100 +.000147, legacy KD90
+.000133, KD50 s4/s3 +.000107/+.000094, fbc1k +.000092, fin256 s3/s4 −.00038/−.00060. All gains ~1e-4, far
below the ~.004 public all-data shift; they cannot be distinguished on public.

Decision: after s7, `queue2.sh` (`wnn-q15-queue2`) fits all-data strict KD90 s3 (`q15f_kd90_s3`), then TLead
s8/s9. Linear-GRU NPZ export verified natively: KD90 corr .99997/.99996 ([PARITY_TORCH_kd90.json](PARITY_TORCH_kd90.json)).

## Dependency map ([dependencies.py](dependencies.py) → [DEPENDENCIES_R1.json](DEPENDENCIES_R1.json))

- TL3C TLead-pair share curve is flat: .5 .692321, .6 .692351 (max), .66 .692340, .8 .692230; tl3's weights are ~optimal.
- Strict KD90 as 4th member: b=.1 .692437, .2 .692492, .25 .692504, .3 .692507, .5 .692417.
- b=.25 vs TL3C: +.000164, paired 579-component bootstrap 95% [−.000035, +.000375], P(>0)=.94; t0 +.000235,
  t1 +.000093. Selection exposure: chosen among 8 fourth members (see FOURTH_MEMBER_R1).

**Pre-declared packages (before any all-data fit result):**
- `tl3_tl7`: S5 .34 + TL5/TL6/TL7 .22 each (keeps student share .34, TLead .66).
- `tl3_kd90`: .75×tl3 + .25 all-data KD90 → S5 .255, TL5 .2475, TL6 .2475, KD90 .25.

Thermal: CPU package reached 95°C under the fit; slice quota lowered to 200% and
[thermal_guard.sh](thermal_guard.sh) (`wnn-q15-thermal`) freezes `wnn-campaign.slice` at ≥97°C ×3, thaws ≤88°C.
