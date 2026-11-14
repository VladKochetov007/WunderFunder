# TL3 + W18-final candidates (2026-11-15, CPU only)

User: "сделай лучшие сабмиты из возможных … нам нужно победить" (2026-11-14, ~26 h before the
Nov 15 23:59 UTC deadline). Local GPU stays reserved for PHCapital, so no fits: only fixed
recombination of existing team weights.

## Basis

- Public feedback (user snapshot 2026-11-14): best team ZIP is the original `tl3.zip`
  (`.34 distilled_s5 + .33 fttall_tlead_s5 + .33 fttall_tlead_s6`) at **.66832**. Every newer line
  scored lower: KD students .66108–.66510, teammate W18-final .66453, fullmix .66543.
  Student-slot swaps to our H390 full-data students gave .66792/.66806; TLead pair alone .66771.
- Train-only local counterpart (PR52, exposed valid, not a holdout): `.34 W18 + .33 TLead5 + .33 TLead6`
  = .692377425 versus `.34 fbc1k + …` TL3C .6923359 (+.00004, CI crosses 0).
- W18-final (`tl3final_w18final_distilled_s5@ep1`, train+valid, 12,480 groups, tabdata archive
  `tabdata-w18-final-13-role-evidence-20261103-r1.zip`) has the identical H256×2/211-input NPZ layout as
  the TL3 student slot.

## Candidates

[build.py](build.py) copies runtime files and TL5/TL6/S5 bytes from the original team tl3.zip
(sha256 `d80e69d3…e569`) and W18 bytes from the W18 final package; only `members.json` changes.
Coefficients are fixed a priori (TL3 slot structure), no search. Output in
`submissions/final_20261115/` (git-ignored weights); [BUILD_RECEIPT_R1.json](BUILD_RECEIPT_R1.json).

| ZIP | Members | Expected runtime |
| --- | --- | --- |
| `tl3_w18swap.zip` | .34 W18-final + .33 TL5 + .33 TL6 | ≈ tl3.zip (39 min) |
| `tl3_s5w18.zip` | .17 S5 + .17 W18-final + .33 TL5 + .33 TL6 | ≈ tl3_h390_s3_s4 (52 min, passed) |

`probe_*.zip` are single-member parity probes, not submissions.

## Checks

[parity.py](parity.py) runs in the offline Python 3.11 `tl3c-cpu-receiver` container (1 CPU, 4 GB,
no network): W18 inside the TL3 wrapper versus its own package, candidate versus linear
reconstruction from original tl3 and probes, finite outputs and A-B-A reset. Results:
[PARITY_N2_R1.json](PARITY_N2_R1.json), [PARITY_N16_R1.json](PARITY_N16_R1.json).

## Status (2026-11-14 ~22:20 UTC): both candidates technically checked

16 valid sequences, 318,416 predicted rows: all finite; A-B-A reset exact (0.0) for every ZIP;
candidate minus linear reconstruction ≤1.5e-7; W18 in the TL3 wrapper versus its own q8vnni package
corr .99998 (max abs .138, quantization backend difference). W18 vs original S5 corr .973; swap vs tl3 corr
.9972. Measured runtime ratio versus tl3 is 1.00 (swap) and 1.32 (four-member), i.e. ≈39 and ≈52 min on
the server if tl3.zip's 39m12s scales; the 4-member ZIP keeps ~8 min margin (tl3_h390_s3_s4 passed in
52m11s). Sizes 12.54/16.18 MB < 20 MB. SHA256: swap `b6c80b90…6b83`, four-member `a6e6e7c8…ac00`.
No upload performed.

## Limits

W18-final is fitted on train+valid; original TL5/TL6/S5 fit exposure is not established by the preserved tl3.zip; no held-out score exists for these ZIPs. Expected gain over
tl3.zip is small and unestablished; public is transfer evidence only.
