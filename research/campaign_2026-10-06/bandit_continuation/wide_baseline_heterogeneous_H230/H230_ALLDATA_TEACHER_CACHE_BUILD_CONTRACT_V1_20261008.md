# H230 all-supplied teacher cache build contract

This is a frozen proposal for root review, not launch authorization. It prepares all supplied train and valid rows for a later own student fit. Since the students will train on the exposed 468 evaluation-valid groups, this data use cannot support a local held-out score claim.

The teachers were trained within clean fold-0 membership (10,535 train and 1,405 valid groups), while their predictions cover the complete target indices (10,606 train and 1,873 valid groups). The existing bd26_h230_w384_family_v1 cache is only the clean-fold subset and is not reusable for this recipe.

Source V3 manifest SHA256: 9938cf880addd22467dd71fc81fc550ea3ab9c7ba792c84cbd73f7073e7de15d. Raw teacher inventory SHA256: 398d7753a127054f3e38cd34c4730099843d60663541787e09ab38c9f45e47ad. Split manifest SHA256: 09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d. The packet rehashed all 55 source files and verified the 16 small prediction sidecars against their recorded hashes; it statted, but did not read or rehash, the large prediction/checkpoint files. Expected content hashes are frozen in the JSON packet and checked by the builder.

| Family | Teacher checkpoint | Coefficient | Checkpoint SHA256 |
|---|---|---:|---|
| strong | c20261003_combo_s3_f0@ep3 | 0.15 | d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e |
| strong | c20261003_combo_s4_f0@ep3 | 0.15 | 0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d |
| full_finetuned | c20260929_ft_since_s3_f0@ep0 | 0.1 | fa9f0353cf6cdd3c85fc56efc1e8a3f3dae0048fb1ccd995eeb7026afd269dc7 |
| full_finetuned | c20260929_ft_since_s4_f0@ep0 | 0.1 | e06b712decb62799c84be6907b8b5285724bbd6efe1f58a4aa1bf4a92f3b11b7 |
| raw_wmse | c20260929_since_s3_f0@ep3 | 0.125 | e3da529a7afe8d78a8cadf4940acacd7ee08c5582e803da7359a7cd9677d3a38 |
| raw_wmse | c20260929_since_s4_f0@ep3 | 0.125 | a1d53aecdef183c94098ffe5fe2422a377a775b022dcc101c48efef5778417e8 |
| samefold_baseline | bd26_w384raw_s3_f0@ep3 | 0.03125 | 90b7357c8d5cf01219764ae58373099710f11d041a01ef5400e683bf854eb08f |
| samefold_baseline | bd26_w384raw_s4_f0@ep3 | 0.03125 | 7e83bebaab7f5de9601b9612ac637a53edd044aec4570d1cf787026b77b43e73 |

The recipe keeps coefficient sum 0.8125, uses strong as the homogeneous comparator, writes source, normalized, homogeneous and raw family targets, and uses chunk size 1. Estimated output is 13,976,480,000 bytes (13.02 GiB). The builder requires a 30 GiB output budget and at least 25 GiB free after writing. The prebuild floor is 40,820,025,600 bytes; current free space at packet time is 63,102,050,304 bytes and must be rechecked before admission.

**The 2 GB worker ceiling has not been profiled.** A prior clean-fold build completed in 67.07s at chunk size 2, but cgroup peak usage reached 6,499,999,744 bytes against a 6,500,000,000-byte reservation. Chunk-size arithmetic is not a memory profile, and the frozen builder has no dry-run/profile-only mode. Root review and a resource decision are required before any launch.

Exact worker argv is in the JSON contract. Contract SHA256: 53bfbdaebf6e3990f4c0df43c067de66703914fadd0045b551650ed6b3513430.
