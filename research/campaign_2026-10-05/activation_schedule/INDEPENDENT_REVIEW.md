# Independent review: E476 activation × optimizer × LR protocol

**Initial review: NO-GO** because the fit-order provenance receipt described the wrong row-group order. **Resolution below: GO for the sealed actual-shape probe only.** This remains no admission for the twelve full fits.

`fit_cell` shuffles `order` from seed 271828 and trains batches from that order, but returns `fit_row_groups` using `[node["row_group"] for node in nodes]` in original cohort order. The one-update preflight correctly records only the shuffled first B4 groups, so its passing receipt does not catch the full-fit receipt defect. Return `[nodes[index]["row_group"] for index in order]` and add a CPU test that verifies the recorded full traversal against the deterministic seeded shuffle before sealing.

Other inspected paths match the fixed design: every cell clones the same E474 state (including buffers); the 2×2×3 matrix retains all activation, optimizer, and LR cells; the four Muon matrices are precisely the two-layer GRU input/recurrent matrices with three gate blocks, while input/head/bias parameters remain AdamW. The E470 weighted loss and `need_prediction` mask are unchanged; no test groups are requested. Each batch resets recurrent state and carries/detaches it over 79 chunks. The sampled updates 0, 46, 236, and 473 log fit loss, pre-clip global gradient norm, and parameter-delta RMS by four parameter families. The actual per-step trace value is assigned to every optimizer group immediately before each update, and the saved 474-value trace hashes bind those values.

The LR interpretation is properly limited: warm-cosine and constant-mean match arithmetic LR sum, not optimizer-update area; constant-peak has a larger mean LR. Preserve all 12 cells and avoid attributing an optimizer-family ranking to this single seed and schedule matrix. The source's CPU checks verify trace boundaries/sums, parameter partition, loss mask, and activation parity, but currently do not check `fit_cell`'s reported shuffle order.

Source identities reviewed:

- `fit_activation_schedule.py`: `5c79edc4b5d334c220a4d6db3fce54268f2fa9c73605b4ad5461c19059755080`
- `tests.py`: `19ed24de96bd82934dc200a031d5be4daeaddc385094cb879d87f201407263e5`
- `PREREG.md`: `2717180e875347e95b53259d0c2d22b8ebfa7b19b361047eb59fd7b841db7671`


## Resolution review

The trainer now routes both preflight and full fitting through `shuffled_fit_order(nodes)`, which rejects any cohort other than 24 fit-role groups. `fit_cell` records `[nodes[index]["row_group"] for index in order]`, matching the order used to construct B4 batches. The added CPU invariant checks the exact NumPy seeded permutation and verifies that test-role nodes are rejected. This resolves the receipt defect; training mathematics and the fit traversal did not change.

I independently reran `tests.py` under `scripts/limited` with one CPU and a 2 GiB memory cap: `E476 bounded CPU invariants passed`. The clean trainer `--help` invocation passed. `preflight_cpu_v5/fit_manifest.json` reports all 12 cells, all finite one-update losses, all parameter changes, 61,058 parameters per cell, and source hashes matching the current files. Its LR trace receipt still binds all 474 values for every schedule.

Resolution identities:

- `fit_activation_schedule.py`: `5acc0efa8295338b39ffc0a0c9412484daec277967d8dc2e161affee1d1786e6`
- `tests.py`: `dfef4ba199f278dad701fb8c3b005d7ea47f57904b603da5c213c5aa0ffc3a90`
- `preflight_cpu_v5/fit_manifest.json`: `e81d601d5424d21475a04ecba3ba6afbbf9fa76dd4ea6412a4f20bf14a1b9531`
- `preflight_cpu_v5/lr_traces.json`: `046f38f67792af73450bbbb958403834d948495e4ba64b263a2a9a2ff935e4ff`

The actual CUDA B4×256 probe remains mandatory after source sealing. A pass gates shape/resource feasibility only and does not authorize a partial full-fit matrix.
