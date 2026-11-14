# Fixed late-trajectory weight mean

Prospective zero-fit amendment to the quality wave's existing trajectory-averaging card. Before deriving weights or exporting these predictions, freeze two endpoints: arithmetic mean of epoch2/epoch3 parameter tensors for `q14s_local_kd90_s3` and `q14s_local_kd100_s3`. No epoch/weight search, learning update, or extra fit. The existing raw fixed-epoch3 endpoints remain unchanged controls. Both parents use the corrected strict-FP32 source48776fba8 and the same clean train-only teacher/split.

Mechanism: aligned late checkpoints may suppress optimization-path variation while retaining one served GRU. Averaging is in weight space within the same trajectory, not a prediction ensemble or an average of independently initialized networks. Normalization buffers must be identical and are preserved. The canonical inference exporter/model and unchanged official metric evaluate each derived endpoint on full official valid. This data is exposed; the two recipes share seed3 and do not constitute distinct-seed confirmation.

[derive_mean.py](derive_mean.py) is preparation code. It binds both original checkpoint hashes to the parent provenance, verifies FP32/layout/argument identity, records derived ancestry and removes inherited quality claims. Derived run names explicitly include `swa23`; `model_ep3.pt` denotes the terminal training boundary but contains the declared epoch2/3 mean, not the raw epoch3 snapshot. No continuation state or copied training history is created. Original files remain unchanged.

Two new fixed selection exposures, zero fits, no automatic final-data refit authority. A derived endpoint needs at least +.0002 over its own raw endpoint to be considered a useful single-model candidate; smaller changes remain recorded but do not earn another recommended ZIP. Any candidate still needs native quality/parity, size and runtime checks. The two-seed cloud gates and final-refit cap are unchanged. Derivation and GPU export wait until the current local paired-fit unit finishes; total local GPU cost remains inside eight hours.

[Serial local exporter](../export_local_r2.sh) first verifies both completed parent optimizer states, creates these two declared means, then exports all four endpoints with the pinned IEEE-FP32 inference wrapper. It refuses to run while the paired-fit unit is active or if final prediction files already exist. Prepared only.

[Prospective confirmation rule R3](../LOCAL_CONFIRMATION_R3.json) allows a fixed seed5 mean only after a useful seed3 effect, with at most two total new parent fits shared with the pure-teacher confirmation. No seed3-only automatic final refit; GPU-hour and final-fit caps stay fixed.

[grade_local.py](grade_local.py) is the prepared full-valid four-endpoint grader. It binds prediction/checkpoint/provenance identities, checks raw-export/history agreement and the independent metric reference, then reports the frozen seed5 triggers. Its output cannot by itself launch a fit or admit final production; resource reservation and eventual second-seed evidence are separate.

## Completed result and independent review

[Four-endpoint grade](LOCAL_ENDPOINTS_R1.json): KD90 raw .6915443853, mean .6915238703 (−.000020515); pure-teacher raw .6915276970, mean .6915636187 (+.000035922). Neither frozen trigger passes; no seed5 pure/SWA confirmation is admitted. Four full-valid GPU exports consumed621.101s wall including thermal idle,356.765s CPU,3.2GiB peak. Raw exports agree with fixed-ep3 history within3e-12.

[Independent review](INDEPENDENT_REVIEW_R1.md) found missing grader source/membership and actual-checkpoint-byte bindings, not a demonstrated wrong export. Root added those checks before grading, including derived-parent receipts. [Actual-producer negative fixtures](check_identity.py) accept the real control and reject wrong source, wrong manifest, valid fitting and mutually consistent false checkpoint hashes ([receipt](IDENTITY_CHECK_R1.json)). Selection thresholds are unchanged. The reviewer did not audit the final patch or independently replay inference.
