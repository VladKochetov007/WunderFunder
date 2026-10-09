# H232 V7B selected epoch-3 checkpoint and source review

The terminally downloaded `model_ep3.pt` is the fixed zero-based epoch-3 checkpoint. Its exact SHA-256 is `aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829`, and the hash matches `provenance.json`. A CPU `torch.load(weights_only=True)` review found 16 finite FP32 tensors with 211 input features; checkpoint epoch, run name, `checkpoint_epoch=3`, split digest, and source-qualified membership identity bind to the fit record. No model fit, Parquet read, NPZ load, prediction, provider call, or source mutation occurred.

`model.pt` is a different serialized file digest (`1e26ab6853b5147def2da958f5b63cb1da49e7b3dee5b4b43022615bb3164686`), but its weights-only deserialized payload equals `model_ep3.pt`; use the exact `model_ep3.pt` hash for the selected fixed checkpoint.

The downloaded V7 source archive and manifest hashes match packet metadata. All 54 extracted source files match the archive manifest, including original `scripts/predict_run.py` hash `de4da30aebba33cb72a535b516839e4e627d00f6d6943cb4b2acbca7440b7fee` and `scripts/train_rnn.py` hash `cc0821a144fbea572dfaa03832fbbf31cf1ea605a917b0943c3db5da6b763e1a`.

See the paired JSON for the complete source member and checkpoint tensor inventories. Root owns the export-only successor, quota admission, push, and grade.
