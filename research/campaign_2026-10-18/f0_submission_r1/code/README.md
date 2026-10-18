# Submission evidence code

- [extract_evidence.py](extract_evidence.py) extracts official scored labels from
  original F0 row groups, checks archived prediction hashes, and writes a small
  complete-sequence callback panel. One serial CPU execution completed in9seconds.
- [select_blend.py](select_blend.py) scores only the five planned scalar mixtures
  with the canonical official accumulator. Its result is independently rescored,
  with exposed-fold selection limitations retained.
- [prepare_reference.py](prepare_reference.py) writes the full and panel75/25
  references from the two hash-pinned FP32 prediction arrays.
- [check_native_panel.py](check_native_panel.py) imports the actual delivered ZIP
  in Python3.11, checks every callback, scores with an independently implemented
  metric, compares required-row predictions, checks A-B-A resets and streams full
  F0 outputs. It records hashes, progress, environment and CPU timing.
- [test_check_native_panel.py](test_check_native_panel.py) contains three passing
  synthetic checks; these are separate from actual ZIP execution receipts.
- [replay_release.sh](replay_release.sh) is the fixed sequential ABI, v2, AVX2 and
  complete-F0 replay invocation. The enclosing systemd unit limits CPU, RAM,
  duration and swap. It does not modify training or make network requests.

The canonical [packager](../../../../scripts/package_submission.py) and
[ELF checker](../../../../scripts/check_submission_abi.py) are reused unchanged.
Native libraries and the tracked native wrapper are pinned in the
[asset manifest](../receipts/LOCAL_ASSETS.json), recovered from the preserved
`xteam_h4.zip` and Git revision7ba6940c, respectively.

- [grade_native_result.py](grade_native_result.py) independently recomputes the
  completed native full-F0 score using the canonical metric and original source
  labels, validates complete membership/hash bindings, and exports only scored
  native rows for a separate direct float64 review. Execution is pending.
