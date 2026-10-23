# V4 local checks

The only test run for this version was the new focused synthetic fixture:

```text
.venv/bin/python -B -m unittest -v research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v4/test_score_result_capture_v4_bounded_metadata.py
Ran 8 tests in 0.201s
OK
```

Ruff format and lint passed for the V4 template and its fixture:

```text
.venv/bin/ruff format .../SCORE_RESULT_CAPTURE_TEMPLATE_V4.py .../test_score_result_capture_v4_bounded_metadata.py
2 files left unchanged
.venv/bin/ruff check .../SCORE_RESULT_CAPTURE_TEMPLATE_V4.py .../test_score_result_capture_v4_bounded_metadata.py
All checks passed!
```

The fixture uses temporary files/directories and two tiny local Python child
commands to verify output and deadline caps. It does not run the capture
`main()`, inspect a real cgroup, call systemd or NVIDIA utilities, access a
score receipt/result, import project code, or read any B payload. No earlier
version's tests or broader suites were rerun.
