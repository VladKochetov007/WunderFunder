# Feature successor planning helpers

`panel.py` is a pure, side-effect-free panel and admission helper for the new PR4 and PR14 successors. Add this directory to `sys.path`, then import `panel`; the directory is intentionally flat because the controller loads it as a pinned module bundle.

The caller must pass root-verified owner receipts and the measured density receipt. `next_admissions` never reads files or runtime state and never launches a worker. Eight standard-library CPU tests run with:

```powershell
.venv\Scripts\python.exe research\campaign_2026-10-18\b300_execution_r1\parallel_expansion_v1\code\test_panel.py
```
