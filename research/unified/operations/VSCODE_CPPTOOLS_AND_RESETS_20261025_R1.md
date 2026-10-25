# VS Code cpptools load and two laptop resets — 2026-10-25

**cpptools.** After each boot the VS Code C/C++ extension re-indexed this workspace: 5.9 GB RSS
and 100–145% CPU, competing with GPU fits for heat budget (package 88–93 °C). Cause: 18,445
C/C++ files visible in the workspace, almost all third-party headers (`.venv` torch/numpy/
pyarrow ≈13.8k, vendored Python 3.11 runtime, preservation copies, `.claude/worktrees`); our own
C++ is `src/wnn/csrc` only. Fix (`.vscode/settings.json`): `C_Cpp.files.exclude` for heavy
trees, `C_Cpp.default.browse.path`/`includePath` pinned to `src/wnn/csrc`, low parsing
priority, 2 threads, 1 cached process, 2 GB IntelliSense limit; watcher/search excludes for
`.venv*`, vendored runtimes, worktrees, submissions. The stale 1.8 GB browse database
(`~/.cache/vscode-cpptools/<hash>`, derived cache) was deleted. Result: cpptools 31 MB, idle.
`ty` servers (2×234 MB) and rust-analyzer (884 MB) were left as they are.

**Resets.** 12:31 EET: clean user restart. 12:52:28 EET: abrupt end of journal (no shutdown,
no thermal/Xid/MCE/OOM entry; last line an unknown key scancode 0x55), ~10 minutes into the
`clip_f0_s1001_noclip_r1` refit both times it was interrupted. Cause unattributed. Response:
partial runs removed and refit; thermal guards tightened to freeze at 92 °C / thaw 84 °C;
GPU-queue CPU quota 120%. If another reset coincides with a fit, stop the GPU queue.
