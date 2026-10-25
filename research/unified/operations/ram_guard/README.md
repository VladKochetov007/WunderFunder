# RAM guard

User-requested periodic RAM check (2026-10-25). [ram_guard.sh](ram_guard.sh) runs as user unit
`wnn-ram-guard` (5% CPU, 100 MB cap): every 120 s it logs used/available memory and sends TERM
only to allowlisted editor language servers above thresholds (cpptools 2 GB, cpptools-srv2
1.5 GB, ty 2 GB, rust-analyzer/pyright/pylance/clangd 3 GB) — VS Code restarts them. It never
kills user applications, VS Code itself, other projects' jobs or `wnn-*` research units; below
6 GB available it only logs the top consumers for root review. Log: `ram_guard.log` (ignored).
