h230_source_v3_lineard2z = exact copy of h230_source_v2 plus one change: WSD cooldown `shape` ("sqrt" default,
"linear" new) exposed as `--wsd-shape`. Default behaviour is byte-for-byte the v2 code path. Purpose: matched
from-scratch fold1 teacher with linear decay-to-zero (Bergsma et al. 2025, arXiv 2502.15938) against v2
onecycle/WSD teachers. Created 2026-10-09 by root (Claude Code).
