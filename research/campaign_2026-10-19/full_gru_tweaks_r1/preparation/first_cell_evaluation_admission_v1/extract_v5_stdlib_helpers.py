from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
V5_PATH = (
    REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
    "child_allthree_terminal_verifier_v5/REMOTE_SUBMISSION_V5.py"
)
V5_BYTES = 710_848
V5_SHA256 = "4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374"
HELPER_PATH = Path(__file__).with_name("V5_STDLIB_HELPERS_EXTRACTED.py")
PROVENANCE_PATH = Path(__file__).with_name("V5_STDLIB_HELPER_EXTRACTION_V1.json")
NAMES = (
    "require",
    "digest",
    "json_object",
    "stable_file",
    "file_hash",
    "_bounded_gpu_query",
    "_memory_bytes",
    "_bounded_proc_read",
    "_proc_starttime",
    "show",
    "write_exclusive",
    "write_json_exclusive",
)
PRELUDE = """from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


"""


def extract() -> tuple[str, dict[str, object]]:
    """Copy exact source spans for argument-bound stdlib helpers from frozen V5."""
    source_bytes = V5_PATH.read_bytes()
    if len(source_bytes) != V5_BYTES or hashlib.sha256(source_bytes).hexdigest() != V5_SHA256:
        raise RuntimeError("frozen V5 source bytes changed; do not refresh helper extraction")
    source = source_bytes.decode("utf-8", errors="strict")
    tree = ast.parse(source, filename=str(V5_PATH))
    lines = source.splitlines(keepends=True)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if set(NAMES) - set(functions):
        raise RuntimeError(f"V5 helper names are missing: {set(NAMES) - set(functions)}")
    records: list[dict[str, object]] = []
    chunks: list[str] = [PRELUDE]
    for index, name in enumerate(NAMES):
        node = functions[name]
        snippet = "".join(lines[node.lineno - 1 : node.end_lineno])
        if not snippet.endswith("\n"):
            snippet += "\n"
        call_names = sorted(
            {
                item.func.id
                for item in ast.walk(node)
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
            }
        )
        records.append(
            {
                "name": name,
                "source_lines": [node.lineno, node.end_lineno],
                "source_segment_sha256": hashlib.sha256(snippet.encode()).hexdigest(),
                "called_names": call_names,
                "source_exact": True,
                "uses_v5_module_globals": False,
            }
        )
        chunks.append(snippet)
        if index + 1 < len(NAMES):
            chunks.append("\n\n")
    extracted = "".join(chunks)
    helper_bytes = extracted.encode()
    provenance: dict[str, object] = {
        "schema": "fg19.first-cell-export-v5-stdlib-helper-extraction.v1",
        "source": {
            "path": str(V5_PATH.relative_to(REPOSITORY)),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "extractor": {
            "path": str(Path(__file__).relative_to(REPOSITORY)),
            "names": list(NAMES),
            "source_representation": "exact UTF-8 source spans selected by top-level AST FunctionDef names; no ast.unparse or transformation",
        },
        "helpers": records,
        "output": {
            "path": str(HELPER_PATH.relative_to(REPOSITORY)),
            "bytes": len(helper_bytes),
            "sha256": hashlib.sha256(helper_bytes).hexdigest(),
        },
        "scope": "Only stdlib file/stat/hash, bounded subprocess/systemd, GPU-query and /proc readers are reused. V5 main, payload constants, checker entrypoints and preflight functions are excluded. All new gates take explicit path/identity inputs.",
    }
    return extracted, provenance


def main() -> None:
    extracted, provenance = extract()
    HELPER_PATH.write_text(extracted, encoding="utf-8")
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["output"], sort_keys=True))


if __name__ == "__main__":
    main()
