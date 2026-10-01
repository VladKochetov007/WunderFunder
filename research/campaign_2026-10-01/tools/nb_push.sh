#!/usr/bin/env bash
# Push cell sources (files named NN_name.py) into the running marimo notebook on port 2721, after the last cell, and run them.
#   nb_push.sh FILE...
set -euo pipefail
files=$(printf '"%s",' "$@")
bash /home/vlad/.claude/skills/marimo-pair/scripts/execute-code.sh --port 2721 <<PY
import marimo._code_mode as cm, pathlib
files = [${files%,}]
async with cm.get_context() as ctx:
    cells = list(ctx.cells)
    for c in cells:
        if c.code.strip() == "" : ctx.delete_cell(c.id)
    last = [c for c in cells if c.code.strip()][-1].id
    existing = {c.name: c.id for c in cells}
    for f in files:
        p = pathlib.Path(f); name = p.stem.split("_", 1)[1]
        code = p.read_text()
        if name in existing:
            ctx.edit_cell(existing[name], code=code); cid = existing[name]
        else:
            cid = ctx.create_cell(code, name=name, after=last)
        ctx.run_cell(cid); last = cid
        print("pushed", name)
PY
