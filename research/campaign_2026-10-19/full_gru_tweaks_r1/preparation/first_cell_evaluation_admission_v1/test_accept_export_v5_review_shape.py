"""Check the new review parser against actual small V3 metadata, without imports."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).parent
PRODUCER = HERE.parent / "first_cell_evaluation_export_v3"
RELATIVE = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
    "first_cell_evaluation_export_v3/EXPORT_BINDING_V3.json"
)


def main() -> None:
    """Exercise only the changed metadata leaf with one actual and three bad records."""
    tree = ast.parse((HERE / "ACCEPT_EXPORT_CAPTURE_V5.py").read_bytes())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reviewed_binding_pin"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<metadata-leaf>", "exec"), namespace
    )
    parser = namespace["reviewed_binding_pin"]
    review = json.loads((PRODUCER / "EXPORT_SOURCE_REVIEW_V3.json").read_bytes())
    assert parser(review, RELATIVE) == (
        15798,
        "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
    )
    duplicate = copy.deepcopy(review)
    duplicate["reviewed_files"].append(duplicate["reviewed_files"][0])
    missing = copy.deepcopy(review)
    missing["reviewed_files"] = [
        item for item in missing["reviewed_files"] if item["path"] != RELATIVE
    ]
    old_shape = {"schema": review["schema"], "status": review["status"], "producer": {}}
    for malformed in (duplicate, missing, old_shape):
        try:
            parser(malformed, RELATIVE)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed source-review binding was accepted")
    print("PASS: actual V3 review + duplicate/missing/old-shape negatives; four cases")


if __name__ == "__main__":
    main()
