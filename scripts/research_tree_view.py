from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_view(root: Path) -> Path:
    """Embed the preserved tree in an offline, dependency-free HTML explorer."""
    generated = root / "research/bandit_tree/generated"
    source = generated / "tree.json"
    source_bytes = source.read_bytes()
    tree = json.loads(source_bytes)
    ids = {node["id"] for node in tree["nodes"]}
    if len(ids) != len(tree["nodes"]):
        raise ValueError("Tree contains duplicate node identifiers")
    for node in tree["nodes"]:
        if node.get("parent_id") is not None and node["parent_id"] not in ids:
            raise ValueError(f"Missing parent for {node['id']}")
        for row in node.get("raw_rows", []):
            # Records and provenance stay visible; exact bytes live in the source tree.
            row.pop("raw_record", None)
    tree["metadata"].pop("journal_records", None)
    tree["view_provenance"] = {
        "source": "tree.json",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "config_snapshot": tree["metadata"].get("config_snapshot"),
    }
    payload = json.dumps(tree, ensure_ascii=False, allow_nan=False)
    # A ledger string containing HTML must never terminate the JSON script element.
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    template = (root / "scripts/research_tree_view.html").read_text()
    if template.count("__TREE_DATA__") != 1:
        raise ValueError("HTML template must contain exactly one data marker")
    destination = generated / "view.html"
    destination.write_text(template.replace("__TREE_DATA__", payload))
    return destination


def main() -> None:
    """Rebuild the HTML snapshot after research_tree.py build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    destination = build_view(args.root.resolve())
    print(destination)


if __name__ == "__main__":
    main()
