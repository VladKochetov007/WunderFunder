from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    receipts = []
    for spec_path in sorted(OUT.glob("f0_z3*_v2.json")):
        spec = json.loads(spec_path.read_text())
        fit = next(step for step in spec["steps"] if step["name"] == "fit")
        argv = fit["argv"]
        index = next(i for i, value in enumerate(argv) if value.endswith("/scripts/train_rnn.py"))
        source = Path(argv[index])
        sys.path.insert(0, str(source.parent.parent / "src"))
        for loaded in list(sys.modules):
            if loaded == "wnn" or loaded.startswith("wnn."):
                del sys.modules[loaded]
        module_spec = importlib.util.spec_from_file_location("audited_trainer", source)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        tree = ast.parse(source.read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        stop = next(
            i
            for i, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
        )
        function.body = function.body[: stop + 1] + [ast.Return(ast.Name("args", ast.Load()))]
        tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(tree, str(source), "exec"), module.__dict__)
        sys.argv = [str(source)] + argv[index + 1 :]
        parsed = module.main()
        assert parsed.seed == (3 if "hft" in spec["run_id"] else 4)
        assert parsed.valid_fold == 0
        assert str(parsed.split_manifest).endswith("/fold0.json")
        if spec["run_id"].endswith("student_v2"):
            assert not parsed.include_heldout_valid and not parsed.distill_cache_all_data
            assert parsed.init == "c20261003_combo_s4_f0@ep3"
            assert parsed.epochs == 2 and parsed.checkpoint_epoch == 1
            assert parsed.distill_alpha == 0.7 and parsed.distill_family_weight == 0.1
        receipts.append(
            {
                "spec": spec_path.name,
                "spec_sha256": sha(spec_path),
                "source": str(source),
                "source_sha256": sha(source),
                "parser": "actual source main truncated immediately after parse_args; no data/GPU execution",
                "passed": True,
            }
        )
    parent = ROOT / "artifacts/runs/c20261003_combo_s4_f0"
    provenance = json.loads((parent / "provenance.json").read_text())
    cache = ROOT / "cache/bd26_h230_w384_family_v1/metadata.json"
    metadata = json.loads(cache.read_text())
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    report = {
        "status": "parser_passed_ancestry_review_required",
        "specs": receipts,
        "parent_checkpoint_sha256": sha(parent / "model_ep3.pt"),
        "parent_provenance_sha256": sha(parent / "provenance.json"),
        "parent_split_sha256": provenance.get(
            "split_manifest_sha256", provenance.get("args", {}).get("split_manifest_sha256")
        ),
        "cache_metadata_sha256": sha(cache),
        "cache_split_sha256": metadata["split_manifest_sha256"],
        "split_sha256": sha(split),
        "exposure": "fold0 repeatedly exposed; fold1 planned robustness only",
        "admission": "root pending",
    }
    assert report["parent_split_sha256"] == report["split_sha256"] == report["cache_split_sha256"]
    (OUT / "Z3_EXACT_FOLD0_PARSER_ANCESTRY_PREFLIGHT_R2.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"status": report["status"], "parsers": len(receipts)}))


if __name__ == "__main__":
    main()
