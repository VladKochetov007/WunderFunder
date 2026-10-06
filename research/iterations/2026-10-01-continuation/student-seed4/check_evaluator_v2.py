"""Source/generated-only check; never opens competition arrays or checkpoints."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    output = HERE / "evaluator-v2-source-check.json"
    assert not output.exists()
    a = json.loads((HERE / "amendment-02.json").read_text())
    for name, digest in a["original_preserved_source_sha256"].items():
        assert sha(HERE / name) == digest, name
    old = load("seed4_v1_generated_only", HERE / "evaluate_seed4.py")
    new = load("seed4_v2_generated_only", HERE / "evaluate_seed4_v2.py")
    spec = json.loads((HERE / "evaluation-spec.v2.json").read_text())
    assert new.ROOT == ROOT and new.HERE == HERE
    assert new.MEMBERS == old.MEMBERS and len(new.MEMBERS) == 6
    assert len(new.RECIPES) == 5 and len(new.CONTRASTS) == 4
    assert {k: new.RECIPES[k] for k in old.RECIPES} == old.RECIPES
    assert {k: new.CONTRASTS[k] for k in old.CONTRASTS} == old.CONTRASTS
    assert new.RECIPES["duplicate_seed3"] == (new.MEMBERS[0], new.MEMBERS[0], new.MEMBERS[3], new.MEMBERS[5])
    assert new.CONTRASTS["double_vs_duplicate_seed3"] == ("double_shallow", "duplicate_seed3")
    assert spec["recipes"] == {k: list(v) for k, v in new.RECIPES.items()}
    assert spec["contrasts"] == {k: list(v) for k, v in new.CONTRASTS.items()}
    assert spec["coefficients"] == {k: [.25] * 4 for k in new.RECIPES}
    assert len(spec["predictions"]) == 12
    assert spec["predictions"] == json.loads((HERE / "evaluation-spec.json").read_text())["predictions"]
    for name, digest in spec["sources"].items():
        assert sha(ROOT / name) == digest, name
    assert spec["prospective_amendment"]["sha256"] == sha(HERE / "amendment-02.json")
    # All numerical/path helpers are verbatim AST equivalents of the old evaluator.
    def functions(path):
        return {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef)}
    of, nf = functions(HERE / "evaluate_seed4.py"), functions(HERE / "evaluate_seed4_v2.py")
    assert set(of) == set(nf)
    assert all(of[k] == nf[k] for k in of if k != "main")
    rng = np.random.default_rng(20261001)
    values = {name: rng.normal(size=(12, 211, 2)).astype(np.float32) for name in new.MEMBERS}
    y = rng.normal(size=(12, 211, 2)).astype(np.float32)
    mask = rng.uniform(size=(12, 211)) > .2
    mask[:, :99] = False
    fold = np.repeat(np.arange(2), 6)
    fe = new.metric_module()
    stats = {}
    for recipe, members in new.RECIPES.items():
        literal = np.float32(0) + values[members[0]]
        for member in members[1:]:
            literal = literal + values[member]
        literal = literal / 4
        got = new.blend([values[member] for member in members])
        assert got.dtype == np.float32 and got.tobytes() == literal.tobytes()
        if recipe in old.RECIPES:
            assert got.tobytes() == old.blend([values[member] for member in old.RECIPES[recipe]]).tobytes()
        stats[recipe] = fe.sums(y, got, mask)
    for name, (candidate, reference) in new.CONTRASTS.items():
        got = new.comparison(fe, stats[candidate], stats[reference], fold)
        assert got["delta"] == fe.pooled(stats[candidate].sum(0)) - fe.pooled(stats[reference].sum(0))
        if name in old.CONTRASTS:
            assert got == old.comparison(fe, stats[candidate], stats[reference], fold)
    values[new.MEMBERS[1]] = values[new.MEMBERS[0]].copy()
    same = {k: new.blend([values[m] for m in new.RECIPES[k]]) for k in ("double_shallow", "duplicate_seed3")}
    assert same["double_shallow"].tobytes() == same["duplicate_seed3"].tobytes()
    ss = {k: fe.sums(y, v, mask) for k, v in same.items()}
    zero = new.comparison(fe, ss["double_shallow"], ss["duplicate_seed3"], fold)
    assert zero["delta"] == 0 and all(x["delta"] == 0 for x in zero["by_fold"].values())
    # An unbound preparation file must stop before importing metric/loading arrays.
    original_metric = new.metric_module
    new.metric_module = lambda: (_ for _ in ()).throw(RuntimeError("Unexpected metric/data entry"))
    argv = sys.argv
    sys.argv = ["evaluate_seed4_v2.py", "--bindings", str(HERE / "evaluation-spec.v2.json"),
                "--bindings-sha256", sha(HERE / "evaluation-spec.v2.json"),
                "--output-dir", str(HERE / "unbound-generated-must-not-exist")]
    try:
        try:
            new.main()
        except AssertionError:
            pass
        else:
            raise AssertionError("Unbound draft was accepted")
    finally:
        new.metric_module = original_metric
        sys.argv = argv
    assert not (HERE / "unbound-generated-must-not-exist").exists()
    receipt = {
        "status": "PASS_SOURCE_ONLY_GENERATED_5_PROFILES_4_CONTRASTS",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha(HERE / "evaluate_seed4_v2.py"),
        "spec_sha256": sha(HERE / "evaluation-spec.v2.json"),
        "checker_sha256": sha(__file__), "amendment_sha256": sha(HERE / "amendment-02.json"),
        "bootstrap_sha256": sha(HERE / "bootstrap_seed4_v2.py"),
        "original_preserved_source_sha256": a["original_preserved_source_sha256"],
        "original_helpers_profiles_contrasts_unchanged": True,
        "generated_sequences": 12, "literal_float32_blends": 5, "pooled_contrasts": 4,
        "same_shallow_predictions_new_contrast_exact_zero": True,
        "unbound_preparation_stops_before_data": True,
        "new_fits": 0, "new_predictions": 0, "actual_data_or_checkpoint_reads": False,
        "actual_WP_queries": 0,
        "resource_intent": spec["resource_intent"],
        "qualification": "Preparation only; one future event replaces, rather than supplements, the original event."
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(output), "sha256": sha(output), "status": receipt["status"]}))


if __name__ == "__main__":
    main()
