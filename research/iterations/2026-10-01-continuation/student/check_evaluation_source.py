"""Generated-only checks: frozen metric wiring and shared-cache path policy."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import importlib.util
import json
import sys
import tempfile
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
path = HERE / "evaluate_student.py"
ast.parse(path.read_text())
spec = importlib.util.spec_from_file_location("student_eval_generated", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fe = m.metric_module()
rng = np.random.default_rng(9817)
y = rng.normal(size=(5, 137, 2)).astype(np.float32)
mask = rng.random((5, 137)) > .23
mask[:, :99] = False
pred = {name: rng.normal(size=y.shape).astype(np.float32) for name in m.MEMBERS}
pred[m.MEMBERS[0]][0, 103] = [8., -8.]
y[0, 107] = [4., -4.]
checks, stats = [], {}
for name, names in m.RECIPES.items():
    batch = sum([pred[n] for n in names]) / len(names)
    seq = np.concatenate([m.blend([pred[n][i:i+1].astype(np.float32) for n in names])
                          for i in range(len(y))])
    assert batch.dtype == seq.dtype == np.float32 and batch.tobytes() == seq.tobytes()
    a = fe.sums(y, batch, mask)
    b = np.asarray([fe.sums(y[i:i+1], seq[i:i+1], mask[i:i+1])[0] for i in range(len(y))])
    assert a.tobytes() == b.tobytes()
    stats[name] = b
    checks.append({"recipe": name, "blend_bytes_exact": True,
                   "sequence_sufficient_statistics_bytes_exact": True})
folds = np.array([0, 0, 1, 1, 1])
for reference in ("reference4", "reference3"):
    c = m.comparison(fe, stats["candidate4"], stats[reference], folds)
    assert c["delta"] == fe.pooled(stats["candidate4"].sum(0)) - fe.pooled(stats[reference].sum(0))
    for k in (0, 1):
        assert c["by_fold"][str(k)]["delta"] == (
            fe.pooled(stats["candidate4"][folds == k].sum(0)) -
            fe.pooled(stats[reference][folds == k].sum(0)))

original_root, original_cont = m.ROOT, m.CONT
with tempfile.TemporaryDirectory(prefix="student-eval-generated-") as tmp:
    base = Path(tmp)
    creative, shared = base / "creative", base / "shared"
    creative.mkdir()
    shared.mkdir()
    (shared / "fixture.txt").write_text("generated read-only input fixture\n")
    (creative / "cache").symlink_to(shared, target_is_directory=True)
    m.ROOT, m.CONT = creative, creative / "continuation"
    m.CONT.mkdir()
    try:
        registered = m.repository_path("cache/fixture.txt")
        assert registered.resolve() == (shared / "fixture.txt").resolve()
        m.pinned(registered, m.sha(shared / "fixture.txt"), {})
        for bad in ("../shared/fixture.txt", "cache/../fixture.txt", str(shared / "fixture.txt")):
            try:
                m.repository_path(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"Lexical input escape accepted: {bad}")
        assert m.contained_output(m.CONT / "fresh") == (m.CONT / "fresh").resolve()
        (m.CONT / "escape").symlink_to(shared, target_is_directory=True)
        for bad in (m.CONT / "escape" / "output", shared / "output"):
            try:
                m.contained_output(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"Physical output escape accepted: {bad}")
    finally:
        m.ROOT, m.CONT = original_root, original_cont

receipt = {
    "status": "PASS_SOURCE_AND_GENERATED_ONLY",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "adapter_sha256": m.sha(path), "checker_sha256": m.sha(__file__),
    "spec_sha256": m.sha(HERE / "evaluation-spec.json"),
    "metric_source_sha256": m.sha(m.ROOT / "scripts/fold_eval.py"),
    "checks": checks, "both_fixed_comparisons_pooled_and_fold_match_frozen_function": True,
    "symlinked_hash_bound_read_only_input": "PASS",
    "lexically_escaping_inputs_rejected": "PASS", "physically_escaping_outputs_rejected": "PASS",
    "generated_shape": list(y.shape), "generated_seed": 9817,
    "includes_clipping_and_warmup": True, "data_checkpoint_prediction_reads": 0,
    "real_WP_queries": 0,
    "limitations": "No main execution; no actual cache/receipt existence or schema verified; future bindings incomplete.",
}
(HERE / "evaluation-source-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "adapter_sha256": receipt["adapter_sha256"],
                  "receipt_sha256": m.sha(HERE / "evaluation-source-check.json")}))
