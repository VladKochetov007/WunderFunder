"""Source-only fixed online-normalization recipe; never imports Torch or launches jobs."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "research/iterations/2026-10-01-continuation/student"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reference = BASE / "commands.json"
    assert sha(reference) == "f0804cda8b2dde75490496e2014898352ce205fe54837f8c7cbcdfa7823ce30a"
    assert sha(BASE / "build_plan.py") == "6340d0fc811e7fafd9a84dbbe14a2327e5c3c44a5c19b7edeb76314529b32cdf"
    spec = importlib.util.spec_from_file_location("student_source_parser", BASE / "build_plan.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    parser = builder.parser_from_source()
    baseline = json.loads(reference.read_text())
    for p, h in baseline["source_sha256"].items():
        assert sha(ROOT / p) == h, p

    original = (BASE / "predict_fixed.py").read_text()
    derived = original.replace("ROOT=HERE.parents[3]", "ROOT=HERE.parents[2]")
    derived = derived.replace("default=HERE.parent/'cache'", "default=HERE/'cache'")
    derived = derived.replace("campaign=HERE.parent.resolve()", "campaign=HERE.resolve()")
    assert derived != original
    ast.parse(derived)
    (HERE / "predict_fixed.py").write_text(derived)
    diff = "".join(difflib.unified_diff(original.splitlines(True), derived.splitlines(True),
                                     fromfile="student/predict_fixed.py", tofile="online-normalization/predict_fixed.py"))
    (HERE / "predictor.diff").write_text(diff)

    jobs = []
    for old in baseline["jobs"]:
        j = copy.deepcopy(old)
        f = j["fold"]
        parent = f"cont_onsh256_s3_f{f}"
        name = parent if j["stage"] == "parent" else f"cont_onsh256_s3_h2_f{f}"
        j["argv"][j["argv"].index("--name") + 1] = name
        j["argv"].append("--online-norm")
        if j["stage"] == "H2":
            j["argv"][j["argv"].index("--init") + 1] = parent + "@ep3"
            j["dependency"] = parent + ": verified fixed epoch3, n_in323 and online_norm true"
            j["OOF_argv"] = [".venv/bin/python", str((HERE / "predict_fixed.py").relative_to(ROOT)),
                             name, "--fold", str(f), "--epoch", "1", "--batch", "16",
                             "--output-dir", str((HERE / "cache").relative_to(ROOT))]
        values = builder.resolve(parser, j["argv"][2:])
        assert len(values) == 103
        assert builder.resolve(parser, old["argv"][2:]) == old["resolved_args"]
        changes = {k: [old["resolved_args"][k], v] for k, v in values.items() if old["resolved_args"][k] != v}
        assert set(changes) == ({"name", "online_norm"} if j["stage"] == "parent" else {"name", "init", "online_norm"})
        assert values["online_norm"] and values["hidden"] == 256 and values["layers"] == 1
        if j["stage"] == "H2":
            assert values["freeze"] == "inp,gru" and not values["distill"] and not values["ema_teacher"] and not values["since_weight"]
        j.update(name=name, resolved_args=values, reference_job=old["name"], inherited_argument_changes=changes,
                 endpoint=f"artifacts/runs/{name}/model_ep{3 if j['stage'] == 'parent' else 1}.pt", expected_n_in=323)
        jobs.append(j)

    sources = dict(baseline["source_sha256"])
    for p in (Path(__file__), reference, HERE / "predict_fixed.py"):
        sources[str(p.relative_to(ROOT))] = sha(p)
    result = {
        "status": "PREPARED_SOURCE_ONLY_NOT_ALLOCATED", "execution_authorized": False,
        "scientific_change": "Append existing full-prefix causal online_zscore_torch(raw112) to raw112+event99; n_in211->323. No other scientific flags changed.",
        "reference_commands": {"path": str(reference.relative_to(ROOT)), "sha256": sha(reference)},
        "source_sha256": sources, "jobs": jobs,
        "budget": {"fits": 4, "H2_OOF_jobs": 2, "parent_OOF_jobs": 0, "WP_queries_authorized": 0, "retries": 0},
        "environment": baseline["environment"],
        "input_lineage": {"reuse_inventory": "research/iterations/2026-10-01-continuation/student/inventory/receipt.json",
                          "reuse_row_order": "research/iterations/2026-10-01-continuation/student/inventory/row-order-followup.json",
                          "new_resource_and_input_stat_preflight_required": True,
                          "teacher_epoch_limitation": "Historical cached causal h256/h256s4 same-fold teachers retained; exact producer checkpoint epoch was not recorded. No cache regeneration or teacher substitution."},
        "fixed_comparison": {"candidate4": ["cont_onsh256_s3_h2", "on_ctl128s3", "on_ctl128s4", "on_ctlc500"],
                             "current_student4": ["cont_sh256_s3h2", "on_ctl128s3", "on_ctl128s4", "on_ctlc500"],
                             "coefficients_each": [0.25, 0.25, 0.25, 0.25], "folds": [0, 1],
                             "control_reuse": "Previously completed exact shallow256 seed3 H2 OOF; all folds already exposed. No epoch/member/weight search."},
        "predictor_derivation": {"original_sha256": sha(BASE / "predict_fixed.py"), "diff": "predictor.diff",
                                 "changes": ["ROOT one shallower directory", "default output N/cache", "allowed output scope N"]},
        "launch_requirements": ["Separate root allocation required; no launcher is provided here.",
                                "Reject every existing new run directory, log, prediction, partial and receipt; stop on any failure.",
                                "Keep batch32 parent/batch16 H2 and OOF, fast/prefetch, precision, epochs and all other flags unchanged.",
                                "Verify parent EMA epoch3 n_in323, mean/std323 and online_norm before own H2; verify frozen inp/gru/mean/std at H2 epoch1.",
                                "Changed input width changes initialization RNG, normalization sample and shuffled-loader order; this is a recipe utility test, not a strict paired-data ablation.",
                                "GPU memory of cumsum temporaries must fit allocated ceiling; no automatic lower-batch/precision/kernel fallback."]}
    (HERE / "commands.json").write_text(json.dumps(result, indent=2) + "\n")
    receipt = {"status": "PASS_SOURCE_ONLY_FOUR_EXACT103_ARG_DIFFS", "commands_sha256": sha(HERE / "commands.json"),
               "predictor_sha256": sha(HERE / "predict_fixed.py"), "jobs": 4, "OOF_jobs": 2,
               "actual_model_data_GPU_execution": False, "changed_args": [j["inherited_argument_changes"] for j in jobs]}
    (HERE / "source-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
