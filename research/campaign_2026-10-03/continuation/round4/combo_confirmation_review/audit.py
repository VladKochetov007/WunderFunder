"""Small JSON/metadata and synthetic arithmetic audit; no real arrays or models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from utils import weighted_pearson  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def fixtures() -> dict:
    path = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
    spec = importlib.util.spec_from_file_location("review_frozen_scorer", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    # Constant one-sided sequences contribute to pooled between-sequence variance.
    y = np.zeros((3, 120, 2), np.float32)
    y[0, 99:] = [1.5, -0.8]
    y[1, 99:] = [-1.0, 1.2]
    y[2, 99:, 0] = np.linspace(-4, 4, 21)
    y[2, 99:, 1] = np.linspace(3, -3, 21)
    p = y * 0.6 + 0.07
    p[1, 99:, 1] -= 0.4
    mask = np.ones((3, 120), bool)
    mask[:, 103::3] = False
    rows = [scorer.score_sequence(y[i], mask[i], {"a": p[i], "b": p[i]}) for i in range(3)]
    moments = np.stack([row["moments"]["a"] for row in rows])
    actual = scorer.pooled(moments)["wp"]
    selected = mask & (np.arange(120)[None] >= 99)
    official = np.mean(
        [weighted_pearson(y[..., k][selected], p[..., k][selected]) for k in range(2)]
    )
    assert abs(actual - official) < 1e-12
    design = {"comparisons": [["a", "b"]], "n_boot": 40, "bootstrap_seed": 271828}
    null = scorer.summary(
        {"a": moments, "b": moments.copy()},
        {},
        {"a": [row["blocks"]["a"] for row in rows], "b": [row["blocks"]["b"] for row in rows]},
        np.array([0, 0, 3]),
        np.array([7, 7, 9]),
        design,
    )["paired_component_bootstrap"]["a_minus_b"]
    assert null["mean_diff"] == 0 and null["ci95"] == [0.0, 0.0] and null["n_components"] == 2
    # Exercise actual class-reweight route, then verify with integer-expanded official rows.
    screen_path = ROOT / "scripts/screen_score.py"
    screen_spec = importlib.util.spec_from_file_location("review_screen", screen_path)
    screen = importlib.util.module_from_spec(screen_spec)
    screen_spec.loader.exec_module(screen)
    classes = np.array([0, 1, 1])
    mix = [0.4, 0.6, 0.0, 0.0]
    reweighted = screen.reweighted(
        scorer, {"a": moments, "b": moments.copy()}, classes, np.array([7, 7, 9]), mix, design
    )
    # Class factors .4/(1/3)=1.2 and .6/(2/3)=.9; integer repeats 4 vs3 cancel common .3.
    expanded_y = np.concatenate(
        [np.tile(y[i, selected[i]], (4 if i == 0 else 3, 1)) for i in range(3)]
    )
    expanded_p = np.concatenate(
        [np.tile(p[i, selected[i]], (4 if i == 0 else 3, 1)) for i in range(3)]
    )
    weighted_official = float(
        np.mean([weighted_pearson(expanded_y[:, k], expanded_p[:, k]) for k in range(2)])
    )
    assert abs(reweighted["exact_global_wp"]["a"]["wp"] - weighted_official) < 1e-12
    return {
        "pooled_score": actual,
        "official_reference": float(official),
        "constant_one_sided_sequences_retained": True,
        "clipping_and_warmup_mask_exercised": True,
        "paired_component_null": null,
        "class_reweighted_score": reweighted["exact_global_wp"]["a"]["wp"],
        "integer_expanded_official_score": weighted_official,
        "pass": True,
    }


def main() -> None:
    started = time.monotonic()
    inputs = read(HERE / "INPUTS.json")
    assert all(sha(HERE / name) == value["sha256"] for name, value in inputs.items())
    report = read(HERE / "combo_fold23_prereg_report.json")
    screen = read(HERE / "combo_fold23_prereg.json")
    assert (
        report["screen"] == screen
        and report["screen_sha256"] == inputs["combo_fold23_prereg.json"]["sha256"]
    )
    assert screen["folds"] == [2, 3] and screen["comparisons"] == [["COMBO", "CTRL"]]
    assert screen["arms"] == {
        "CTRL": {"c20261001_d256_s3_f{fold}": 1.0},
        "COMBO": {"c20261003_combo_s3_f{fold}": 1.0},
    }
    primary = report["pooled"]["paired_component_bootstrap"]["COMBO_minus_CTRL"]
    pub = report["reweighted"]["pub143"]["paired_component_bootstrap"]["COMBO_minus_CTRL"]
    assert primary["n_boot"] == pub["n_boot"] == 2000 and primary["seed"] == pub["seed"] == 271828
    assert primary["centered_moments"] and report["metric_id"] == "global_wp_clip2_v1"
    for part in [report["pooled"], *report["folds"].values(), report["reweighted"]["pub143"]]:
        scores = part["exact_global_wp"]
        for value in scores.values():
            assert abs(value["wp"] - 0.5 * (value["t0"] + value["t1"])) < 1e-12
        delta = part["paired_component_bootstrap"]["COMBO_minus_CTRL"]
        assert abs(delta["mean_diff"] - (scores["COMBO"]["wp"] - scores["CTRL"]["wp"])) < 1e-12
    class_delta = {
        str(c): part["COMBO"]["wp"] - part["CTRL"]["wp"]
        for c, part in report["pooled"]["by_class"].items()
    }
    fold_class3 = {
        fold: part["by_class"]["3"]["COMBO"]["wp"] - part["by_class"]["3"]["CTRL"]["wp"]
        for fold, part in report["folds"].items()
    }
    gates = {
        "positive_primary_mean": primary["mean_diff"] > 0,
        "primary_ci_lower_positive": primary["ci95"][0] > 0,
        "positive_pub143_mean": pub["mean_diff"] > 0,
        "positive_pooled_class3": class_delta["3"] > 0,
        "fold2_class3_not_below_minus002": fold_class3["2"] >= -0.002,
        "fold3_class3_not_below_minus002": fold_class3["3"] >= -0.002,
    }
    assert all(gates.values())
    manifests = {
        f: read(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json")
        for f in (2, 3)
    }
    members, component_sets, row_orders = {}, {}, []
    for fold, manifest in manifests.items():
        path = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
        component_map = {
            (source, node["row_group"]): node["component"]
            for source, nodes in manifest["sequences"].items()
            for node in nodes
        }
        evaluation = set(manifest["evaluation"]["valid"])
        component_sets[fold] = {component_map["valid", group] for group in evaluation}
        for arm, run in (
            ("CTRL", f"c20261001_d256_s3_f{fold}"),
            ("COMBO", f"c20261003_combo_s3_f{fold}"),
        ):
            provenance_path = ROOT / f"artifacts/runs/{run}/provenance.json"
            sidecar_path = ROOT / f"cache/pred_{run}_valid.json"
            provenance, sidecar = read(provenance_path), read(sidecar_path)
            args = provenance["args"]
            identity = report["members"][run]
            assert args["name"] == sidecar["run"] == run and sidecar["split"] == "valid"
            assert (
                args["checkpoint_policy"] == "fixed"
                and args["checkpoint_epoch"] == 3
                and args["no_eval_during_fit"]
            )
            assert provenance["selected_epoch"] == sidecar["epoch"] == identity["epoch"] == 3
            assert (
                args["epochs"] == 4
                and args["seed"] == 3
                and args["hidden"] == 256
                and args["layers"] == 2
            )
            assert args["distill_alpha"] == 0.5 and args["eval_ema"]
            assert (
                sidecar["checkpoint_sha256"]
                == identity["checkpoint_sha256"]
                == provenance["checkpoint_sha256"]["model_ep3.pt"]
            )
            assert sidecar["prediction_sha256"] == identity["prediction_sha256"]
            assert (
                sidecar["split_manifest_sha256"] == provenance["split_manifest_sha256"] == sha(path)
            )
            assert set(provenance["evaluation_row_groups"]) == evaluation
            fitted = set()
            for source in ("train", "valid"):
                groups = set(provenance["fit_row_groups"][source])
                assert groups == set(manifest["fit"][source])
                fitted.update(component_map[source, group] for group in groups)
            assert not fitted & component_sets[fold]
            assert sidecar.get("quote_cache") is None and sidecar["gate_mode"] == "soft"
            row_orders.append(sidecar["row_groups"])
            if arm == "COMBO":
                assert (
                    args["optim"] == "muon"
                    and args["chunk"] == 1000
                    and args["distill_boot_h"] == 8
                )
            else:
                assert (
                    args.get("optim", "adamw") == "adamw"
                    and args["chunk"] == 2000
                    and args.get("distill_boot_h", 0) == 0
                )
            subset_keys = (
                "hidden",
                "layers",
                "batch",
                "chunk",
                "optim",
                "sched",
                "epochs",
                "checkpoint_policy",
                "checkpoint_epoch",
                "distill",
                "distill_alpha",
                "distill_boot_h",
                "ema_teacher",
                "ema_decay",
                "ema_metric_weight",
                "aux_struct",
                "aux_next",
                "propensity",
                "prop_power",
                "prop_eps",
                "valid_true_mask",
                "since_weight",
                "since_floor",
                "fast",
                "eval_ema",
            )
            members[run] = {
                "args": {k: args.get(k) for k in subset_keys},
                "identity": identity,
                "provenance_sha256": sha(provenance_path),
                "sidecar_sha256": sha(sidecar_path),
                "fit_component_exclusion": True,
                "n_fit_train": len(provenance["fit_row_groups"]["train"]),
                "n_fit_valid": len(provenance["fit_row_groups"]["valid"]),
                "n_eval": len(evaluation),
                "update_diagnostics": provenance.get("update_diagnostics"),
                "cumulative_updates": provenance.get("cumulative_updates"),
                "teacher_metadata": args.get("distill_provenance", []),
            }
    assert all(order == row_orders[0] for order in row_orders)
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as target_index:
        # Only this small metadata member is decompressed; no targets/masks/predictions.
        assert row_orders[0] == target_index["row_group"].tolist()
    prediction_headers = {}
    for name in members:
        with (ROOT / f"cache/pred_{name}_valid.npy").open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        assert (
            shape == (len(row_orders[0]), 20000, 2) and dtype == np.dtype("float32") and not fortran
        )
        prediction_headers[name] = {"shape": list(shape), "dtype": str(dtype), "header_only": True}
    assert not component_sets[2] & component_sets[3]
    assert [len(component_sets[f]) for f in (2, 3)] == [144, 145]
    assert primary["n_components"] == 289 and report["n_sequences"] == 948
    prior = {}
    for name in (
        "combo_matched_2seed_report",
        "combo_s3_audit_report",
        "width320_prereg_report",
        "muonfull_report",
        "c1_report",
    ):
        prior_path = ROOT / f"research/campaign_2026-10-03/scores/{name}.json"
        data = read(prior_path)
        prior[name] = {
            "sha256": sha(prior_path),
            "screen": data["screen"],
            "pooled": data["pooled"]["paired_component_bootstrap"],
            "folds": {f: p["paired_component_bootstrap"] for f, p in data["folds"].items()},
            "pooled_class3": data["pooled"]["by_class"]["3"],
            "members": data["members"],
            "pub143": data["reweighted"]["pub143"]["paired_component_bootstrap"],
        }
    teacher_checks = []
    for fold in (2, 3):
        control = members[f"c20261001_d256_s3_f{fold}"]["teacher_metadata"]
        candidate = members[f"c20261003_combo_s3_f{fold}"]["teacher_metadata"]
        assert control == candidate and len(candidate) == 4
        manifest = manifests[fold]
        component_map = {
            (source, node["row_group"]): node["component"]
            for source, nodes in manifest["sequences"].items()
            for node in nodes
        }
        for identity in candidate:
            run = identity["run"]
            sidecar = read(ROOT / f"cache/pred_{run}_{identity['split']}.json")
            teacher = read(ROOT / f"artifacts/runs/{run}/provenance.json")
            assert sidecar["epoch"] == identity["epoch"] == 3
            assert sidecar["checkpoint_sha256"] == identity["checkpoint_sha256"]
            assert sidecar["prediction_sha256"] == identity["prediction_sha256"]
            assert (
                sidecar["split_manifest_sha256"]
                == teacher["split_manifest_sha256"]
                == members[f"c20261003_combo_s3_f{fold}"]["identity"].get(
                    "split_manifest_sha256",
                    sha(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"),
                )
            )
            fitted = {
                component_map[source, group]
                for source in ("train", "valid")
                for group in teacher["fit_row_groups"][source]
            }
            assert not fitted & component_sets[fold]
            teacher_checks.append(
                {
                    "fold": fold,
                    "run": run,
                    "split": identity["split"],
                    "matched_parent_teacher_identity": True,
                    "teacher_fit_component_exclusion": True,
                }
            )
    fixtures_result = fixtures()
    result = {
        "gate_pass": all(gates.values()),
        "gates": gates,
        "primary": primary,
        "pub143": pub,
        "pooled_class_deltas": class_delta,
        "fold_class3_deltas": fold_class3,
        "metadata_members": members,
        "same_sidecar_row_order": True,
        "target_row_index_alignment": True,
        "prediction_headers": prediction_headers,
        "prediction_array_or_checkpoint_tensor_loads": 0,
        "real_score_recomputed": False,
        "prior_comparisons": prior,
        "teacher_metadata_checks": teacher_checks,
        "score_fixtures": fixtures_result,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gates": gates,
                "pooled_class_deltas": class_delta,
                "fold_class3_deltas": fold_class3,
                "score_fixtures": fixtures_result,
                "cost": result["cost"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
