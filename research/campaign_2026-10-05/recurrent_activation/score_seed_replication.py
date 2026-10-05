"""Independently score the three fixed recurrent-activation fit seeds.

This grader is deliberately source-only until both E475 jobs finish and the
controller explicitly requests grading. It refuses incomplete attempts and
will not overwrite an existing result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ACTIVATION = HERE.parent / "activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ACTIVATION))

import score_activation as common  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402

OUTPUT = HERE / "THREE_SEED_ACTIVATION_RESULT.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
ARMS = tuple(
    f"{alpha}__{optimizer}"
    for alpha in ("m10", "zero", "p10")
    for optimizer in ("adamw", "muon_adamw")
)
SEEDS = (271828, 271829, 271830)
CONTRASTS = (
    ("m10__adamw", "zero__adamw"),
    ("m10__muon_adamw", "zero__muon_adamw"),
    ("p10__adamw", "zero__adamw"),
    ("p10__muon_adamw", "zero__muon_adamw"),
    ("m10__muon_adamw", "m10__adamw"),
    ("zero__muon_adamw", "zero__adamw"),
    ("p10__muon_adamw", "p10__adamw"),
)
EXPECTED_POLICY = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_enabled": True,
    "float32_matmul_precision": "highest",
}
EXPECTED_SOURCE_SHA256 = {
    "fit_recurrent_candidate.py": "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf",
    "tests.py": "266df5c2a7f126aecf67e37416b3ce17cc011620f3848e747b64019b03678015",
    "gpu_probe.py": "aeac54f44ade8370845746a65c3e5d7c7285d3c4241e96c768d73f1a7c5a5b5d",
    "fp32_policy_run.py": "f035341f36fbac6ae7c6074b4ea3656b2e6b91b1967edbec45681a9c310aeb84",
    "fit_seed_replication.py": "cbf82953164f7ca7541afcec1f68e2b25ddc71395994dfe559e5793be6173546",
}


def verify_seal_and_attempt(seed: int) -> tuple[dict, dict, Path]:
    """Require exact sealed inputs/code and a successful, torn-down attempt."""
    if seed == 271828:
        seal_path = HERE / "e471_six_fit_v1.sealed.json"
        attempt_path = (
            ROOT / "research/campaign_2026-10-04/runner/attempts/e471-recurrent-six-fit-v1.json"
        )
        run = HERE / "runs/six_v1"
        expected_name = "e471_recurrent_six_fit_fp32_v1"
        expected_attempt_id = "e471-recurrent-six-fit-v1"
    else:
        seal_path = HERE / f"e475_seed{seed}_v1.sealed.json"
        attempt_path = (
            ROOT
            / "research/campaign_2026-10-04/runner/attempts"
            / f"e475-candidate-seed{seed}-v1.json"
        )
        run = HERE / f"runs/seed{seed}_v1"
        expected_name = f"e475_candidate_seed{seed}_v1"
        expected_attempt_id = expected_name.replace("_", "-")

    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    spec = json.loads(seal_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    if spec.get("name") != expected_name or attempt.get("id") != expected_attempt_id:
        raise ValueError(f"sealed spec or attempt identity differs for seed {seed}")
    if attempt.get("spec") != spec or attempt.get("spec_sha256") != spec_digest(spec):
        raise ValueError(f"attempt does not contain the exact sealed spec for seed {seed}")
    if not (
        attempt.get("status") == "finished"
        and attempt.get("exit_code") == 0
        and attempt.get("teardown_verified") is True
        and attempt.get("failure") is None
    ):
        raise ValueError(f"seed {seed} attempt has not succeeded and torn down")

    for field in ("code", "inputs"):
        for entry in spec[field]:
            path = ROOT / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"sealed {field} file missing: {entry['path']}")
            if path.stat().st_size != entry["size"] or common.sha256_file(path) != entry["sha256"]:
                raise ValueError(f"sealed {field} identity changed: {entry['path']}")

    expected_artifacts = set(spec["artifacts"])
    attempt_artifacts = attempt.get("artifacts")
    if (
        not isinstance(attempt_artifacts, list)
        or {x.get("path") for x in attempt_artifacts} != expected_artifacts
    ):
        raise ValueError(f"runner artifacts do not match the seal for seed {seed}")
    for record in attempt_artifacts:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or common.sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"runner artifact identity changed for seed {seed}: {record['path']}")
    return spec, attempt, run


def load_and_score_seed(
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, list[int], dict[str, str]]:
    """Validate one sealed run and recompute its official scores from exports."""
    spec, attempt, run = verify_seal_and_attempt(seed)
    result_path = run / "result.json"
    report = json.loads(result_path.read_text())
    if report.get("seed") != seed or report.get("metric") != METRIC_ID:
        raise ValueError(f"seed or metric mismatch in result for {seed}")
    if report.get("cohort_sha256") != common.EXPECTED_COHORT_SHA256:
        raise ValueError(f"cohort identity mismatch in result for {seed}")
    if report.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError(f"dataset identity mismatch in result for {seed}")
    if report.get("kind") != "fixed recurrent candidate activation screen":
        raise ValueError(f"unexpected run kind for seed {seed}")

    sealed_code = {item["path"]: item["sha256"] for item in spec["code"]}
    if seed == 271828:
        expected_sources = {
            name: sealed_code[str((HERE / name).relative_to(ROOT))]
            for name in ("fit_recurrent_candidate.py", "tests.py", "gpu_probe.py")
        }
        expected_wrapper = sealed_code[str((HERE / "fp32_policy_run.py").relative_to(ROOT))]
        if report.get("source_sha256") != expected_sources:
            raise ValueError("E471 report source hashes differ from its seal")
        if report.get("precision_wrapper_sha256") != expected_wrapper:
            raise ValueError("E471 precision wrapper hash differs from its seal")
        policy = report.get("explicit_precision_policy", {})
        if policy.get("mode") != "fit" or any(
            policy.get(key) != value for key, value in EXPECTED_POLICY.items()
        ):
            raise ValueError("E471 precision policy differs")
        if (
            report.get("recipe_source_sha256")
            != sealed_code[str((ACTIVATION / "fit_activation_optimizer.py").relative_to(ROOT))]
        ):
            raise ValueError("E471 shared recipe hash differs from its seal")
    else:
        wrapper_path = HERE / "fit_seed_replication.py"
        if (
            report.get("replication_wrapper_sha256")
            != sealed_code[str(wrapper_path.relative_to(ROOT))]
        ):
            raise ValueError("E475 replication wrapper hash differs from its seal")
        if report.get("explicit_precision_policy") != EXPECTED_POLICY:
            raise ValueError("E475 precision policy differs")
        recipe_path = ACTIVATION / "fit_activation_optimizer.py"
        if report.get("recipe_source_sha256") != sealed_code[str(recipe_path.relative_to(ROOT))]:
            raise ValueError("E475 shared recipe hash differs from its seal")
        for relative, expected_hash in EXPECTED_SOURCE_SHA256.items():
            if sealed_code.get(str((HERE / relative).relative_to(ROOT))) != expected_hash:
                raise ValueError(f"E475 source identity differs for {relative}")

    nodes, components = common.load_nodes(run, report)
    records = {item.get("arm"): item for item in report.get("arms", [])}
    if len(report.get("arms", [])) != len(ARMS) or set(records) != set(ARMS):
        raise ValueError(f"wrong arm set for seed {seed}")
    targets: np.ndarray | None = None
    counts_ref: list[int] | None = None
    moments: dict[str, np.ndarray] = {}
    score_records: dict[str, dict] = {}
    export_hashes: dict[str, str] = {}
    for arm in ARMS:
        item = records[arm]
        alpha_text, optimizer = arm.split("__", maxsplit=1)
        expected_alpha = {"m10": -0.1, "zero": 0.0, "p10": 0.1}[alpha_text]
        if (
            item.get("alpha") != expected_alpha
            or item.get("optimizer") != optimizer
            or item.get("parameters") != 8_235
            or item.get("steps") != 474
            or item.get("metric") != METRIC_ID
        ):
            raise ValueError(f"recipe, parameter count or update budget differs for {seed}/{arm}")
        sequences = item.get("sequences", [])
        if [entry.get("row_group") for entry in sequences] != [node["row_group"] for node in nodes]:
            raise ValueError(f"sequence provenance/order differs for {seed}/{arm}")
        counts = [int(entry.get("scored_rows", -1)) for entry in sequences]
        if any(count <= 0 for count in counts) or sum(counts) != item.get("scored_rows"):
            raise ValueError(f"invalid selected-row segmentation for {seed}/{arm}")
        if counts_ref is None:
            counts_ref = counts
        elif counts != counts_ref:
            raise ValueError(f"selected-row segmentation differs for {seed}/{arm}")
        export_path = run / f"{arm}_selected_predictions.npz"
        if not export_path.is_file():
            raise FileNotFoundError(f"selected export missing: {export_path}")
        with np.load(export_path, allow_pickle=False) as archive:
            if set(archive.files) != {"targets", "predictions"}:
                raise ValueError(f"unexpected export schema for {seed}/{arm}")
            y = common.finite_float32(archive["targets"], f"{seed}/{arm} raw selected targets")
            prediction = common.finite_float32(
                archive["predictions"], f"{seed}/{arm} raw selected predictions"
            )
        shape = (sum(counts), 2)
        if y.shape != shape or prediction.shape != shape:
            raise ValueError(f"export shape differs for {seed}/{arm}")
        if targets is None:
            targets = y.copy()
            common.verify_target_export(targets, nodes, counts)
        elif not np.array_equal(targets, y):
            raise ValueError(f"selected labels do not align across arms for seed {seed}")
        official = global_wp(y[None, ...], prediction[None, ...], np.ones((1, len(y)), dtype=bool))
        if not np.allclose(
            [official["wp"], official["t0"], official["t1"]],
            [item["wp"], *item["target_wp"]],
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError(f"official global/target WP differs from fit receipt for {seed}/{arm}")
        moments[arm] = common.sequence_moments(y, prediction, counts)
        score_records[arm] = {
            "wp": official["wp"],
            "t0": official["t0"],
            "t1": official["t1"],
            "selected_rows": len(y),
            "metric_id": METRIC_ID,
        }
        export_hashes[export_path.name] = common.sha256_file(export_path)
        checkpoint = run / f"{arm}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"checkpoint artifact missing for {seed}/{arm}")
        export_hashes[checkpoint.name] = common.sha256_file(checkpoint)

    if targets is None or counts_ref is None:
        raise ValueError(f"no selected exports loaded for seed {seed}")
    target_data_identity = common.sealed_data_identity()
    if target_data_identity["sha256"] != DATASET_SHA256:
        raise ValueError("dataset hash differs from the frozen E470 identity")
    return (
        {
            "seed": seed,
            "attempt_id": attempt["id"],
            "attempt_spec_sha256": attempt["spec_sha256"],
            "fit_result_sha256": common.sha256_file(result_path),
            "source_seal_sha256": common.sha256_file(
                HERE
                / (
                    "e471_six_fit_v1.sealed.json"
                    if seed == 271828
                    else f"e475_seed{seed}_v1.sealed.json"
                )
            ),
            "artifact_sha256": export_hashes,
            "scores": score_records,
        },
        components,
        targets,
        counts_ref,
        moments,
    )


def main() -> None:
    """Grade only after all three attempts have successful verified teardown."""
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seed_results = {}
    shared_components: np.ndarray | None = None
    shared_targets: np.ndarray | None = None
    shared_counts: list[int] | None = None
    all_moments: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        result, components, targets, counts, moments = load_and_score_seed(seed)
        if shared_components is None:
            shared_components, shared_targets, shared_counts = components, targets, counts
        elif (
            not np.array_equal(components, shared_components)
            or not np.array_equal(targets, shared_targets)
            or counts != shared_counts
        ):
            raise ValueError("seed exports do not share exact component/target alignment")
        seed_results[str(seed)] = result
        all_moments[seed] = moments

    assert shared_components is not None
    contrasts: dict[str, dict] = {}
    for seed in SEEDS:
        for candidate, reference in CONTRASTS:
            key = f"seed{seed}:{candidate}_minus_{reference}"
            contrasts[key] = paired_component_bootstrap(
                all_moments[seed][candidate],
                all_moments[seed][reference],
                shared_components,
                n_boot=2_000,
                seed=seed,
                centered=True,
            )

    result = {
        "scope": (
            "18 descriptive scores and 21 within-seed paired contrasts on the same 16 exposed "
            "components; one fitted realization per seed; no pooled CI or public forecast."
        ),
        "metric_id": METRIC_ID,
        "seeds": list(SEEDS),
        "arms_per_seed": list(ARMS),
        "fit_updates_per_arm": 474,
        "parameters_per_arm": 8_235,
        "components": shared_components.tolist(),
        "selected_sequence_counts": shared_counts,
        "attempts_and_scores": seed_results,
        "within_seed_contrasts": contrasts,
        "contrast_count": len(contrasts),
        "paired_component_bootstrap": {
            "replicates_per_contrast": 2_000,
            "centering": True,
            "seed": "fit seed",
            "interpretation": "conditional on each fitted seed and this historically exposed component panel",
        },
        "grader_sha256": common.sha256_file(Path(__file__)),
        "shared_metric_helper_sha256": common.sha256_file(Path(common.__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scores": 18, "within_seed_contrasts": 21, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
