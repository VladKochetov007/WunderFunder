from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from scripts.train_rnn import load_component_fit_subset
from wnn.splits import CampaignSplit, load_campaign_split

ROOT = Path(__file__).resolve().parents[5]
DELIVERY = Path(__file__).resolve().parent
DELIVERY_REPO_PATH = Path("research/campaign_2026-10-13/qat_renewable_r1/delivery_r1")
EXPERIMENT_ID = "E-QAT-DEPLOYMENT-20261013-R1"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"


def file_sha256(path: Path) -> str:
    """Hash a frozen artifact in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, maximum_bytes: int = 5_000_000) -> dict[str, object]:
    """Read only bounded metadata; large model and cache payloads stay unopened."""
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"metadata file exceeds its {maximum_bytes}-byte bound: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def verify_file(
    root: Path, relative_path: str, expected_sha256: str, expected_bytes: int | None = None
) -> dict[str, object]:
    """Require exact bytes for one ancestry artifact."""
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required ancestry artifact is missing: {relative_path}")
    size = path.stat().st_size
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 differs for {relative_path}: {digest}")
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"byte length differs for {relative_path}: {size}")
    return {"path": relative_path, "bytes": size, "sha256": digest}


def expect_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {name}")
    return cast(dict[str, object], value)


def expect_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"expected list at {name}")
    return value


def verify_parent_chain(root: Path, split: CampaignSplit) -> dict[str, object]:
    """Verify the F0 CP5 checkpoint, its optimizer-state chain, and initializer."""
    parent_run = "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1"
    p2_run = "clean_t3_f0_control_s3_a9_t3r5_r2_p2_envr1"
    p1_run = "clean_t3_f0_control_s3_a9_t3r5_r2_p1"
    init_run = "c20261003_combo_s3_f0"
    chain_files = [
        verify_file(
            root,
            f"artifacts/runs/{parent_run}/model_ep5.pt",
            "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda",
            3_983_015,
        ),
        verify_file(
            root,
            f"artifacts/runs/{parent_run}/provenance.json",
            "bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961",
            166_152,
        ),
        verify_file(
            root,
            f"artifacts/runs/{p2_run}/training_state.pt",
            "80eb6d3e85646248a997c04f20ca3e3ae61eb5d7fdcd7501752cc47359ad25d6",
            11_825_903,
        ),
        verify_file(
            root,
            f"artifacts/runs/{p2_run}/provenance.json",
            "3e1781b4ff28053016c8517595e8914dce2ce4d9ab2f88ed679e9c558e83e368",
            162_639,
        ),
        verify_file(
            root,
            f"artifacts/runs/{p1_run}/training_state.pt",
            "185c4408a43e69c2f43bcd85a2bc227775aaadaccb34eb4334b6fce211255e7c",
            11_823_087,
        ),
        verify_file(
            root,
            f"artifacts/runs/{p1_run}/provenance.json",
            "13ad38efe7837c04dbf98e766361ca7ab2a7335a6e1604f60f4e0893a2c41e5a",
            158_960,
        ),
        verify_file(
            root,
            f"artifacts/runs/{init_run}/model_ep3.pt",
            "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
            3_977_183,
        ),
        verify_file(
            root,
            f"artifacts/runs/{init_run}/provenance.json",
            "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
            159_653,
        ),
        verify_file(
            root,
            "artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1/model_ep5.pt",
            "e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c",
            3_983_015,
        ),
        verify_file(
            root,
            "artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1/provenance.json",
            "bdd531eb4e1e8ce031fda1b03ecf1c33c133c892768b7d404da30ec69679dd49",
            166_153,
        ),
        verify_file(
            root,
            "artifacts/runs/c20261003_combo_s4_f0/model_ep3.pt",
            "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d",
            3_977_183,
        ),
        verify_file(
            root,
            "artifacts/runs/c20261003_combo_s4_f0/provenance.json",
            "c9fb263175f9b13c5b897f8c24d8f2dbe1164415bfd5adcb22f59dc95a72b332",
            159_653,
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s3_f0/model.pt",
            "d0b3bf32497f22a7fa940ff47b21bbc4ea2414d9f955731d7fbc3e33f2c9350b",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s3_f0/provenance.json",
            "52438c2c9d1a551251aee7a18cc1cca5fce2b9aed55bc558d1adafcae29c166a",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s4_f0/model.pt",
            "25119fefe36f20fb93c8bcafb0748215e24ab956f6944f9ace34d629d36e0d2f",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s4_f0/provenance.json",
            "5ec060b96a202bdd13dc7ec6b03bf1b878ea70811b99d82844a99ce164d8a99c",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt",
            "e3da529a7afe8d78a8cadf4940acacd7ee08c5582e803da7359a7cd9677d3a38",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_since_s4_f0/model_ep3.pt",
            "a1d53aecdef183c94098ffe5fe2422a377a775b022dcc101c48efef5778417e8",
        ),
        verify_file(
            root,
            "artifacts/runs/c20260929_prop_f0/model.pt",
            "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
        ),
    ]

    p3 = read_json(root / f"artifacts/runs/{parent_run}/provenance.json")
    s4 = read_json(
        root / "artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1/provenance.json"
    )
    p2 = read_json(root / f"artifacts/runs/{p2_run}/provenance.json")
    p1 = read_json(root / f"artifacts/runs/{p1_run}/provenance.json")
    combo = read_json(root / f"artifacts/runs/{init_run}/provenance.json")
    combo_s4 = read_json(root / "artifacts/runs/c20261003_combo_s4_f0/provenance.json")
    p3_args = expect_object(p3.get("args"), "P3.args")
    s4_args = expect_object(s4.get("args"), "S4.args")
    p1_args = expect_object(p1.get("args"), "P1.args")
    combo_args = expect_object(combo.get("args"), "initializer.args")
    combo_s4_args = expect_object(combo_s4.get("args"), "S4.initializer.args")
    p3_resume = expect_object(p3.get("resumed_from"), "P3.resumed_from")
    p2_resume = expect_object(p2.get("resumed_from"), "P2.resumed_from")

    if (
        p3_args.get("init") != f"{init_run}@ep3"
        or p3.get("initial_checkpoint_sha256")
        != "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
    ):
        raise ValueError("CP5 provenance does not bind the expected combo-S3 EP3 initializer")
    if (
        p3_resume.get("sha256")
        != "80eb6d3e85646248a997c04f20ca3e3ae61eb5d7fdcd7501752cc47359ad25d6"
    ):
        raise ValueError("CP5 provenance does not bind the expected P2 training state")
    if (
        p2_resume.get("sha256")
        != "185c4408a43e69c2f43bcd85a2bc227775aaadaccb34eb4334b6fce211255e7c"
    ):
        raise ValueError("P2 provenance does not bind the expected P1 training state")
    if p1_args.get("resume_state") is not None or p1_args.get("init") != f"{init_run}@ep3":
        raise ValueError("P1 is not a new optimizer continuation from combo-S3 EP3")
    if p1.get("initial_checkpoint_sha256") != p3.get("initial_checkpoint_sha256"):
        raise ValueError("P1 initializer hash differs from the CP5 lineage")
    if combo_args.get("distill") != "c20260929_since_s3_f0,c20260929_since_s4_f0":
        raise ValueError("combo initializer teacher names changed")
    if (
        s4_args.get("init") != "c20261003_combo_s4_f0@ep3"
        or s4.get("initial_checkpoint_sha256")
        != "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d"
        or combo_s4_args.get("distill") != combo_args.get("distill")
    ):
        raise ValueError("verified seed-4 sibling initializer ancestry changed")
    if p3_args.get("distill_cache_variant") != "homogeneous":
        raise ValueError("CP5 parent target-cache variant changed")

    source_provenances = {
        "CP5": p3,
        "P2": p2,
        "P1": p1,
        "combo initializer": combo,
        "seed-4 CP5 sibling": s4,
        "seed-4 combo initializer": combo_s4,
    }
    for label, provenance in source_provenances.items():
        if provenance.get("split_manifest_sha256") != SPLIT_SHA256:
            raise ValueError(f"{label} provenance is not bound to source-qualified F0")
        fit_rows = expect_object(provenance.get("fit_row_groups"), f"{label}.fit_row_groups")
        split.assert_fit_groups(
            set(map(int, expect_list(fit_rows.get("train"), f"{label}.fit.train"))),
            set(map(int, expect_list(fit_rows.get("valid"), f"{label}.fit.valid"))),
        )
        evaluation_rows = set(
            map(int, expect_list(provenance.get("evaluation_row_groups"), f"{label}.eval"))
        )
        if evaluation_rows != set(split.evaluation_groups):
            raise ValueError(f"{label} evaluation membership differs from source-qualified F0")

    initializer_teachers: list[dict[str, object]] = []
    for teacher_run in ("c20260929_since_s3_f0", "c20260929_since_s4_f0"):
        teacher_provenance = read_json(root / f"artifacts/runs/{teacher_run}/provenance.json")
        teacher_args = expect_object(teacher_provenance.get("args"), f"{teacher_run}.args")
        if (
            teacher_args.get("init")
            or teacher_args.get("resume_state") is not None
            or teacher_provenance.get("initial_checkpoint_sha256") is not None
        ):
            raise ValueError(f"{teacher_run} is not a direct scratch-trained combo teacher")
        if teacher_provenance.get("split_manifest_sha256") != SPLIT_SHA256:
            raise ValueError(f"{teacher_run} is not source-qualified to F0")
        fit_rows = expect_object(
            teacher_provenance.get("fit_row_groups"), f"{teacher_run}.fit_row_groups"
        )
        split.assert_fit_groups(
            set(map(int, expect_list(fit_rows.get("train"), f"{teacher_run}.fit.train"))),
            set(map(int, expect_list(fit_rows.get("valid"), f"{teacher_run}.fit.valid"))),
        )
        initializer_teachers.append(
            {
                "run": teacher_run,
                "scratch_initialized": True,
                "epoch3_checkpoint_sha256": next(
                    item["sha256"]
                    for item in chain_files
                    if item["path"] == f"artifacts/runs/{teacher_run}/model_ep3.pt"
                ),
            }
        )

    prop_provenance = read_json(root / "artifacts/runs/c20260929_prop_f0/provenance.json")
    if prop_provenance.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ValueError("F0 propensity model is not bound to the source-qualified split")

    return {
        "parent_run": parent_run,
        "parent_epoch": 5,
        "parent_checkpoint_sha256": chain_files[0]["sha256"],
        "parent_provenance_sha256": chain_files[1]["sha256"],
        "source_args": {
            "CP5_seed3": p3_args,
            "P2_seed3": expect_object(p2.get("args"), "P2.args"),
            "P1_seed3": p1_args,
            "combo_initializer_seed3": combo_args,
            "CP5_seed4_sibling": s4_args,
            "combo_initializer_seed4": combo_s4_args,
        },
        "combo_seed3_teacher_prediction_ancestry": combo.get("distill_provenance"),
        "resume_chain": [
            {"run": p1_run, "resume_state": None, "initializer": f"{init_run}@ep3"},
            {
                "run": p2_run,
                "resume_state": f"artifacts/runs/{p1_run}/training_state.pt",
                "resume_state_sha256": p2_resume["sha256"],
                "next_epoch": p2_resume.get("next_epoch"),
            },
            {
                "run": parent_run,
                "resume_state": f"artifacts/runs/{p2_run}/training_state.pt",
                "resume_state_sha256": p3_resume["sha256"],
                "next_epoch": p3_resume.get("next_epoch"),
            },
        ],
        "initializer": {
            "run": init_run,
            "epoch": 3,
            "checkpoint_sha256": chain_files[6]["sha256"],
            "provenance_sha256": chain_files[7]["sha256"],
            "teacher_runs": combo_args["distill"],
            "teacher_initialization": initializer_teachers,
        },
        "verified_f0_seed4_sibling": {
            "run": "clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1",
            "epoch": 5,
            "checkpoint_sha256": chain_files[8]["sha256"],
            "provenance_sha256": chain_files[9]["sha256"],
            "initializer": "c20261003_combo_s4_f0@ep3",
            "initializer_checkpoint_sha256": chain_files[10]["sha256"],
            "initializer_provenance_sha256": chain_files[11]["sha256"],
        },
        "verified_files": chain_files,
    }


def verify_teacher_cache(root: Path, split: CampaignSplit) -> dict[str, object]:
    """Bind target metadata and teacher ancestry without reading target arrays."""
    cache = root / "cache/clean_t3_f0_h390_t3r5_a9_r2"
    metadata_path = cache / "metadata.json"
    metadata_sha = "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8"
    if file_sha256(metadata_path) != metadata_sha:
        raise ValueError("teacher-cache metadata SHA-256 differs")
    metadata = read_json(metadata_path)
    if metadata.get("split_manifest_sha256") != SPLIT_SHA256 or metadata.get("fold") != 0:
        raise ValueError("teacher cache is not bound to source-qualified F0")
    fit_groups = expect_object(metadata.get("fit_groups"), "cache.fit_groups")
    if set(map(int, expect_list(fit_groups.get("train"), "cache.fit_groups.train"))) != set(
        split.train_groups
    ):
        raise ValueError("teacher-cache train membership differs from source-qualified F0")
    if set(map(int, expect_list(fit_groups.get("valid"), "cache.fit_groups.valid"))) != set(
        split.valid_groups
    ):
        raise ValueError("teacher-cache valid membership differs from source-qualified F0")
    if set(map(int, expect_list(metadata.get("evaluation_valid_groups"), "cache.eval"))) != set(
        split.evaluation_groups
    ):
        raise ValueError("teacher-cache evaluation group identity changed")
    qualification = expect_object(metadata.get("source_qualification"), "cache.qualification")
    if qualification.get("evaluation_valid_groups_in_cache") != 0:
        raise ValueError("evaluation groups entered fitted teacher-cache targets")
    files = expect_object(metadata.get("files"), "cache.files")
    payloads: list[dict[str, object]] = []
    for key in ("main_homogeneous_train", "main_homogeneous_valid"):
        record = expect_object(files.get(key), f"cache.files.{key}")
        payload_path = cache / str(record["path"])
        if not payload_path.is_file() or payload_path.stat().st_size != record["bytes"]:
            raise ValueError(f"teacher-cache payload size differs: {payload_path}")
        payloads.append(
            {
                "path": str(payload_path.relative_to(root)),
                "bytes": record["bytes"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "sha256_from_metadata": record["sha256"],
                "payload_hashed": False,
            }
        )
    sources = expect_list(metadata.get("sources"), "cache.sources")
    teachers: list[dict[str, object]] = []
    for source_value in sources:
        source = expect_object(source_value, "cache.source")
        checkpoint = str(source["checkpoint_path"])
        provenance = str(source["provenance_path"])
        teachers.append(
            {
                "run": source["run"],
                "seed": source["seed"],
                "epoch": source["epoch"],
                "coefficient": source["coefficient"],
                "checkpoint": verify_file(
                    root,
                    str(Path(checkpoint).relative_to(root)),
                    str(source["checkpoint_sha256"]),
                ),
                "provenance": verify_file(
                    root,
                    str(Path(provenance).relative_to(root)),
                    str(source["provenance_sha256"]),
                ),
                "train_count": expect_object(source["fit_group_memberships"], "source.membership")[
                    "train"
                ]["count"],
                "valid_count": expect_object(source["fit_group_memberships"], "source.membership")[
                    "valid"
                ]["count"],
                "members_match_f0": all(
                    bool(
                        expect_object(source["fit_group_memberships"], "source.membership")[part][
                            "member_set_matches_f0"
                        ]
                    )
                    for part in ("train", "valid")
                ),
            }
        )
    if len(teachers) != 3 or not all(teacher["members_match_f0"] for teacher in teachers):
        raise ValueError("clean-T3 teacher sources are not the exact source-qualified F0 cohort")
    return {
        "metadata_path": str(metadata_path.relative_to(root)),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": metadata_sha,
        "profile": metadata.get("cache_profile"),
        "fit_group_counts": {
            "train": len(split.train_groups),
            "valid": len(split.valid_groups),
        },
        "evaluation_group_count": len(split.evaluation_groups),
        "homogeneous_scale": metadata.get("homogeneous_scale"),
        "target_contract": metadata.get("target_contract"),
        "payloads": payloads,
        "teachers": teachers,
    }


def verify_subsets(delivery: Path, split: CampaignSplit) -> dict[str, object]:
    """Validate all nested manifests against the same split and evaluation panel."""
    subset_root = delivery / "subsets/f0_s3"
    names = {
        "10pct": "fit_components_10pct.json",
        "30pct": "fit_components_30pct.json",
        "100pct": "fit_components_100pct.json",
    }
    loaded: dict[str, dict[str, object]] = {}
    rows: dict[str, dict[str, object]] = {}
    for label, filename in names.items():
        path = subset_root / filename
        digest = file_sha256(path)
        subset = load_component_fit_subset(path, split, expected_experiment_id=EXPERIMENT_ID)
        manifest = expect_object(subset["manifest"], f"subset.{label}.manifest")
        loaded[label] = manifest
        rows[label] = {
            "path": str(path.relative_to(delivery)),
            "sha256": digest,
            "selected_components": manifest["selected_component_count"],
            "train_groups": len(expect_list(manifest["selected_groups"]["train"], label)),
            "valid_groups": len(expect_list(manifest["selected_groups"]["valid"], label)),
            "selected_members_sha256": manifest["selected_members_sha256"],
            "evaluation_groups": manifest["evaluation_group_count"],
        }
    sets = {
        label: set(map(int, expect_list(manifest["selected_components"], label)))
        for label, manifest in loaded.items()
    }
    if not sets["10pct"].issubset(sets["30pct"]) or not sets["30pct"].issubset(sets["100pct"]):
        raise ValueError("component fit subsets are not nested")
    fit_members = {("train", group) for group in split.train_groups} | {
        ("valid", group) for group in split.valid_groups
    }
    fit_components = {
        component for source, group, component in split.components if (source, group) in fit_members
    }
    evaluation_components = {
        component
        for source, group, component in split.components
        if source == "valid" and group in split.evaluation_groups
    }
    if fit_components.intersection(evaluation_components):
        raise ValueError("campaign fitting components overlap fixed evaluation components")
    if sets["100pct"] != fit_components:
        raise ValueError("full fitting-component subset does not match the campaign split")
    return {
        "experiment_id": EXPERIMENT_ID,
        "split_manifest_sha256": SPLIT_SHA256,
        "selection_seed": 20261013,
        "subsets": rows,
        "nested": True,
    }


def option_value(argv: list[object], option: str) -> object:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"recipe command is missing {option}") from error


def normalized_train_argv(argv: list[object]) -> list[object]:
    """Remove run-specific fields before comparing matched commands."""
    omitted = {"--name", "--qat", "--fit-subset-manifest"}
    normalized: list[object] = []
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value in omitted:
            skip_next = True
            continue
        normalized.append(value)
    return normalized


def verify_run_recipe(delivery: Path) -> dict[str, object]:
    """Check exact matched fit/profile vectors and post-fit prediction commands."""
    recipe = read_json(delivery / "RUN_RECIPE_R1.json", maximum_bytes=1_000_000)
    if recipe.get("schema") != "wnn.qat-matched-delivery-recipe.v1":
        raise ValueError("QAT delivery recipe schema changed")
    fits = expect_list(recipe.get("fits"), "recipe.fits")
    profiles = expect_list(recipe.get("profiles"), "recipe.profiles")
    evaluations = expect_list(recipe.get("postfit_evaluation"), "recipe.postfit_evaluation")
    expected_modes = {
        "fp32": "none",
        "uniform_w4": "recurrent_w4",
        "mixed_w4w8": "input_w4_recurrent_w8",
    }
    if len(fits) != 9 or len(profiles) != 3 or len(evaluations) != 9:
        raise ValueError("QAT delivery recipe must contain 9 fits, 3 profiles, and 9 evaluations")
    run_names: set[str] = set()
    fit_vectors: dict[str, list[list[object]]] = {}
    for value in fits:
        item = expect_object(value, "recipe.fit")
        argv = expect_list(item.get("argv"), "recipe.fit.argv")
        if item.get("condition") not in expected_modes:
            raise ValueError("unknown QAT delivery condition")
        condition = str(item["condition"])
        if item.get("qat") != expected_modes[condition]:
            raise ValueError("fit command QAT mode does not match its condition")
        if item.get("name") != option_value(argv, "--name"):
            raise ValueError("fit name and trainer argument differ")
        if option_value(argv, "--qat") != expected_modes[condition]:
            raise ValueError("trainer QAT argument differs from the recipe label")
        if option_value(argv, "--precision") != "fp32" or option_value(argv, "--epochs") != "1":
            raise ValueError("matched QAT fit must be one-epoch FP32")
        if option_value(argv, "--batch") != "16" or option_value(argv, "--chunk") != "1000":
            raise ValueError("matched QAT fit batch/chunk shape changed")
        if option_value(argv, "--lr") != "0.0001" or option_value(argv, "--optim") != "adamw":
            raise ValueError("matched QAT fit optimizer configuration changed")
        if "--save-training-state" not in argv:
            raise ValueError("fit command must save its epoch-boundary training state")
        if item.get("name") in run_names:
            raise ValueError("duplicate QAT run name")
        run_names.add(str(item["name"]))
        fit_vectors.setdefault(str(item["subset"]), []).append(normalized_train_argv(argv))
    if any(
        len(vectors) != 3 or any(vector != vectors[0] for vector in vectors[1:])
        for vectors in fit_vectors.values()
    ):
        raise ValueError("matched QAT conditions do not share identical trainer arguments")

    profile_modes: set[str] = set()
    for value in profiles:
        item = expect_object(value, "recipe.profile")
        argv = expect_list(item.get("argv"), "recipe.profile.argv")
        condition = str(item.get("condition"))
        if condition not in expected_modes or condition in profile_modes:
            raise ValueError("profile condition is missing or duplicated")
        profile_modes.add(condition)
        if option_value(argv, "--qat") != expected_modes[condition]:
            raise ValueError("profile QAT mode differs from its condition")
        if (
            option_value(argv, "--profile-warmup") != "2"
            or option_value(argv, "--profile-steps") != "3"
        ):
            raise ValueError("profile batch budget changed")
        expected_subset = (
            f"{DELIVERY_REPO_PATH}/subsets/f0_s3/fit_components_10pct.json"
        )
        if option_value(argv, "--fit-subset-manifest") != expected_subset:
            raise ValueError("profiles must use the frozen 10% component subset")
    if profile_modes != set(expected_modes):
        raise ValueError("one actual-shape profile per QAT condition is required")

    eval_names: set[str] = set()
    for value in evaluations:
        item = expect_object(value, "recipe.evaluation")
        predict = expect_list(item.get("prediction_argv"), "recipe.prediction.argv")
        score = expect_list(item.get("score_argv"), "recipe.score.argv")
        name = str(item.get("name"))
        if name not in run_names or name in eval_names:
            raise ValueError("evaluation run name is missing or duplicated")
        eval_names.add(name)
        if "--campaign-full-valid-feature-only" not in predict or name not in predict:
            raise ValueError("post-fit prediction must use all valid groups feature-only")
        if "--fit-subset-manifest" in predict:
            raise ValueError("fixed evaluation prediction must not reuse fit-subset membership")
        if f"{name}@0" not in score:
            raise ValueError("score command does not bind the fixed epoch-zero candidate")
    if eval_names != run_names:
        raise ValueError("every fit requires one distinct fresh prediction and score command")
    return {
        "schema": recipe["schema"],
        "fit_count": len(fits),
        "profile_count": len(profiles),
        "evaluation_count": len(evaluations),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--delivery-dir", type=Path, default=DELIVERY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    delivery = args.delivery_dir.resolve()
    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = load_campaign_split(split_path, root)
    if split.sha256 != SPLIT_SHA256 or split.fold != 0:
        raise ValueError("source-qualified fold-0 manifest identity changed")
    if len(split.train_groups) != 10_535 or len(split.valid_groups) != 1_405:
        raise ValueError("source-qualified fold-0 fitting membership counts changed")
    if len(split.evaluation_groups) != 468:
        raise ValueError("source-qualified fold-0 evaluation membership count changed")

    chain = verify_parent_chain(root, split)
    cache = verify_teacher_cache(root, split)
    subsets = verify_subsets(delivery, split)
    recipe = verify_run_recipe(delivery)
    result = {
        "schema": "wnn.qat-delivery-input-verification.v1",
        "status": "pass",
        "scope": "metadata and membership only; no training or evaluation",
        "split": {
            "path": str(split_path.relative_to(root)),
            "sha256": split.sha256,
            "fold": split.fold,
            "train_groups": len(split.train_groups),
            "fit_valid_groups": len(split.valid_groups),
            "evaluation_groups": len(split.evaluation_groups),
        },
        "parent_chain": chain,
        "teacher_cache": cache,
        "subsets": subsets,
        "recipe": recipe,
    }
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
        print(json.dumps({"status": "pass", "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
