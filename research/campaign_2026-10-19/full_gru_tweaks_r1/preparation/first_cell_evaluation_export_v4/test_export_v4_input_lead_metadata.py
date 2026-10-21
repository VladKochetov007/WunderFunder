"""Focused metadata and source-contract checks for the V4 input-lead default."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[4]
RECORDS = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/records"
BINDING = PACKET / "EXPORT_BINDING_V4.json"
SOURCE_MANIFEST = PACKET / "EXPORT_SOURCE_MANIFEST.json"
EXPORTER = PACKET / "export_first_cell_v4.py"
CONTROL_ADMISSION = RECORDS / "child_control_fit_v1/ADMISSION.json"
CONTROL_LAUNCH = RECORDS / "child_control_fit_v1/LAUNCH.json"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
EXPECTED_CONTROL_ADMISSION = (
    9_059,
    "f749d0c4ba4925506320f1a480d7e76164a2b0f84b7efac5b432cb18522c5973",
)
EXPECTED_CONTROL_LAUNCH = (
    6_999,
    "afd19428c8b6034a9dd40fac78d6adb78b70f3d95f1f964917cbadc828ed27e9",
)
TRAINING_ARGUMENT_SOURCES = (
    "scripts/train_rnn.py",
    "src/wnn/training/checkpoints.py",
    "src/wnn/training/cli.py",
    "src/wnn/training/cli_data.py",
    "src/wnn/training/cli_features.py",
    "src/wnn/training/cli_model.py",
    "src/wnn/training/cli_normalization.py",
    "src/wnn/training/cli_optimizer.py",
    "src/wnn/training/cli_performance.py",
    "src/wnn/training/cli_targets.py",
    "src/wnn/training/cli_views.py",
)
LINEAGE_SOURCE = "src/wnn/training/setup_lineage.py"


def _pinned_commit_source(path: str, pins: dict[str, dict[str, object]]) -> str:
    """Read a source file from the exact source commit and verify its manifest row."""
    completed = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        check=True,
        capture_output=True,
    )
    raw = completed.stdout
    pin = pins[path]
    if (len(raw), hashlib.sha256(raw).hexdigest()) != (pin["bytes"], pin["sha256"]):
        raise AssertionError(f"canonical source differs from the pinned manifest: {path}")
    return raw.decode("utf-8")


def _validator() -> object:
    """Compile only the pure V4 argument-validation helper from its source AST."""
    tree = ast.parse(EXPORTER.read_text(encoding="utf-8"), filename=str(EXPORTER))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_checkpoint_arguments"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(isolated), str(EXPORTER), "exec"), namespace)
    return namespace["_validate_checkpoint_arguments"]


def _checkpoint_metadata() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    binding = json.loads(BINDING.read_bytes())
    arm = next(item for item in binding["arms"] if item["arm"] == "control")
    expected = binding["expected_checkpoint"]
    args = {
        "arch": expected["arch"],
        "hidden": expected["hidden"],
        "layers": expected["layers"],
        "mean_variance_head": expected["mean_variance_head"],
        "precision": expected["precision"],
        "split_manifest_sha256": binding["split"]["sha256"],
        "representation_mode": arm["representation_mode"],
        "online_norm": expected["online_norm"],
        "heuristics": expected["heuristics"],
        "raw_only": expected["raw_only"],
        "research_features": expected["research_features"],
        "research_feature_control": expected["research_feature_control"],
        "privileged_mode": expected["privileged_mode"],
        "prop_input": expected["prop_input"],
    }
    return binding, arm, args


class ExportV4InputLeadMetadataTests(unittest.TestCase):
    def test_actual_control_receipts_and_canonical_cli_omit_input_lead(self) -> None:
        admission_raw = CONTROL_ADMISSION.read_bytes()
        launch_raw = CONTROL_LAUNCH.read_bytes()
        self.assertEqual(
            (len(admission_raw), hashlib.sha256(admission_raw).hexdigest()),
            EXPECTED_CONTROL_ADMISSION,
        )
        self.assertEqual(
            (len(launch_raw), hashlib.sha256(launch_raw).hexdigest()),
            EXPECTED_CONTROL_LAUNCH,
        )
        admission = json.loads(admission_raw)
        launch = json.loads(launch_raw)
        self.assertEqual(
            (admission["schema"], admission["run_name"], admission["unit"]),
            (
                "fg19-child-control-fit-admission.v1",
                "fg19_child_control_f0_s1001_fit_v1",
                "computer-vision-homework-fg19-child-control-fit-v1.service",
            ),
        )
        command = launch["command"]
        trainer_paths = [
            index for index, value in enumerate(command) if value.endswith("/scripts/train_rnn.py")
        ]
        self.assertEqual(len(trainer_paths), 1)
        trainer_argv = command[trainer_paths[0] + 1 :]
        trainer_argv_sha = hashlib.sha256(
            json.dumps(trainer_argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(trainer_argv_sha, admission["recipe"]["trainer_argv_sha256"])
        self.assertFalse(
            any(
                argument == "--input-lead"
                or argument == "--input_lead"
                or argument.startswith("--input-lead=")
                or argument.startswith("--input_lead=")
                for argument in trainer_argv
            )
        )

        manifest = json.loads(SOURCE_MANIFEST.read_bytes())
        self.assertEqual(manifest["source_commit"], SOURCE_COMMIT)
        pins = {item["path"]: item for item in manifest["files"]}
        serving = _pinned_commit_source("scripts/predict_run.py", pins)
        self.assertIn('train_rnn.INPUT_LEAD = int(ck["args"].get("input_lead", 0))', serving)
        checkpoint_source = _pinned_commit_source("src/wnn/training/checkpoints.py", pins)
        self.assertIn("vars(args).items()", checkpoint_source)
        cli_sources: dict[str, str] = {}
        for path in TRAINING_ARGUMENT_SOURCES:
            cli_sources[path] = _pinned_commit_source(path, pins)
            self.assertNotIn("input_lead", cli_sources[path])
        lineage = _pinned_commit_source(LINEAGE_SOURCE, pins)
        self.assertIn("args.split_manifest_sha256 = campaign_split.sha256", lineage)

        binding, _, _ = _checkpoint_metadata()
        expected = binding["expected_checkpoint"]

        def option(flag: str) -> str:
            index = trainer_argv.index(flag)
            return trainer_argv[index + 1]

        explicit_values = {
            "arch": option("--arch"),
            "hidden": int(option("--hidden")),
            "layers": int(option("--layers")),
            "mean_variance_head": int(option("--mean-variance-head")),
            "precision": option("--precision"),
            "dropout": float(option("--dropout")),
        }
        for key, value in explicit_values.items():
            self.assertEqual(expected[key], value, key)
        self.assertEqual(option("--cell"), "gru")
        self.assertEqual(option("--enc"), "linear")
        self.assertEqual(
            next(item for item in binding["arms"] if item["arm"] == "control")[
                "representation_mode"
            ],
            option("--representation-mode"),
        )
        model_cli = cli_sources["src/wnn/training/cli_model.py"]
        self.assertIn(
            'ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")', model_cli
        )
        self.assertIn('default="linear",', model_cli)
        self.assertIn('ap.add_argument("--dropout", type=float, default=0.0)', model_cli)
        features = cli_sources["src/wnn/training/cli_features.py"]
        self.assertIn('"--prop-input",\n        default="",', features)
        self.assertNotIn("--prop-input", trainer_argv)
        self.assertEqual(expected["prop_input"], "")
        self.assertIn("if input_propensity\n        else None", serving)
        self.assertIn('ap.add_argument("--online-norm", action="store_true")', features)
        self.assertIn('"--heuristics", action="store_true"', features)
        self.assertIn('"--raw-only",\n        action="store_true",', features)
        self.assertIn(
            '"--research-features",\n        choices=RESEARCH_FEATURE_MODES,\n        default="none",',
            features,
        )
        self.assertIn(
            '"--research-feature-control",\n        choices=("none", "newest"),\n        default="none",',
            features,
        )
        self.assertIn(
            '"--privileged-mode",\n        choices=PRIVILEGED_MODES,\n        default="none",',
            features,
        )
        for flag in ("--online-norm", "--heuristics", "--raw-only"):
            self.assertNotIn(flag, trainer_argv)
        self.assertEqual(expected["online_norm"], False)
        self.assertEqual(expected["heuristics"], False)
        self.assertEqual(expected["raw_only"], False)
        self.assertEqual(expected["research_features"], "none")
        self.assertEqual(expected["research_feature_control"], "none")
        self.assertEqual(expected["privileged_mode"], "none")

    def test_absent_or_exact_zero_matches_the_canonical_default(self) -> None:
        validator = _validator()
        binding, arm, args = _checkpoint_metadata()
        validator(args, arm, binding)
        validator(args | {"input_lead": 0}, arm, binding)

    def test_noncanonical_input_lead_values_are_rejected(self) -> None:
        validator = _validator()
        binding, arm, args = _checkpoint_metadata()
        for value in (None, False, True, "0", 1, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "input_lead"):
                validator(args | {"input_lead": value}, arm, binding)

    def test_binding_keeps_the_fixed_zero_input_lead_recipe(self) -> None:
        binding = json.loads(BINDING.read_bytes())
        self.assertIs(type(binding["expected_checkpoint"]["input_lead"]), int)
        self.assertEqual(binding["expected_checkpoint"]["input_lead"], 0)
        self.assertEqual(binding["expected_checkpoint"]["prop_input"], "")
        self.assertTrue(binding["evidence_root"].endswith("evaluation_v3"))
        self.assertEqual(binding["output_dir"], f"{binding['evidence_root']}/exports")

    def test_all_required_arguments_and_optional_defaults_are_enforced(self) -> None:
        validator = _validator()
        binding, arm, args = _checkpoint_metadata()
        validator(args, arm, binding)
        for key, value in args.items():
            invalid = dict(args)
            invalid[key] = None if value is not None else "unexpected"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validator(invalid, arm, binding)

        validator(args | {"cell": "gru", "enc": "linear", "dropout": 0.0}, arm, binding)
        for key, value in (("cell", "lstm"), ("enc", "mlp"), ("dropout", 0.1)):
            with self.subTest(default=key), self.assertRaisesRegex(ValueError, key):
                validator(args | {key: value}, arm, binding)


if __name__ == "__main__":
    unittest.main()
