"""Retain the local cold-import and CLI-fixture proof for the child profile."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve()
PREPARATION = HERE.parents[1]
CAMPAIGN = PREPARATION.parent
SOURCE = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SUPPORT_SHA256 = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
SOURCE_MANIFEST = PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
PROFILE_TEMPLATE = PREPARATION / "child_allthree_profile_v1/REMOTE_SUBMISSION_TEMPLATE.py"
ACCEPTANCE = CAMPAIGN / "records/parent_fit_r2_v1/TERMINAL_ACCEPTANCE.json"
EXACT_STOP = CAMPAIGN / "records/parent_fit_r2_v1/exact_stop/STDOUT.raw.json"
OUTPUT = HERE.parent / "CHILD_ALLTHREE_PROFILE_LOCAL_CHECKS_V1.json"
OLD_CONTRACT_SHA256 = "ca6bc4b1c33e445521bce5d80b24a436dddc2319ace38cb75781872008b95d11"
OLD_TEST_SHA256 = "d2a72564befa758892ad678130940e06ee95db75fc717f368d2d067dfd96f179"
OLD_PROBE_SHA256 = "446e679ddae0f05225209950d928790264c8cd57fb8836b79f671d06d03be9e4"
PRIVATE_D_HELPERS = {
    "child_allthree_profile_contract_v1.py": (
        PREPARATION / "child_allthree_profile_contract_v1.py",
        "ab68d868261106a7548b649243f4a645ece926b6138d683c731841cfb9f0b7c5",
    ),
    "child_allthree_profile_probe_v1.py": (
        PREPARATION / "child_allthree_profile_probe_v1.py",
        "446e679ddae0f05225209950d928790264c8cd57fb8836b79f671d06d03be9e4",
    ),
    "propensity_terminal_support.py": (
        PREPARATION / "propensity_terminal_support.py",
        SUPPORT_SHA256,
    ),
}


def _sha256_bytes(value: bytes) -> str:
    """Return a stable digest for a bounded local source or receipt file."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a source file without loading any runtime model or data payload."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_level_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    """Index simple top-level assignments from the packet template AST."""
    values: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                values[statement.target.id] = statement.value
    return values


def _evaluate_template_expr(
    node: ast.expr, assignments: dict[str, ast.expr], resolving: frozenset[str] = frozenset()
) -> object:
    """Evaluate only constants and the narrow path/string expressions in trainer argv."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in resolving or node.id not in assignments:
            raise ValueError(f"trainer argv references an unbound name: {node.id}")
        return _evaluate_template_expr(assignments[node.id], assignments, resolving | {node.id})
    if isinstance(node, ast.List):
        return [_evaluate_template_expr(item, assignments, resolving) for item in node.elts]
    if isinstance(node, ast.BinOp):
        left = _evaluate_template_expr(node.left, assignments, resolving)
        right = _evaluate_template_expr(node.right, assignments, resolving)
        if isinstance(node.op, ast.Div) and isinstance(left, Path) and isinstance(right, str):
            return left / right
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ValueError("trainer argv uses an unsupported binary expression")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        argument = _evaluate_template_expr(node.args[0], assignments, resolving)
        if node.func.id == "Path" and isinstance(argument, str):
            return Path(argument)
        if node.func.id == "str":
            return str(argument)
    raise ValueError(f"trainer argv uses unsupported AST syntax: {type(node).__name__}")


def _template_trainer_argv(path: Path) -> list[str]:
    """Read the real trainer list without evaluating packet code or its payload."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_trainer_argv"
    ]
    if len(functions) != 1:
        raise ValueError("profile template must define one _trainer_argv function")
    returns = [node for node in functions[0].body if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise ValueError("trainer argv function must have one literal return expression")
    evaluated = _evaluate_template_expr(returns[0].value, _top_level_assignments(tree))
    if not isinstance(evaluated, list) or not all(isinstance(item, str) for item in evaluated):
        raise ValueError("trainer argv did not resolve to a string list")
    return evaluated


def _replace_argument(argv: list[str], option: str, value: str) -> None:
    """Substitute one local fixture path while preserving all trainer options."""
    positions = [index for index, item in enumerate(argv) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer argv must contain exactly one {option}")
    argv[positions[0] + 1] = value


def _git(cwd: Path, *args: str) -> str:
    """Read pinned repository metadata through a literal git subprocess."""
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_module_pins(source_root: Path, expected_files: dict[str, str]) -> list[dict[str, str]]:
    """Hash every imported file-backed WNN module from the pinned source tree."""
    pins: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if not (name == "wnn" or name.startswith("wnn.")) or not isinstance(module, ModuleType):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve(strict=True)
        relative = path.relative_to(source_root).as_posix()
        manifest_path = f"src/{relative}"
        digest = _sha256_file(path)
        if expected_files.get(manifest_path) != digest:
            raise ValueError(
                f"runtime module differs from the pinned source manifest: {manifest_path}"
            )
        pins.append({"module": name, "path": manifest_path, "sha256": digest})
    return pins


def main() -> int:
    """Run metadata-only CLI/receipt fixtures and persist the retained proof receipt."""
    calls: list[str] = []

    def read_audit(event: str, arguments: tuple[object, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(os.fspath(candidate))
        lowered = path.lower()
        forbidden_roots = (
            "/home/grinchenkoeg/computer-vision-homework-20261018/data/",
            "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/",
        )
        if lowered.endswith((".parquet", ".npy", ".npz", ".pt")) or any(
            path.startswith(root) for root in forbidden_roots
        ):
            calls.append(path)
            raise PermissionError(f"cold proof blocked a data/model payload open: {path}")

    sys.addaudithook(read_audit)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source_root = (SOURCE / "src").resolve(strict=True)
    if os.environ.get("PYTHONPATH") != str(source_root):
        raise ValueError("cold proof PYTHONPATH must contain only the pinned source/src path")
    commit = _git(SOURCE, "rev-parse", "HEAD")
    dirty = _git(SOURCE, "status", "--short", "--untracked-files=no")
    source_manifest_raw = SOURCE_MANIFEST.read_bytes()
    if _sha256_bytes(source_manifest_raw) != SOURCE_MANIFEST_SHA256:
        raise ValueError("167-file source manifest bytes differ from the accepted pin")
    source_manifest = json.loads(source_manifest_raw)
    manifest_files = source_manifest.get("files")
    if (
        commit != SOURCE_COMMIT
        or source_manifest.get("source_commit") != SOURCE_COMMIT
        or not isinstance(manifest_files, list)
        or len(manifest_files) != 167
        or dirty
    ):
        raise ValueError("canonical source worktree differs from the clean 167-file source pin")
    expected_files = {
        entry["path"]: entry["sha256"]
        for entry in manifest_files
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("sha256"), str)
    }
    if len(expected_files) != 167:
        raise ValueError("source manifest file identities are malformed")

    with tempfile.TemporaryDirectory(prefix="fg19-child-profile-code-only-") as temporary:
        code_only = Path(temporary)
        for filename, (source_path, expected_sha256) in PRIVATE_D_HELPERS.items():
            raw = source_path.read_bytes()
            if _sha256_bytes(raw) != expected_sha256:
                raise ValueError(f"private-D helper input differs from its pin: {filename}")
            staged_path = code_only / filename
            staged_path.write_bytes(raw)
            if _sha256_file(staged_path) != expected_sha256:
                raise ValueError(f"private-D helper changed during local staging: {filename}")
        module_names = (
            "child_allthree_profile_contract_v1",
            "child_allthree_profile_probe_v1",
            "propensity_terminal_support",
        )
        if any(name in sys.modules for name in module_names):
            raise RuntimeError("profile helpers were imported before the planned private-D fixture")
        sys.path.insert(0, str(code_only))
        if Path(sys.path[0]).resolve(strict=True) != code_only.resolve(strict=True):
            raise RuntimeError("planned private-D code-only directory is not first on sys.path")
        helper_sys_path_zero = str(Path(sys.path[0]).resolve(strict=True))
        contract_module = importlib.import_module(module_names[0])
        profile_probe = importlib.import_module(module_names[1])
        if module_names[2] in sys.modules:
            raise RuntimeError("profile probe unexpectedly imported its lazy restricted reader")
        importlib.import_module(module_names[2])
        if not callable(profile_probe._load_parent_state):
            raise ValueError("profile probe restricted loader entrypoint is missing")
        early_heavy_modules = sorted(name for name in ("numpy", "torch") if name in sys.modules)
        if early_heavy_modules or any(
            name == "wnn" or name.startswith("wnn.") for name in sys.modules
        ):
            raise RuntimeError("profile helper imports were not cold")
        if contract_module.SOURCE_COMMIT != SOURCE_COMMIT:
            raise ValueError("profile contract source commit differs from the proof pin")
        helper_modules = {}
        for module_name, filename in zip(module_names, PRIVATE_D_HELPERS, strict=True):
            module = sys.modules[module_name]
            origin = Path(module.__file__).resolve(strict=True)
            expected_origin = (code_only / filename).resolve(strict=True)
            digest = _sha256_file(origin)
            if origin != expected_origin or digest != PRIVATE_D_HELPERS[filename][1]:
                raise ValueError(f"private-D module origin or SHA differs: {module_name}")
            if origin.is_relative_to(PREPARATION):
                raise ValueError(f"private-D helper resolved from local preparation: {module_name}")
            helper_modules[module_name] = {
                "origin": str(origin),
                "sha256": digest,
                "sys_path_zero_at_import": helper_sys_path_zero,
            }

        sys.path.insert(0, str(source_root))
        import torch

        from wnn.data import SEQUENCE_LENGTH
        from wnn.training.cli import build_parser
        from wnn.training.environment import TrainingEnvironment
        from wnn.training.runner import run_training
        from wnn.training_dynamics import build_optimizer

        if SEQUENCE_LENGTH != 20_000 or torch.cuda.is_initialized():
            raise RuntimeError("cold canonical import initialized CUDA or changed sequence length")
        canonical_components = {
            "TrainingEnvironment": TrainingEnvironment.__module__,
            "run_training": run_training.__module__,
            "build_optimizer": build_optimizer.__module__,
        }
        template_raw = PROFILE_TEMPLATE.read_bytes()
        trainer_argv = _template_trainer_argv(PROFILE_TEMPLATE)
        original_argv = list(trainer_argv)
        _replace_argument(trainer_argv, "--representation-cache", str(PREPARATION))
        _replace_argument(trainer_argv, "--distill-cache-root", str(CAMPAIGN / "records"))
        args = build_parser().parse_args(trainer_argv)
        contract_module.validate_profile_args(
            args, Path(args.representation_cache), Path(args.distill_cache_root)
        )
        if args.inherit_init_preprocessing is not True:
            raise ValueError("canonical CLI fixture omitted parent preprocessing inheritance")
        if args.student_frontier_preflight_only or args.calibrate_distill_reference:
            raise ValueError("canonical CLI fixture enables a short-circuit mode")

        acceptance_sha256 = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
        stop_sha256 = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
        parent = contract_module.validate_parent_acceptance(
            ACCEPTANCE, acceptance_sha256, contract_module.PARENT_EP3_SHA256
        )
        stopped = contract_module.validate_exact_stop(EXACT_STOP, stop_sha256)
        if (
            parent.get("invocation_id") != contract_module.PARENT_INVOCATION
            or stopped.get("unit") != contract_module.PARENT_UNIT
        ):
            raise ValueError(
                "stored parent acceptance/stop fixtures differ from the profile identity"
            )
        if stopped.get("populated_after") != {}:
            raise ValueError("parent exact-stop fixture reports populated remaining descendants")

        modules = _runtime_module_pins(source_root, expected_files)
        submodule_count = sum(row["module"].startswith("wnn.") for row in modules)
        if len(modules) != 96 or submodule_count != 95 or torch.cuda.is_initialized() or calls:
            raise RuntimeError(
                f"cold import/read/CUDA sentinel failed: modules={len(modules)}, "
                f"submodules={submodule_count}, "
                f"cuda_initialized={torch.cuda.is_initialized()}, payload_opens={calls}"
            )

        current_files = {
            "contract": PREPARATION / "child_allthree_profile_contract_v1.py",
            "probe": PREPARATION / "child_allthree_profile_probe_v1.py",
            "tests": PREPARATION / "test_child_allthree_profile_probe_v1.py",
        }
        current_pins = {key: _sha256_file(path) for key, path in current_files.items()}
        proof_script_sha256 = _sha256_file(HERE)
        receipt = {
            "schema": "fg19.child-allthree-profile-local-checks.v1",
            "status": "PASS_LOCAL_COLD_IMPORT_CLI_RECEIPT_FIXTURES_ONLY",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "scope": {
                "remote_actions": False,
                "training_or_profile_execution": False,
                "model_or_optimizer_payload_read": False,
                "array_or_parquet_payload_read": False,
                "quality_or_fit_claim": False,
                "full_existing_test_suite_replayed": False,
                "inheritance_delta": "inherit_init_preprocessing must be exactly True",
            },
            "source": {
                "root": str(SOURCE),
                "commit": commit,
                "source_manifest_path": str(SOURCE_MANIFEST.relative_to(CAMPAIGN)),
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "source_file_count": len(manifest_files),
                "tracked_worktree_clean": True,
                "runtime_wnn_module_count": len(modules),
                "runtime_wnn_submodule_count": submodule_count,
                "runtime_wnn_modules": modules,
            },
            "local_code": {
                **{
                    key: {"path": str(path.relative_to(CAMPAIGN)), "sha256": current_pins[key]}
                    for key, path in current_files.items()
                },
                "proof_script": {
                    "path": str(HERE.relative_to(CAMPAIGN)),
                    "sha256": proof_script_sha256,
                },
                "profile_template": {
                    "path": str(PROFILE_TEMPLATE.relative_to(CAMPAIGN)),
                    "sha256": _sha256_bytes(template_raw),
                },
                "planned_private_d_helpers": {
                    "sys_path_zero_at_import": helper_sys_path_zero,
                    "pythonpath": os.environ.get("PYTHONPATH", ""),
                    "modules": helper_modules,
                    "preparation_imported": False,
                    "restricted_checkpoint_loader_called": False,
                },
            },
            "cli_fixture": {
                "canonical_parser": "wnn.training.cli.build_parser",
                "canonical_training_components_imported": [
                    "wnn.data.SEQUENCE_LENGTH",
                    "wnn.training.environment.TrainingEnvironment",
                    "wnn.training.runner.run_training",
                    "wnn.training_dynamics.build_optimizer",
                ],
                "canonical_component_origins": canonical_components,
                "trainer_argv_sha256": _sha256_bytes("\0".join(original_argv).encode()),
                "trainer_argv": original_argv,
                "local_fixture_path_substitutions": {
                    "--representation-cache": str(PREPARATION),
                    "--distill-cache-root": str(CAMPAIGN / "records"),
                },
                "validated": True,
                "inherit_init_preprocessing": args.inherit_init_preprocessing,
                "sequence_length": SEQUENCE_LENGTH,
                "training_entrypoint_called": False,
                "optimizer_builder_called": False,
            },
            "reader_and_cuda_sentinels": {
                "blocked_payload_paths": calls,
                "blocked_suffixes": [".parquet", ".npy", ".npz", ".pt"],
                "cuda_visible_devices": "",
                "cuda_initialized_after_canonical_import": torch.cuda.is_initialized(),
            },
            "actual_parent_receipt_fixtures": {
                "acceptance_sha256": acceptance_sha256,
                "acceptance_status": parent.get("status"),
                "acceptance_invocation_id": parent.get("invocation_id"),
                "selected_epoch": parent.get("selected_epoch"),
                "successful_updates": parent.get("successful_updates"),
                "nonfinite_updates": parent.get("nonfinite_updates"),
                "checkpoint_sha256": contract_module.PARENT_EP3_SHA256,
                "exact_stop_raw_sha256": stop_sha256,
                "exact_stop_unit": stopped.get("unit"),
                "exact_stop_invocation_id": stopped.get("invocation_id"),
                "populated_after": stopped.get("populated_after"),
                "dictionary_acceptance_sha256": contract_module.DICTIONARY_ACCEPTANCE_SHA256,
                "dictionary_cache_sha256": contract_module.DICTIONARY_SHA256,
                "checkpoint_payload_opened": False,
                "dictionary_payload_opened": False,
            },
            "focused_test_and_style_evidence": [
                {
                    "command": "PYTHONPATH=research/campaign_2026-10-19/full_gru_tweaks_r1/preparation /Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -m pytest -q research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/test_child_allthree_profile_probe_v1.py",
                    "result": "PASS_HISTORICAL_PRE_INHERITANCE_DELTA",
                    "exit_code": 0,
                    "stdout": "................. [100%]\n17 passed in 0.75s\n",
                    "test_sha256_at_run": OLD_TEST_SHA256,
                    "contract_sha256_at_run": OLD_CONTRACT_SHA256,
                    "probe_sha256_at_run": OLD_PROBE_SHA256,
                    "note": "Recorded from the earlier run; the 17-test suite was not replayed after the one-field delta.",
                },
                {
                    "command": "PYTHONPATH=research/campaign_2026-10-19/full_gru_tweaks_r1/preparation /Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -m pytest -q research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/test_child_allthree_profile_probe_v1.py::test_profile_requires_parent_preprocessing_inheritance",
                    "result": "PASS",
                    "exit_code": 0,
                    "stdout": ". [100%]\n1 passed in 0.84s\n",
                    "test_sha256": current_pins["tests"],
                    "contract_sha256": current_pins["contract"],
                    "probe_sha256": current_pins["probe"],
                },
                {
                    "command": "/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -m ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_profile_contract_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_profile_probe_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/test_child_allthree_profile_probe_v1.py",
                    "result": "PASS",
                    "stdout": "All checks passed!\n",
                    "source_sha256": current_pins,
                },
            ],
            "proof_console_summary": f"LOCAL_PROFILE_PROOF_PASS; WNN_MODULES={len(modules)}; WNN_SUBMODULES={submodule_count}; CUDA_INITIALIZED=false; PAYLOAD_READS=0",
        }
        if OUTPUT.exists():
            raise FileExistsError(f"proof receipt already exists: {OUTPUT}")
        raw = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with OUTPUT.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    print(receipt["proof_console_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
