"""Focused local metadata and invariant tests for the EMA terminal checker."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PREPARATION = Path(__file__).resolve().parents[1]
CAMPAIGN = PREPARATION.parent
PACKET = PREPARATION / "child_ema_terminal_check_v1"
RECORDS = CAMPAIGN / "records"
SOURCE_ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
SOURCE_MANIFEST = PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
sys.path.insert(0, str(PACKET))
sys.path.insert(0, str(PREPARATION))
sys.path.insert(0, str(PREPARATION / "child_control_terminal_verifier_v3"))

checker = importlib.import_module("child_ema_terminal_check_v1")


def _read_json(path: Path) -> tuple[dict[str, object], bytes]:
    """Read one small local JSON receipt fixture."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON mapping: {path.name}")
    return value, raw


class EmaTerminalIdentityTests(unittest.TestCase):
    """Bind actual small receipts and reject nearby identity substitutions."""

    def setUp(self) -> None:
        self.ema_records = RECORDS / "child_ema_fit_v1"
        self.admission, self.admission_raw = _read_json(self.ema_records / "ADMISSION.json")
        self.launch, self.launch_raw = _read_json(self.ema_records / "LAUNCH.json")
        self.acceptance, self.acceptance_raw = _read_json(
            self.ema_records / "LAUNCH_ACCEPTANCE.json"
        )

    def test_actual_admission_launch_and_root_acceptance_bind(self) -> None:
        """The original EMA receipts have exact pins and nested body binding."""
        checker._check_launch_records(
            self.admission,
            hashlib.sha256(self.admission_raw).hexdigest(),
            self.launch,
            hashlib.sha256(self.launch_raw).hexdigest(),
            self.acceptance,
            hashlib.sha256(self.acceptance_raw).hexdigest(),
        )
        checker._check_admission(self.admission)
        self.assertEqual(checker._trainer_argv_sha(self.launch), checker.TRAINER_ARGV_SHA)

    def test_launch_acceptance_rejects_body_and_predecessor_drift(self) -> None:
        """The root body pin and accepted CONTROL predecessor cannot be substituted."""
        bad_acceptance = {
            **self.acceptance,
            "body": {**self.acceptance["body"], "sha256": "0" * 64},
        }
        with self.assertRaisesRegex(ValueError, "root actual EMA launch acceptance"):
            checker._check_launch_records(
                self.admission,
                checker.ADMISSION_SHA,
                self.launch,
                checker.LAUNCH_SHA,
                bad_acceptance,
                checker.LAUNCH_ACCEPTANCE_SHA,
            )
        bad_admission = {**self.admission, "control_predecessor": {}}
        with self.assertRaisesRegex(ValueError, "EMA admission predecessor"):
            checker._check_launch_records(
                bad_admission,
                checker.ADMISSION_SHA,
                self.launch,
                checker.LAUNCH_SHA,
                self.acceptance,
                checker.LAUNCH_ACCEPTANCE_SHA,
            )

    def test_actual_control_acceptance_and_exact_stop_bind_metadata_only(self) -> None:
        """The CONTROL predecessor is checked from its accepted metadata receipts only."""
        control_root = RECORDS / "child_control_fit_v1"
        with tempfile.TemporaryDirectory(prefix="ema-control-receipts-") as temporary:
            root = Path(temporary)
            (root / "CONTROL_TERMINAL_ACCEPTANCE.json").write_bytes(
                (control_root / "TERMINAL_ACCEPTANCE.json").read_bytes()
            )
            (root / "CONTROL_EXACT_STOP.json").write_bytes(
                (control_root / "exact_stop/STDOUT.raw.json").read_bytes()
            )
            result = checker._check_control_predecessor(root, self.admission)
            with self.assertRaisesRegex(ValueError, "EMA staged CONTROL predecessor binding"):
                checker._check_control_predecessor(
                    root, {**self.admission, "control_predecessor": {}}
                )
        self.assertEqual(result["acceptance_sha256"], checker.CONTROL_ACCEPTANCE_SHA)
        self.assertEqual(result["exact_stop_sha256"], checker.CONTROL_STOP_SHA)
        self.assertEqual(result["selected_epoch"], 5)

    def test_ema_native_terminal_requires_exact_typed_exit(self) -> None:
        """EMA exit evidence is invocation-specific and rejects non-integer status fields."""
        terminal: dict[str, object] = {
            "schema": "fg19.child-ema-fit-native-terminal-evidence.v1",
            "unit": checker.UNIT,
            "run_name": checker.RUN,
            "invocation_id": checker.INVOCATION,
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
            "raw_bytes": 128,
            "raw_sha256": "a" * 64,
            "raw_path": "capture/STDOUT.raw.json",
            "captured_utc": "2026-10-20T14:00:00+00:00",
            "native_exit_utc": "2026-10-20T13:58:00Z",
        }
        checker._check_native_terminal(terminal)
        for change in (
            {"main_pid": True},
            {"main_pid": "0"},
            {"exec_main_code": 0},
            {"invocation_id": "0" * 32},
            {"sub_state": "dead"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checker._check_native_terminal({**terminal, **change})

    def test_registered_data_stat_gate_allows_only_recorded_alias(self) -> None:
        """Stat-only checks permit an exact admitted symlink target and reject path drift."""
        with tempfile.TemporaryDirectory(prefix="ema-stat-fixture-") as temporary:
            root = Path(temporary)
            data_root = root / "data"
            alias = root / "approved-aliases/chain_folds.npz"
            alias.parent.mkdir(parents=True)
            alias.write_bytes(b"synthetic-stat-fixture")
            records: dict[str, object] = {}
            for index, relative in enumerate(checker.REGISTERED_DATA_NAMES):
                path = data_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if index == 0:
                    path.symlink_to(alias)
                else:
                    path.write_bytes(f"synthetic-{index}".encode())
                stat_info = path.stat()
                records[relative] = {
                    "bytes": stat_info.st_size,
                    "device": stat_info.st_dev,
                    "inode": stat_info.st_ino,
                    "islink": path.is_symlink(),
                    "mtime_ns": stat_info.st_mtime_ns,
                    "path": str(path),
                    "registered_sha256": "b" * 64,
                    "resolved": str(path.resolve(strict=True)),
                }
            observed = checker._check_registered_data_stats(
                {"data_identity_stats_only": records}, data_root
            )
            self.assertTrue(observed[checker.REGISTERED_DATA_NAMES[0]]["islink"])
            altered = json.loads(json.dumps(records))
            altered[checker.REGISTERED_DATA_NAMES[0]]["resolved"] = str(root / "unapproved.npz")
            with self.assertRaisesRegex(ValueError, "resolved path"):
                checker._check_registered_data_stats(
                    {"data_identity_stats_only": altered}, data_root
                )

    def test_recipe_requires_ema_mode_and_pinned_optimizer(self) -> None:
        """The complete argv digest and semantic fields retain the frozen EMA recipe."""
        args: dict[str, object] = {
            "name": checker.RUN,
            "device": "cuda",
            "seed": 1001,
            "train_source": "train+valid",
            "valid_fold": 0,
            "train_seqs": 10_535,
            "valid_repeat": 1,
            "valid_train_fraction": 1.0,
            "batch_order": "seeded",
            "batch": 16,
            "chunk": 2_000,
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 5,
            "arch": "gru",
            "cell": "gru",
            "enc": "linear",
            "hidden": 256,
            "layers": 2,
            "head_layers": 1,
            "dropout": 0.0,
            "init": f"{checker.PARENT_RUN}@ep3",
            "inherit_init_preprocessing": True,
            "representation_mode": "ema",
            "representation_cache": str(checker.DICTIONARY_PATH),
            "split_manifest": "/remote/splits/fold0.json",
            "mean_variance_head": 1,
            "mean_variance_objective": "wmse",
            "loss": "wmse",
            "loss_norm": "chunk",
            "distill_cache_root": "/remote/research_tmp/fg19_t3_fit_homogeneous_cache_v1",
            "distill_cache_variant": "homogeneous",
            "distill_cache_scale": 1.0,
            "distill_alpha": 0.9,
            "optim": "adamw",
            "lr": 1e-4,
            "wd": 0.01,
            "beta2": 0.999,
            "sched": "onecycle",
            "warmup_frac": 0.05,
            "clip": 1.0,
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
        }
        split = Path("/remote/splits/fold0.json")
        cache = Path(str(args["distill_cache_root"]))
        checker._check_recipe_args(args, split, cache)
        with self.assertRaisesRegex(ValueError, "EMA launched scientific recipe"):
            checker._check_recipe_args({**args, "representation_mode": "base"}, split, cache)
        with self.assertRaisesRegex(ValueError, "EMA launched scientific recipe"):
            checker._check_recipe_args({**args, "optim": "sgd"}, split, cache)

    def test_ema_auxiliary_provenance_keeps_the_shared_dictionary(self) -> None:
        """Only input representation mode is EMA; the dictionary tensors remain identical."""
        representation = {
            "schema": "wnn.fitted_input_representation.v1",
            "raw_mean": torch.tensor([0.0]),
            "raw_scale": torch.tensor([1.0]),
            "encoder_weight": torch.tensor([[1.0]]),
            "decoder_weight": torch.tensor([[1.0]]),
            "fit_metadata": {"fit_members_sha256": "c" * 64},
            "state_sha256": "d" * 64,
        }
        identity = {
            "mode": "ema",
            "state_sha256": "d" * 64,
            "artifact_path": str(checker.DICTIONARY_PATH),
            "artifact_file_sha256": checker.control_v3.DICTIONARY_SHA,
            "artifact_binding": "shared_fit_cache",
            "fit_metadata": {"fit_members_sha256": "c" * 64},
        }
        auxiliary = {
            "norm_ema": None,
            "dynamics": {
                "switch": None,
                "caps": {},
                "eta": 0.0,
                "clip": 1.0,
                "no_clip": False,
                "seed": 1001 + 104729,
                "is_switched": False,
                "events": [],
                "generators": {},
            },
            "pool_memory": None,
            "readout_moments": None,
            "input_representation": representation,
        }
        result = checker._check_ema_auxiliary(
            auxiliary,
            {"input_representation": identity},
            {"representation": representation},
            checker.DICTIONARY_PATH,
        )
        self.assertEqual(result["mode"], "ema")
        with self.assertRaisesRegex(ValueError, "EMA fitted-representation ancestry"):
            checker._check_ema_auxiliary(
                auxiliary,
                {"input_representation": {**identity, "mode": "base"}},
                {"representation": representation},
                checker.DICTIONARY_PATH,
            )


class ColdClosureSentinelTests(unittest.TestCase):
    """Exercise private and lazy WNN imports without payload reads or CUDA calls."""

    def test_import_closure_is_payload_read_and_cuda_silent(self) -> None:
        """A fresh interpreter verifies real WNN origins and hashes against source-68."""
        script = """
import hashlib, importlib, json, os, sys, tempfile
from pathlib import Path
import torch
def deny(*args, **kwargs): raise AssertionError("unexpected CUDA call")
for name in ("is_available", "is_initialized", "device_count", "current_device", "_lazy_init"):
    setattr(torch.cuda, name, deny)
blocked = []
def audit(event, args):
    if event == "open" and args:
        value = args[0]
        if isinstance(value, (str, bytes, os.PathLike)):
            path = os.fsdecode(value)
            if path.endswith((".pt", ".npy", ".npz", ".parquet")) or path.startswith(
                "/home/grinchenkoeg/computer-vision-homework-20261018/"
            ):
                blocked.append(path)
                raise AssertionError("payload or B source read during import")
sys.addaudithook(audit)
private_specs = (
    ("child_ema_terminal_check_v1", Path(sys.argv[1]), 36887, "a63ec6c4f6b0a3c4ba7e0cb8800e4787c5d89f1a652e0deec15ad84ffc95d30a"),
    ("child_control_terminal_check_v3", Path(sys.argv[2]), 79972, "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167"),
    ("parent_fit_terminal_check_v3", Path(sys.argv[3]), 64550, "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41"),
    ("propensity_terminal_support", Path(sys.argv[4]), 13107, "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"),
    ("teacher_s5_terminal_check_v1", Path(sys.argv[5]), 37448, "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"),
)
source = Path(sys.argv[6]).resolve(strict=True)
manifest_path = Path(sys.argv[7]).resolve(strict=True)
manifest_raw = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
if manifest_sha != "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621":
    raise AssertionError("source manifest digest changed")
manifest = json.loads(manifest_raw)
entries = manifest.get("files")
if manifest.get("source_commit") != "68ff682977b822f38351aac0f621146f5f13b440" or not isinstance(entries, list) or len(entries) != 167:
    raise AssertionError("source manifest identity changed")
source_files = {entry["path"]: entry for entry in entries}
if len(source_files) != 167:
    raise AssertionError("source manifest has duplicate paths")
with tempfile.TemporaryDirectory(prefix="ema-private-d-") as temporary:
    private_dir = Path(temporary).resolve(strict=True)
    staged_private = []
    for module_name, original_path, expected_bytes, expected_sha in private_specs:
        original = original_path.read_bytes()
        if (len(original), hashlib.sha256(original).hexdigest()) != (expected_bytes, expected_sha):
            raise AssertionError(f"private helper pin changed: {module_name}")
        staged_path = private_dir / f"{module_name}.py"
        staged_path.write_bytes(original)
        staged_private.append((module_name, staged_path, expected_bytes, expected_sha))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(private_dir))
    modules = tuple(item[0] for item in staged_private)
    if any(module_name in sys.modules for module_name in modules):
        raise AssertionError("private module was loaded before isolated-D setup")
    imported = {module_name: importlib.import_module(module_name) for module_name in modules}
    private_origins = []
    for module_name, staged_path, expected_bytes, expected_sha in staged_private:
        module = imported[module_name]
        origin = Path(module.__file__).resolve(strict=True)
        content = origin.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if origin != staged_path.resolve(strict=True) or (len(content), digest) != (expected_bytes, expected_sha):
            raise AssertionError(f"isolated private import origin or digest differs: {module_name}")
        private_origins.append({"module": module_name, "origin": origin.name, "bytes": len(content), "sha256": digest, "origin_is_private_temp_d": True})
    teacher = imported["teacher_s5_terminal_check_v1"]
    launch_args = teacher._launch_cli_args(
        {"command": [str(source / "scripts/train_rnn.py"), "--name", "fg19_child_ema_f0_s1001_fit_v1", "--seed", "1001", "--representation-mode", "ema"]},
        source,
    )
    if (launch_args.get("name"), launch_args.get("seed"), launch_args.get("representation_mode")) != ("fg19_child_ema_f0_s1001_fit_v1", 1001, "ema"):
        raise AssertionError("actual source-qualified CLI helper returned unexpected args")
    identity_fn = teacher._canonical_tensor_identity(source)
    identity = identity_fn(torch.tensor([1.25], dtype=torch.float32))
    if identity.get("shape") != [1] or identity.get("dtype") != "torch.float32" or not isinstance(identity.get("bytes_sha256"), str):
        raise AssertionError("actual source-qualified tensor identity helper returned unexpected metadata")
    lazy_modules = (
        "wnn.training.cli",
        "wnn.run_identity",
        "wnn.training.input_representation",
        "wnn.training.input_representation_fit",
        "wnn.training.input_representation_dictionary",
    )
    for module_name in lazy_modules[2:]:
        importlib.import_module(module_name)
    wnn_files = []
    for name, module in sorted(sys.modules.items()):
        if name != "wnn" and not name.startswith("wnn."):
            continue
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            raise AssertionError(f"canonical WNN module has no file origin: {name}")
        origin = Path(filename).resolve(strict=True)
        if not origin.is_relative_to(source / "src"):
            raise AssertionError(f"WNN module escaped canonical source tree: {name}")
        relative = origin.relative_to(source).as_posix()
        entry = source_files.get(relative)
        if not isinstance(entry, dict):
            raise AssertionError(f"WNN origin is not in the 167-file manifest: {relative}")
        content = origin.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if (len(content), digest) != (entry.get("bytes"), entry.get("sha256")):
            raise AssertionError(f"WNN source differs from the pinned manifest: {relative}")
        wnn_files.append({"module": name, "path": relative, "bytes": len(content), "sha256": digest})
    loaded = {item["module"] for item in wnn_files}
    if not set(lazy_modules).issubset(loaded):
        raise AssertionError("the real lazy WNN helper imports were not exercised")
    runtime = {
        "scope": "local cold-import preparation only",
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_initialized": torch.cuda._initialized,
        "remote_worker_runtime_claimed": False,
    }
    if runtime["cuda_visible_devices"] != "" or runtime["torch_cuda_initialized"] is not False:
        raise AssertionError("local CUDA isolation metadata changed")
    print(json.dumps({"modules": modules, "private_origins": private_origins, "lazy_modules": lazy_modules, "canonical_wnn_files": wnn_files, "source_commit": manifest["source_commit"], "source_manifest_sha256": manifest_sha, "runtime": runtime, "blocked_reads": blocked, "cuda_calls": 0}, sort_keys=True))
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(PACKET / "child_ema_terminal_check_v1.py"),
                str(
                    PREPARATION
                    / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py"
                ),
                str(PREPARATION / "parent_fit_terminal_check_v3.py"),
                str(PREPARATION / "propensity_terminal_support.py"),
                str(PREPARATION / "teacher_s5_terminal_check_v1.py"),
                str(SOURCE_ROOT),
                str(SOURCE_MANIFEST),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        proof = json.loads(completed.stdout)
        self.assertEqual(proof["blocked_reads"], [])
        self.assertEqual(proof["cuda_calls"], 0)
        self.assertEqual(len(proof["modules"]), 5)
        self.assertEqual(len(proof["private_origins"]), 5)
        self.assertTrue(all(item["origin_is_private_temp_d"] for item in proof["private_origins"]))
        self.assertEqual(
            proof["lazy_modules"],
            [
                "wnn.training.cli",
                "wnn.run_identity",
                "wnn.training.input_representation",
                "wnn.training.input_representation_fit",
                "wnn.training.input_representation_dictionary",
            ],
        )
        self.assertEqual(proof["source_commit"], SOURCE_COMMIT)
        self.assertEqual(proof["source_manifest_sha256"], SOURCE_MANIFEST_SHA)
        self.assertGreater(len(proof["canonical_wnn_files"]), 5)
        self.assertEqual(proof["runtime"]["torch_version"], torch.__version__)
        self.assertEqual(proof["runtime"]["cuda_visible_devices"], "")
        self.assertIs(proof["runtime"]["torch_cuda_initialized"], False)
        proof_path = PACKET / "IMPORT_READ_CUDA_PROOF.json"
        proof_path.write_text(
            json.dumps(
                {
                    "schema": "fg19.child-ema-cold-import-read-cuda-proof.v1",
                    "status": "PASS",
                    "checker_path": str(PACKET / "child_ema_terminal_check_v1.py"),
                    "checker_sha256": hashlib.sha256(
                        (PACKET / "child_ema_terminal_check_v1.py").read_bytes()
                    ).hexdigest(),
                    "test_path": str(Path(__file__).resolve()),
                    "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "source_tree": str(SOURCE_ROOT),
                    "source_commit": SOURCE_COMMIT,
                    "source_manifest_path": str(SOURCE_MANIFEST.resolve()),
                    "source_manifest_sha256": SOURCE_MANIFEST_SHA,
                    "method": "Fresh subprocess copied exactly the five pinned private Python files into a temporary D, put only D on sys.path for those private module names, and asserted every module origin and SHA/size. It exercised the lazy S5 CLI and tensor-identity imports plus the three lazy representation modules used by the reused CONTROL dictionary checker, and checked every loaded WNN origin/content digest against the 167-entry source manifest. Its audit hook denied model, array, parquet, and B source-root opens; CUDA probes raise if called.",
                    "observed": proof,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
