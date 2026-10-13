from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_cross_team_submission import (
    NATIVE_VARIANTS,
    _read_native_manifest,
    _read_qat_release_spec,
)
from scripts.templates.qat_archive import read_qat_archive
from wnn.quantization import quantize_model
from wnn.quantization_export import read_packed_quantized_export, write_packed_quantized_export


def _write_source_archive(path: Path, preset: str = "recurrent_w4") -> Path:
    rng = np.random.default_rng(2047)
    weights = {
        "mean": np.zeros(211, np.float32),
        "std": np.ones(211, np.float32),
        "inp.bias": np.zeros(5, np.float32),
        "inp.weight": rng.normal(size=(5, 211)).astype(np.float32),
        "gru.weight_ih_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "gru.weight_hh_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "gru.bias_ih_l0": np.zeros(15, np.float32),
        "gru.bias_hh_l0": np.zeros(15, np.float32),
        "head.0.weight": rng.normal(size=(5, 10)).astype(np.float32),
        "head.0.bias": np.zeros(5, np.float32),
        "head.2.weight": rng.normal(size=(2, 5)).astype(np.float32),
        "head.2.bias": np.zeros(2, np.float32),
    }
    effective, encoded, _ = quantize_model(weights, preset)
    write_packed_quantized_export(path, effective, preset, encoded)
    return path


def _mutate_archive(source: Path, target: Path, field: str, update) -> Path:
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload[field] = update(payload[field])
    with target.open("wb") as output:
        np.savez_compressed(output, **payload)
    return target


@pytest.mark.parametrize("preset", ["recurrent_w4", "input_w4_recurrent_w8"])
def test_package_reader_matches_source_reader_without_requantization(
    tmp_path: Path, preset: str
) -> None:
    path = _write_source_archive(tmp_path / "source_format.npz", preset)
    source_weights, source_codes, metadata = read_packed_quantized_export(path)
    package_weights, package_codes, package_metadata = read_qat_archive(path, preset)

    assert package_metadata == metadata
    assert package_codes.keys() == source_codes.keys()
    for name in source_codes:
        assert package_codes[name].dtype == source_codes[name].dtype
        np.testing.assert_array_equal(package_codes[name].view(np.uint8), source_codes[name].view(np.uint8))
    for name in source_weights:
        np.testing.assert_array_equal(package_weights[name], source_weights[name])


def test_package_reader_rejects_invalid_q4_code_and_padding(tmp_path: Path) -> None:
    source = _write_source_archive(tmp_path / "valid.npz")
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(archive["__quantized_rows_metadata__"].tobytes())
    q4 = metadata["matrices"]["gru.weight_ih_l0"]["blocks"][0]
    q4_field = q4["packed_codes"]

    bad_code = _mutate_archive(
        source,
        tmp_path / "bad_minus8.npz",
        q4_field,
        lambda value: np.concatenate(
            [np.array([(int(value[0]) & 0xF0) | 8], np.uint8), value[1:]]
        ),
    )
    with pytest.raises(ValueError, match="invalid -8"):
        read_qat_archive(bad_code)

    bad_pad = _mutate_archive(
        source,
        tmp_path / "bad_padding.npz",
        q4_field,
        lambda value: np.concatenate(
            [value[:-1], np.array([(int(value[-1]) & 0x0F) | 0x10], np.uint8)]
        ),
    )
    with pytest.raises(ValueError, match="zero high-nibble pad"):
        read_qat_archive(bad_pad)


def test_package_reader_rejects_invalid_q8_code_and_scale(tmp_path: Path) -> None:
    source = _write_source_archive(tmp_path / "valid_mixed.npz", "input_w4_recurrent_w8")
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(archive["__quantized_rows_metadata__"].tobytes())
    q8 = metadata["matrices"]["gru.weight_hh_l0"]["blocks"][0]
    code_field = q8["packed_codes"]
    scale_field = q8["scales"]

    bad_code = _mutate_archive(
        source,
        tmp_path / "bad_minus128.npz",
        code_field,
        lambda value: np.concatenate([np.array([128], np.uint8), value[1:]]),
    )
    with pytest.raises(ValueError, match="invalid -128"):
        read_qat_archive(bad_code)

    bad_scale = _mutate_archive(
        source,
        tmp_path / "bad_nan_scale.npz",
        scale_field,
        lambda value: np.concatenate([np.array([np.nan], np.float32), value[1:]]),
    )
    with pytest.raises(ValueError, match="scales are invalid"):
        read_qat_archive(bad_scale)

    with np.load(source, allow_pickle=False) as archive:
        codes = archive[code_field].copy()
        shape = tuple(metadata["matrices"]["gru.weight_hh_l0"]["blocks"][0]["shape"])
    signed_codes = codes.view(np.int8).reshape(shape)
    row = int(np.flatnonzero(np.any(signed_codes != 0, axis=1))[0])
    zero_nonzero_row = _mutate_archive(
        source,
        tmp_path / "zero_scale_nonzero_code.npz",
        scale_field,
        lambda value: np.concatenate(
            [
                np.zeros(row, np.float32),
                np.array([0.0], np.float32),
                value[row + 1 :],
            ]
        ),
    )
    with pytest.raises(ValueError, match="scales are invalid"):
        read_qat_archive(zero_nonzero_row)


def test_package_reader_requires_the_preserved_preset(tmp_path: Path) -> None:
    path = _write_source_archive(tmp_path / "source.npz")
    with pytest.raises(ValueError, match="differs from the package manifest"):
        read_qat_archive(path, "input_w4_recurrent_w8")


def test_release_spec_and_native_manifest_are_hash_bound(tmp_path: Path) -> None:
    archive = _write_source_archive(tmp_path / "qat.npz")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint source fixture")
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    spec = {
        "version": 1,
        "replacement": {
            "index": 3,
            "candidate": "E",
            "weights": "weights_0.npz",
            "source_member_sha256": "a" * 64,
            "coef": 0.25,
        },
        "archive": {"path": str(archive), "sha256": digest(archive), "preset": "recurrent_w4"},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest(checkpoint),
            "run_reference": "seed3@ep0",
            "training_rule": "identity STE; LSQ-inspired learned scales",
            "data_membership": "all training data",
            "seed": 3,
            "updates": 14_920,
            "parent_checkpoint_sha256": "b" * 64,
            "scale_learning": {"version": 1},
        },
    }
    spec_path = tmp_path / "qat_release.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    _, binding = _read_qat_release_spec(spec_path)
    assert binding["archive"]["sha256"] == digest(archive)
    assert binding["checkpoint"]["sha256"] == digest(checkpoint)
    assert binding["archive"]["metadata"]["preset"] == "recurrent_w4"

    lib = tmp_path / "native.abi3.so"
    lib.write_bytes(b"binary fixture")
    native = {
        "native_api": "exact_q8_codes_scales_v1",
        "abi": "abi3-py311",
        "target_cpu": "x86-64-v2",
        "runtime_dispatch": True,
        "files": {
            name: {"path": str(lib), "sha256": digest(lib), "bytes": lib.stat().st_size}
            for name in NATIVE_VARIANTS
        },
    }
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    _, native_binding = _read_native_manifest(native_path)
    assert native_binding["native_api"] == "exact_q8_codes_scales_v1"
    assert set(native_binding["files"]) == set(NATIVE_VARIANTS)

    native["files"][NATIVE_VARIANTS[0]]["sha256"] = "0" * 64
    native_path.write_text(json.dumps(native), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _read_native_manifest(native_path)


def test_qat_callback_replaces_only_target_native_code_map(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "package"
    (package / "v2").mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "scripts/templates/qat_solution.py", package / "solution.py")
    shutil.copyfile(Path(__file__).parents[1] / "scripts/templates/qat_archive.py", package / "qat_archive.py")
    source_archive = _write_source_archive(tmp_path / "source.npz")
    shutil.copyfile(source_archive, package / "qat_weights_3.npz")
    _, exact_codes, _ = read_packed_quantized_export(source_archive)
    for index in range(3):
        np.savez(package / f"weights_{index}.npz", **{"placeholder": np.zeros(1, np.float32)})
    (package / "members.json").write_text(
        json.dumps(
            [
                {"weights": f"weights_{index}.npz", "coef": 0.25}
                for index in range(3)
            ]
            + [{"qat_archive": "qat_weights_3.npz", "qat_preset": "recurrent_w4", "coef": 0.25}]
        ),
        encoding="utf-8",
    )

    class FakeEnsemble:
        calls = []

        def __init__(self, weights, coefficients, **kwargs):
            self.weights = weights
            self.coefficients = coefficients
            self.kwargs = kwargs
            self.calls.append(self)
            self.precision = "q8vnni"

        def quantization_codes(self):
            return [{"generated": np.array([index], np.int8)} for index in range(4)]

        def predict(self, data_point):
            return data_point

    fake_native = types.ModuleType("wnn_native")
    fake_native.EnsemblePredictor = FakeEnsemble
    monkeypatch.setitem(sys.modules, "wnn_native", fake_native)
    monkeypatch.setenv("WNN_NATIVE_LEVEL", "v2")
    module_name = "qat_package_solution_fixture"
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
        module_name, package / "solution.py"
    )
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    monkeypatch.syspath_prepend(str(package))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        model = module.PredictionModel()
    finally:
        sys.modules.pop(module_name, None)

    assert len(FakeEnsemble.calls) == 2
    seed, final = FakeEnsemble.calls
    assert "q8_codes" not in seed.kwargs
    assert seed.kwargs["budget_s"] is None
    assert final.kwargs["precision"] == "q8vnni"
    assert final.kwargs["budget_s"] is None
    assert final.coefficients == [np.float32(0.25)] * 4
    maps = final.kwargs["q8_codes"]
    assert [row["generated"][0] for row in maps[:3]] == [0, 1, 2]
    assert maps[3].keys() == exact_codes.keys()
    for name in exact_codes:
        np.testing.assert_array_equal(maps[3][name], exact_codes[name])
    marker = object()
    assert model.predict(marker) is marker


def test_actual_s4_learned_archive_when_supplied() -> None:
    archive_path = os.environ.get("WNN_QAT_SOURCE_FIXTURE")
    expected_hash = os.environ.get("WNN_QAT_SOURCE_FIXTURE_SHA256")
    if not archive_path or not expected_hash:
        pytest.skip("set WNN_QAT_SOURCE_FIXTURE and its expected SHA-256 for the bounded actual fixture")
    path = Path(archive_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    weights, encoded, metadata = read_qat_archive(path, expected_preset="recurrent_w4")
    assert metadata["version"] == 2
    assert weights["gru.weight_ih_l0"].shape == (768, 256)
    assert weights["head.0.weight"].shape == (256, 512)
    assert encoded["head.0.weight.0.codes"].shape == (256, 256)


def test_actual_native_exact_import_and_unchanged_member_replay_when_supplied() -> None:
    archive_value = os.environ.get("WNN_QAT_SOURCE_FIXTURE")
    archive_hash = os.environ.get("WNN_QAT_SOURCE_FIXTURE_SHA256")
    native_value = os.environ.get("WNN_NATIVE_EXACT_LIBRARY")
    baseline_value = os.environ.get("WNN_XTEAM_H4_PACKAGE")
    if not all((archive_value, archive_hash, native_value, baseline_value)):
        pytest.skip("set the QAT archive, exact native library, and xteam_h4 package fixture paths")
    archive_path = Path(archive_value)
    native_path = Path(native_value).resolve(strict=True)
    base_path = Path(baseline_value).resolve(strict=True)
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == archive_hash
    assert native_path.name == "wnn_native.abi3.so"
    sys.path.insert(0, str(native_path.parent))
    try:
        native = importlib.import_module("wnn_native")
        qat_weights, qat_codes, _ = read_qat_archive(archive_path, "recurrent_w4")
        base_weights = []
        for index in range(4):
            with np.load(base_path / f"weights_{index}.npz", allow_pickle=False) as source:
                base_weights.append(
                    {
                        name: np.ascontiguousarray(source[name], dtype=np.float32)
                        for name in source.files
                    }
                )
        assert base_weights[3].keys() == qat_weights.keys()
        for name in qat_weights:
            assert base_weights[3][name].shape == qat_weights[name].shape
        coefficients = [np.float32(0.25)] * 4
        common = {
            "fast_activations": True,
            "precision": "q8vnni",
            "budget_s": None,
        }

        baseline = native.EnsemblePredictor(base_weights, coefficients, **common)
        baseline_codes = baseline.quantization_codes()
        exact_roundtrip = native.EnsemblePredictor(
            base_weights, coefficients, q8_codes=baseline_codes, **common
        )
        roundtrip_codes = exact_roundtrip.quantization_codes()
        assert len(roundtrip_codes) == 4
        for before, after in zip(baseline_codes, roundtrip_codes, strict=True):
            assert before.keys() == after.keys()
            for name in before:
                np.testing.assert_array_equal(before[name], after[name])

        states = [
            np.linspace(-0.4, 0.6, 112, dtype=np.float32),
            np.linspace(0.2, -0.2, 112, dtype=np.float32),
        ]
        trace = [
            SimpleNamespace(seq_ix=7, state=states[0], need_prediction=True),
            SimpleNamespace(seq_ix=7, state=states[1], need_prediction=False),
            SimpleNamespace(seq_ix=8, state=states[0], need_prediction=True),
        ]
        base_outputs = [baseline.predict(point) for point in trace]
        exact_outputs = [exact_roundtrip.predict(point) for point in trace]
        for before, after in zip(base_outputs, exact_outputs, strict=True):
            if before is None:
                assert after is None
            else:
                np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(base_outputs[0], base_outputs[2])

        qat_weights_for_blend = list(base_weights)
        qat_weights_for_blend[3] = qat_weights
        seed = native.EnsemblePredictor(qat_weights_for_blend, coefficients, **common)
        qat_maps = seed.quantization_codes()
        qat_maps[3] = qat_codes
        qat_model = native.EnsemblePredictor(
            qat_weights_for_blend, coefficients, q8_codes=qat_maps, **common
        )
        imported = qat_model.quantization_codes()[3]
        assert imported.keys() == qat_codes.keys()
        for name in qat_codes:
            np.testing.assert_array_equal(imported[name], qat_codes[name])
        replay_a = qat_model.predict(trace[0])
        qat_model.predict(trace[1])
        replay_b = qat_model.predict(trace[2])
        np.testing.assert_array_equal(replay_a, replay_b)
    finally:
        sys.path.remove(str(native_path.parent))


def test_actual_qat_builder_cli_and_callback_when_supplied(monkeypatch) -> None:
    archive_value = os.environ.get("WNN_QAT_SOURCE_FIXTURE")
    archive_hash = os.environ.get("WNN_QAT_SOURCE_FIXTURE_SHA256")
    native_manifest_value = os.environ.get("WNN_QAT_NATIVE_MANIFEST")
    package_fixture_root = os.environ.get("WNN_QAT_PACKAGE_TEST_ROOT")
    if not all((archive_value, archive_hash, native_manifest_value, package_fixture_root)):
        pytest.skip("set QAT source, native manifest, and a durable package-test directory")
    root = Path(__file__).parents[1]
    fixture_root = Path(package_fixture_root).resolve(strict=True)
    archive_path = Path(archive_value).resolve(strict=True)
    inventory_path = root / "research/campaign_2026-10-13/team_models_r1/INVENTORY.json"
    recipe_source = root / "research/campaign_2026-10-13/team_models_r1/SUBMISSION_RECIPE_R1.json"
    qat_manifest_path = archive_path.parent / "manifest.json"
    qat_manifest = json.loads(qat_manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    original_member = inventory["packages"]["E"]["files"]["weights_0.npz"]
    checkpoint = qat_manifest["parent"]
    quantization = qat_manifest["checkpoint_quantization"]
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert archive_digest == archive_hash

    recipe = json.loads(recipe_source.read_text(encoding="utf-8"))
    recipe["name"] = "qat_cross_team_builder_fixture_r1"
    recipe_path = fixture_root / "qat_recipe_fixture.json"
    release_path = fixture_root / "qat_release_fixture.json"
    output_path = fixture_root / "qat_package_direct_cli_fixture_output"
    module_output_path = fixture_root / "qat_package_module_cli_fixture_output"
    if output_path.exists() or module_output_path.exists():
        raise FileExistsError(output_path if output_path.exists() else module_output_path)
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    release = {
        "version": 1,
        "replacement": {
            "index": 3,
            "candidate": "E",
            "weights": "weights_0.npz",
            "source_member_sha256": original_member["sha256"],
            "coef": 0.25,
        },
        "archive": {
            "path": str(archive_path),
            "sha256": archive_digest,
            "preset": quantization["preset"],
        },
        "checkpoint": {
            "path": checkpoint["checkpoint_path"],
            "sha256": checkpoint["checkpoint_sha256"],
            "run_reference": checkpoint["run_reference"],
            "training_rule": quantization["training_rule"],
            "data_membership": "10% component subset; exposed F0 diagnostic only",
            "seed": 4,
            "updates": 1_440,
            "parent_checkpoint_sha256": "e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c",
            "scale_learning": quantization["scale_learning"],
        },
    }
    release_path.write_text(json.dumps(release), encoding="utf-8")

    def build_cli(entrypoint: list[str], destination: Path) -> dict[str, object]:
        command = [
            sys.executable,
            *entrypoint,
            "--inventory",
            str(inventory_path),
            "--recipe",
            str(recipe_path),
            "--qat-release-spec",
            str(release_path),
            "--native-manifest",
            native_manifest_value,
            "--out-dir",
            str(destination),
            "--max-zip-bytes",
            "20000000",
        ]
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            command,
            cwd=root,
            env=child_env,
            capture_output=True,
            check=False,
            text=True,
            timeout=160,
        )
        assert completed.returncode == 0, (
            f"builder CLI failed ({' '.join(entrypoint)}):\n{completed.stderr}"
        )
        return json.loads(completed.stdout)

    try:
        direct_result = build_cli(
            [str(root / "scripts/build_cross_team_submission.py")], output_path
        )
        module_result = build_cli(["-m", "scripts.build_cross_team_submission"], module_output_path)
        manifest_name = "qat_cross_team_builder_fixture_r1.manifest.json"
        assert direct_result["manifest"] == str(output_path / manifest_name)
        assert module_result["manifest"] == str(
            module_output_path / manifest_name
        )
        manifest = json.loads(Path(direct_result["manifest"]).read_text(encoding="utf-8"))
        package_root = output_path / "pkg"
        assert manifest["artifact"]["bytes"] < 20_000_000
        assert module_result["zip"]["sha256"] == direct_result["zip"]["sha256"]
        assert module_result["zip"]["bytes"] == direct_result["zip"]["bytes"]
        assert (package_root / "solution.py").is_file()
        assert (package_root / "qat_provenance.json").is_file()
        packaged_members = json.loads((package_root / "members.json").read_text())
        assert [member["coef"] for member in packaged_members] == [0.25] * 4
        assert packaged_members[3]["qat_archive"] == "qat_weights_3.npz"
        assert packaged_members[3]["training_checkpoint_sha256"] == checkpoint[
            "checkpoint_sha256"
        ]

        monkeypatch.setenv("WNN_NATIVE_LEVEL", "v2")
        monkeypatch.syspath_prepend(str(package_root))
        importlib.invalidate_caches()
        module_name = "qat_cross_team_built_callback_fixture"
        callback_spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
            module_name, package_root / "solution.py"
        )
        callback = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(
            callback_spec
        )
        sys.path.insert(0, str(package_root / "v2"))
        sys.modules[module_name] = callback
        try:
            callback_spec.loader.exec_module(callback)
            model = callback.PredictionModel()
            point = SimpleNamespace(
                seq_ix=1,
                state=np.linspace(-0.1, 0.1, 112, dtype=np.float32),
                need_prediction=True,
            )
            first = model.predict(point)
            point.seq_ix = 2
            replay = model.predict(point)
            assert first.shape == (2,)
            np.testing.assert_array_equal(first, replay)
        finally:
            sys.modules.pop(module_name, None)
            sys.path.remove(str(package_root / "v2"))
    finally:
        shutil.rmtree(output_path, ignore_errors=True)
        shutil.rmtree(module_output_path, ignore_errors=True)
        recipe_path.unlink(missing_ok=True)
        release_path.unlink(missing_ok=True)
