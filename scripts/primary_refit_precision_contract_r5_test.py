from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/experiment_drafts/multistage-optuna-b300-20261031"
sys.path.insert(0, str(PACKET))
from primary_refit_precision_contract_r5 import validate_precision_contract  # noqa: E402


def _actual_source4() -> tuple[dict[str, object], dict[str, object]]:
    profile = json.loads(
        (
            PACKET
            / "receiving/actual-profiles-r6/profiles/lstm-b256-r6-lstm384x2/profile_receipt.json"
        ).read_bytes()
    )
    config = json.loads(
        (PACKET / "receiving/actual-profiles-r6/requests/r6/lstm384x2.json").read_bytes()
    )
    assert (
        profile["identity"]["source_sha256"]
        == "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
    )
    return config, profile["identity"]["runtime"]


def test_vendored_precision_helper_is_exact_source6_semantics_and_torch_free() -> None:
    helper = PACKET / "primary_refit_precision_contract_r5.py"
    assert (
        helper.read_bytes() == (ROOT / "src/wnn/training/sequence_cache_precision.py").read_bytes()
    )
    imports = [
        node
        for node in ast.walk(ast.parse(helper.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        isinstance(node, ast.ImportFrom) and node.module in {"__future__", "typing"}
        for node in imports
    )


def test_actual_source4_runtime_envelope_passes_without_source4_cache_module() -> None:
    config, runtime = _actual_source4()
    assert validate_precision_contract(config, runtime, runtime) == runtime["precision"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("precision", "mode", "bf16_bf16x9"),
        ("precision", "parameter_dtype", "bfloat16"),
        ("precision", "cublas_emulate_single_precision", "1"),
        ("compiler", "enabled", True),
        ("compiler", "backend", "inductor"),
    ],
)
def test_standalone_helper_rejects_precision_compiler_drift(
    section: str, key: str, value: object
) -> None:
    config, runtime = _actual_source4()
    changed = copy.deepcopy(runtime)
    changed[section][key] = value
    with pytest.raises(ValueError):
        validate_precision_contract(config, changed, changed)


def test_r5_primary_alias_gate_preserves_exact_membership_and_ancestry() -> None:
    tree = ast.parse((PACKET / "review_primary_refit_profile_r5.py").read_text())
    expression = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "fit_only_normalization"
            for target in node.targets
        )
    )
    gate = compile(ast.Expression(expression), "r5-primary-alias-gate", "eval")
    primary = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
    fits = [["train", index] for index in range(10325)]
    evaluation = [["valid", index] for index in range(1873)]
    normalization = {
        "scope": "prefix",
        "fit_prefix_sequences": 32,
        "sequence_count": 32,
        "identity_sha256": "norm",
        "moments_sha256": "a" * 64,
    }
    data = {
        "fit_members": fits,
        "evaluation_members": evaluation,
        "training_mask": "all_after_warmup",
        "split_sha256": primary,
        "primary_split_sha256": None,
        "evaluation_source": "valid",
        "evaluation_role": "valid",
        "teacher": None,
        "initializer": None,
    }
    context = {
        "data": data,
        "fit_members": fits,
        "eval_members": evaluation,
        "config": {"training_mask": "all_after_warmup"},
        "split": SimpleNamespace(sha256=primary),
        "normalization": normalization,
        "norm_sources": {"datasets/train.parquet": "trusted"},
        "train_fingerprint": "trusted",
        "normalization_identity_sha256": lambda value: "norm",
        "norm_identity": {},
    }
    assert eval(gate, context)
    data["primary_split_sha256"] = primary
    assert eval(gate, context)
    for key, value in (
        ("primary_split_sha256", "wrong"),
        ("split_sha256", "wrong"),
        ("fit_members", fits[:-1]),
        ("evaluation_members", evaluation[:-1]),
        ("teacher", {"ancestor": "unknown"}),
        ("initializer", {"ancestor": "unknown"}),
        ("evaluation_source", "train"),
        ("evaluation_role", "control"),
    ):
        changed = dict(data) | {key: value}
        assert not eval(gate, context | {"data": changed})
