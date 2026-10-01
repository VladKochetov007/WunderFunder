from __future__ import annotations

import ast
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from build_patch import proposed_source
from map_support import PRICE_COLUMNS, UNKNOWN, load_table, lookup, read_prices, sha256
from torch import Tensor
from trainer_support import PackedSpreadCache, structural_loss, validate_spread_cache

from wnn.data import FEATURE_COLUMNS
from wnn.outcomes import N_SP_AUX

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def legacy(
    output: Tensor, spread: Tensor, branch: Tensor, classes: Tensor, weights: Tensor
) -> Tensor:
    """Independent exact copy of original three-head CE aggregation."""
    o = output.float()
    wv = weights.reshape(-1)
    ce = 0.0
    for a0, a1, lab in [
        (2, 2 + N_SP_AUX, spread),
        (2 + N_SP_AUX, 2 + N_SP_AUX + 3, branch),
        (2 + N_SP_AUX + 3, 2 + N_SP_AUX + 7, classes),
    ]:
        ce = ce + F.cross_entropy(
            o[..., a0:a1].reshape(-1, a1 - a0), lab.reshape(-1), reduction="none"
        )
    return (ce * wv).sum() / wv.sum().clamp_min(1e-8)


def loss_proof() -> dict:
    """Show matched masks, original denominator, and unchanged nonspread gradients."""
    torch.manual_seed(3)
    logits = torch.randn(2, 17, 23)
    proxy = torch.randint(0, N_SP_AUX, (2, 17))
    current = torch.randint(0, N_SP_AUX, (2, 17))
    branch, classes = torch.randint(0, 3, (2, 17)), torch.randint(0, 4, (2, 17))
    weights = torch.rand(2, 17)
    weights[:, 0] = 0
    encoded = current.to(torch.uint8)
    encoded[:, ::3] = UNKNOWN
    original_logits = logits.clone().requires_grad_()
    original = legacy(original_logits, proxy, branch, classes, weights)
    original.backward()
    identical_logits = logits.clone().requires_grad_()
    identical = structural_loss(identical_logits, proxy, branch, classes, weights, 2, N_SP_AUX)
    identical.backward()
    assert torch.equal(original, identical)
    assert torch.equal(original_logits.grad, identical_logits.grad)
    gradients, losses = [], []
    for labels in [proxy, current]:
        argument = logits.clone().requires_grad_()
        loss = structural_loss(argument, labels, branch, classes, weights, 2, N_SP_AUX, encoded)
        loss.backward()
        gradients.append(argument.grad)
        losses.append(float(loss.detach()))
        expected = argument.detach()[..., 2 : 2 + N_SP_AUX].softmax(-1)
        expected = expected - F.one_hot(labels, N_SP_AUX)
        expected *= (weights * (encoded != UNKNOWN) / weights.sum())[..., None]
        assert torch.allclose(argument.grad[..., 2 : 2 + N_SP_AUX], expected, atol=1e-8, rtol=1e-6)
    nonspread = torch.cat([gradients[0][..., :2], gradients[0][..., 2 + N_SP_AUX :]], dim=-1)
    nonspread_other = torch.cat([gradients[1][..., :2], gradients[1][..., 2 + N_SP_AUX :]], dim=-1)
    assert torch.equal(nonspread, nonspread_other)
    assert torch.equal(gradients[0][..., 2 + N_SP_AUX :], original_logits.grad[..., 2 + N_SP_AUX :])
    assert gradients[0][..., 2 : 2 + N_SP_AUX][encoded == UNKNOWN].eq(0).all()
    assert gradients[1][..., 2 : 2 + N_SP_AUX][encoded == UNKNOWN].eq(0).all()
    changed_unknown = proxy.clone()
    changed_unknown[encoded == UNKNOWN] = (changed_unknown[encoded == UNKNOWN] + 1) % N_SP_AUX
    a = structural_loss(logits, proxy, branch, classes, weights, 2, N_SP_AUX, encoded)
    b = structural_loss(logits, changed_unknown, branch, classes, weights, 2, N_SP_AUX, encoded)
    assert torch.equal(a, b)
    all_known = torch.zeros_like(encoded)
    assert torch.equal(
        legacy(logits, proxy, branch, classes, weights),
        structural_loss(logits, proxy, branch, classes, weights, 2, N_SP_AUX, all_known),
    )
    unknown_only = torch.full_like(encoded, UNKNOWN)
    zeros = logits.clone().requires_grad_()
    structural_loss(zeros, proxy, branch, classes, weights, 2, N_SP_AUX, unknown_only).backward()
    assert zeros.grad[..., 2 : 2 + N_SP_AUX].eq(0).all() and torch.isfinite(zeros.grad).all()
    return {
        "original_mode_loss_and_gradient_exact": True,
        "branch_class_gradients_exact": True,
        "masked_rows_zero_spread_gradient": True,
        "unknown_proxy_label_invariant": True,
        "all_known_equals_original": True,
        "all_unknown_finite_zero_spread": True,
        "spread_gradient_matches_original_all_sw_denominator": True,
        "synthetic_arm_losses": losses,
        "same_known_mask": True,
    }


def cache_proof(selection: dict) -> dict:
    """Check exact FIT row alignment using prices only, never VALID targets/masks."""
    provenance = validate_spread_cache(
        HERE,
        selection["split_manifest_sha256"],
        selection["fit_train_groups"],
        selection["fit_valid_groups"],
        selection["evaluation_valid_groups"],
    )
    table = load_table(HERE / "interior_code_table.json")
    checked = []
    for split in ["train", "valid"]:
        source = pq.ParquetFile(ROOT / f"datasets/{split}.parquet")
        cache = PackedSpreadCache(HERE, split)
        groups = selection[f"fit_{split}_groups"]
        chosen = [groups[0], groups[len(groups) // 2], groups[-1]]
        for group in chosen:
            prices = read_prices(source, group)
            expected = lookup(prices, table)
            actual = cache.read(group)
            assert np.array_equal(expected, actual)
            full_features = source.read_row_group(group, columns=FEATURE_COLUMNS, use_threads=False)
            from_training_float32 = np.column_stack(
                [full_features.column(column).to_numpy() for column in PRICE_COLUMNS]
            ).astype(np.float32)
            assert np.array_equal(prices.view(np.uint32), from_training_float32.view(np.uint32))
            assert np.array_equal(actual, lookup(from_training_float32, table))
            checked.append(
                {
                    "split": split,
                    "group": group,
                    "rows": len(prices),
                    "lookup_exact": True,
                    "float32_training_parquet_alignment_exact": True,
                    "known_rows": int((actual != UNKNOWN).sum()),
                }
            )
        forbidden = (
            selection["evaluation_valid_groups"][0] if split == "valid" else max(groups) + 10000
        )
        try:
            cache.read(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError("outside FIT row group silently entered cache")
        del source, cache
    return {
        "provenance": provenance,
        "checked_fit_groups": checked,
        "outside_fit_rejected": True,
        "read_columns": FEATURE_COLUMNS,
        "targets_or_scoring_masks_read": False,
    }


def dataset_proof() -> dict:
    """Exercise proposed batch tuple layout with synthetic targets, no raw labels."""
    _, proposed = proposed_source()
    definition = next(
        node
        for node in ast.parse(proposed).body
        if isinstance(node, ast.ClassDef) and node.name == "Sequences"
    )
    features = np.zeros((20_000, 112), dtype=np.float32)
    targets = np.zeros((20_000, 2), dtype=np.float32)

    class SyntheticFile:
        """Synthetic data reader replaces all parquet target/mask access."""

        def __init__(self, path: Path) -> None:
            self.path = path

        def read(self, index: int) -> SimpleNamespace:
            return SimpleNamespace(features=features, targets=targets, scored=None)

    namespace: dict[str, object] = {
        "Dataset": torch.utils.data.Dataset,
        "ROOT": ROOT,
        "Path": Path,
        "np": np,
        "torch": torch,
        "SequenceFile": SyntheticFile,
        "HEURISTICS": False,
        "PackedSpreadCache": PackedSpreadCache,
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), "proposed_dataset", "exec"),
        namespace,
    )
    dataset = namespace["Sequences"]
    original = dataset("train", [0])[0]
    cached = dataset("train", [0], spread_cache_root=HERE)[0]
    assert len(original) == 3 and len(cached) == 4
    assert all(torch.equal(a, b) for a, b in zip(original, cached[:3], strict=True))
    assert cached[-1].dtype == torch.uint8 and cached[-1].shape == (20_000,)
    propensity_path = HERE / "synthetic_propensity.npy"
    if propensity_path.exists():
        raise FileExistsError("synthetic fixture exists")
    np.save(propensity_path, np.full((1, 20_000), 0.5, dtype=np.float32))
    with_prop = dataset("train", [0], prop_cache=propensity_path, spread_cache_root=HERE)[0]
    assert (
        len(with_prop) == 5
        and with_prop[3].dtype == torch.float32
        and with_prop[4].dtype == torch.uint8
    )
    assert torch.equal(cached[-1], with_prop[-1])
    return {
        "targets_features_scored_exact": True,
        "no_prop_tuple": 4,
        "with_prop_tuple": 5,
        "last_tensor_supervision_only": True,
        "synthetic_targets_only": True,
    }


def main() -> None:
    """Produce actor CPU proof without candidate prediction/evaluation."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = {
        "candidate_scored": False,
        "source_sha256": sha256(Path(__file__)),
        "support_sha256": sha256(HERE / "trainer_support.py"),
        "N_SP_AUX": N_SP_AUX,
    }
    try:
        selection = json.loads((HERE / "selection.json").read_text())
        result["loss"] = loss_proof()
        result["cache"] = cache_proof(selection)
        result["dataset"] = dataset_proof()
        parent = ROOT / "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt"
        assert sha256(parent) == selection["source_sha256"][str(parent.relative_to(ROOT))]
        checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
        result["parent_n_out"] = checkpoint["model"]["head.2.weight"].shape[0]
        assert result["parent_n_out"] == 2 + N_SP_AUX + 3 + 4 + 2
        assert not torch.cuda.is_initialized()
        result["passed"] = True
    except (AssertionError, RuntimeError, ValueError, OSError) as error:
        result.update({"passed": False, "failure": str(error)})
        raise
    finally:
        result["cuda_initialized"] = torch.cuda.is_initialized()
        result["wall_seconds"] = time.monotonic() - started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with (HERE / "cpu_proof.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
