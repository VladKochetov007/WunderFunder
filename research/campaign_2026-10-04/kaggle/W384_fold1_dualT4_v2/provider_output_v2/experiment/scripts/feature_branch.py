"""H090 frozen modern-trunk branch/U screen with matched fitting budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from feature_branch_head import ARMS, PARAMETER_BUDGET, Head, empirical_moments, rational_readout
from feature_screen import assert_outer_separation, digest_file, fit_residual, load_cache
from feature_v2_screen import compare

from wnn.gmetric import reference_global_wp
from wnn.nn import weighted_mse_loss
from wnn.outcomes import outcome_codes
from wnn.resources import configure_torch


def label_classes(data: dict, labels_path: Path) -> np.ndarray:
    """Return class supervision/strata by sequence identity; never an inference input."""
    with np.load(labels_path, allow_pickle=False) as stored:
        sequence, classes = stored["seq_ix"], stored["market_class"]
    if sequence.shape != classes.shape or len(np.unique(sequence)) != len(sequence):
        raise ValueError("Class cache has invalid sequence identity")
    mapping = {int(seq): int(label) for seq, label in zip(sequence, classes, strict=True)}
    values = np.array([mapping[int(seq)] for seq in data["sequence"]], dtype=np.int64)
    if not np.isin(values, range(4)).all():
        raise ValueError("The common fixed sample contains an unknown class label")
    return values


def train_heads(
    fit: dict, classes: np.ndarray, device: str, updates: int = 256
) -> tuple[dict, dict, dict]:
    """Train every head without opening evaluation inputs/labels or selecting checkpoints."""
    if not 1 <= updates <= 1024:
        raise ValueError("Bounded fixed update count required")
    configure_torch()
    torch.use_deterministic_algorithms(True)
    target = torch.as_tensor(fit["y"], dtype=torch.float32, device=device)
    code = outcome_codes(target[:, None], torch.as_tensor(classes, device=device))[:, 0]
    moment1, moment0, support = empirical_moments(code.cpu().numpy(), fit["y"], fit["fit_weight"])
    m1, m0 = (torch.as_tensor(value, device=device) for value in (moment1, moment0))
    state = torch.as_tensor(fit["state"], dtype=torch.float32, device=device)
    weight = torch.as_tensor(fit["fit_weight"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(
        fit["state"].mean(0, dtype=np.float64), dtype=torch.float32, device=device
    )
    scale = torch.as_tensor(
        np.maximum(fit["state"].std(0, dtype=np.float64), 1e-4), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(271828)
    batches = []
    while len(batches) < updates:
        order = rng.permutation(len(state))
        batches.extend(order[start : start + 512] for start in range(0, len(order), 512))
    batches = batches[:updates]
    heads, history, counts = {}, {}, {}
    for arm in ARMS:
        torch.manual_seed(271828)
        head = Head(state.shape[1], arm, mean, scale).to(device)
        counts[arm] = sum(parameter.numel() for parameter in head.parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        losses = []
        for indices in batches:
            selected = torch.as_tensor(indices, device=device)
            x, y, w = state[selected], target[selected], weight[selected]
            optimizer.zero_grad(set_to_none=True)
            if arm == "wmse":
                loss = weighted_mse_loss(head(x), y, w)
            else:
                probability = head.log_probabilities(x)
                ce = -probability.gather(1, code[selected, None])[:, 0]
                ce_weight = w * y.clamp(-2, 2).abs().mean(-1) if "tilted" in arm else w
                loss = (ce * ce_weight).sum() / ce_weight.sum().clamp_min(1e-8)
                if "tilted" in arm:
                    prediction = rational_readout(probability, m1, m0, tilted=True)
                    loss = loss + weighted_mse_loss(prediction, y, w)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite head fitting loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        heads[arm] = head.eval()
        history[arm] = losses
    if min(counts.values()) < 0.98 * PARAMETER_BUDGET:
        raise ValueError("Integer head widths exceeded the prespecified2% capacity tolerance")
    return (
        heads,
        {"m1": m1, "m0": m0},
        {
            "updates": updates,
            "batch_rows": 512,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "parameter_budget": PARAMETER_BUDGET,
            "learned_parameters": counts,
            "batch_identity_sha256": hashlib.sha256(
                b"".join(batch.astype("<i8").tobytes() for batch in batches)
            ).hexdigest(),
            "support": support,
            "history": history,
        },
    )


def run(
    fit_path: Path,
    eval_path: Path,
    labels_path: Path,
    output: Path,
    device: str = "cpu",
    updates: int = 256,
    bootstrap: int = 2000,
) -> None:
    """Fit only one partition, then report every fixed arm on the opposite components."""
    if output.exists() or bootstrap < 100:
        raise ValueError("New output directory and at least100 bootstrap draws required")
    fit, fit_meta = load_cache(fit_path)
    evaluation, eval_meta = load_cache(eval_path)
    assert_outer_separation(fit, evaluation, fit_meta, eval_meta)
    if len(np.unique(fit["component"])) < 4:
        raise ValueError("At least four fitting components required")
    fit_classes = label_classes(fit, labels_path)
    heads, moments, training = train_heads(fit, fit_classes, device, updates)
    # No class/spread/branch labels are arguments to prediction.
    prediction = {"unchanged": evaluation["pred"]}
    state = torch.as_tensor(evaluation["state"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        for arm, head in heads.items():
            prediction[arm] = (
                torch.cat(
                    [
                        head.prediction(batch, moments["m1"], moments["m0"])
                        for batch in state.split(512)
                    ]
                )
                .cpu()
                .numpy()
            )
    affine = []
    for target in range(2):
        train = np.column_stack((np.ones(len(fit["row"])), fit["pred"][:, target]))
        test = np.column_stack((np.ones(len(evaluation["row"])), evaluation["pred"][:, target]))
        beta = fit_residual(train, fit["y"], fit["pred"], fit["fit_weight"], 0.01, "numpy")
        affine.append(evaluation["pred"][:, target] + test @ beta[:, target])
    prediction["global_affine"] = np.column_stack(affine)
    reports = compare(evaluation, prediction, bootstrap)
    secondary = dict(prediction)
    secondary["candidate"] = prediction["factorized_tilted_wmse"]
    secondary.pop("factorized_tilted_wmse")
    secondary["factorized_ce"] = prediction["candidate"]
    secondary_reports = compare(evaluation, secondary, bootstrap)
    # Only now load evaluation classes for descriptive strata, never for predictions.
    eval_classes = label_classes(evaluation, labels_path)
    class_scores, class_contrasts = {}, {}
    for label in range(4):
        selected = eval_classes == label
        if selected.any():
            class_scores[str(label)] = {
                arm: reference_global_wp(evaluation["y"], values, selected)
                for arm, values in prediction.items()
            }
            class_data = {key: evaluation[key][selected] for key in ("y", "sequence", "component")}
            component_count = len(np.unique(class_data["component"]))
            if component_count >= 4:
                class_contrasts[str(label)] = {
                    "factorized_ce": compare(
                        class_data,
                        {key: value[selected] for key, value in prediction.items()},
                        bootstrap,
                    ),
                    "factorized_tilted_wmse": compare(
                        class_data,
                        {key: value[selected] for key, value in secondary.items()},
                        bootstrap,
                    ),
                }
            else:
                class_contrasts[str(label)] = {
                    "status": "insufficient_components",
                    "components": component_count,
                }
    output.mkdir(parents=True)
    np.savez(
        output / "predictions.npz",
        **prediction,
        sequence=evaluation["sequence"],
        component=evaluation["component"],
        row=evaluation["row"],
    )
    torch.save(
        {
            "heads": {arm: head.cpu().state_dict() for arm, head in heads.items()},
            "moments": {key: value.cpu() for key, value in moments.items()},
            "arms": list(ARMS),
            "input_width": fit["state"].shape[1],
        },
        output / "heads.pt",
    )
    (output / "report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "fit_cache_sha256": fit_meta["sha256"],
                "eval_cache_sha256": eval_meta["sha256"],
                "label_cache_sha256": digest_file(labels_path),
                "training": training,
                "reports": reports,
                "factorized_tilted_wmse_reports": secondary_reports,
                "class_wp": class_scores,
                "class_contrasts": class_contrasts,
                "bootstrap": bootstrap,
                "candidate": "factorized plain CE with empirical rational readout",
                "factorization": "P(class|h)P(spread-code|h)P(branch|h)P(U-bin|branch,h)",
                "interpretation": "statistical code categories; no claim of true physical spread reconstruction",
                "row_weight_policy": "identical frozen-cache fit_weight for every arm; no new propensity fitting",
                "selection": "none; all fixed arms reported; no promotion without untouched/global-affine benefit and independent confirmation",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Run a separately admitted fixed-budget modern-state categorical readout screen."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.fit, args.evaluation, args.labels, args.output, args.device)


if __name__ == "__main__":
    main()
