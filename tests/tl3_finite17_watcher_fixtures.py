from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.tl3_finite17_dashboard_core import _canonical_sha256, _sha256_file


class FakeGitHub:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object]] = []
        self.comments: list[dict[str, str]] = []
        self.account_checks = 0

    def verify_account(self, _deadline: float) -> None:
        self.account_checks += 1

    def request(self, method: str, path: str, data: object, _deadline: float) -> object:
        self.requests.append((method, path, data))
        if method == "GET":
            return self.comments
        url = f"https://github.com/VladKochetov007/WunderFunder/pull/123#issuecomment-{len(self.comments) + 1}"
        self.comments.append({"body": str(data["body"]), "html_url": url})
        return {"html_url": url}


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], list[int], list[int]]:
    root = tmp_path / "flat"
    state_dir = root / "finite17-state"
    root.mkdir()
    state_dir.mkdir()
    train = list(range(10_325))
    # Row-group IDs are local to each source, so both source manifests may start at zero.
    valid = list(range(1_873))
    split_path = root / "split_manifest.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "wnn.chronological-primary.v1",
                "fit": {"train": train, "valid": []},
                "evaluation": {"valid": valid},
            }
        ),
        encoding="utf-8",
    )
    split_sha = _sha256_file(split_path)
    source_sha = "a" * 64
    names = [
        "fin256_s3",
        "fmu128_s3",
        "frawW_s3",
        "fin256_s4",
        "fmu128_s4",
        "frawW_s4",
        "orgbase",
        "fin256ft_s3",
        "ffull_s3",
        "frawW_s3ft",
        "fin256ft_s4",
        "ffull_s4",
        "frawW_s4ft",
        "fbc1k_s5",
        "fttall_tlead_s5",
        "fttall_tlead_s6",
        "w18final_distilled_s5",
    ]
    target_id = "fin256_s4"
    run = f"tl3c_{target_id}"
    args = [
        "--name",
        run,
        "--train-source",
        "train",
        "--train-seqs",
        "10325",
        "--train-eval-seqs",
        "32",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "4",
        "--eval-source",
        "dev",
        "--checkpoint-epoch",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--precision",
        "fp32",
        "--full-valid",
    ]
    nodes = []
    for name in names:
        node_run = f"tl3c_{name}"
        node_args = args if name == target_id else ["--name", node_run]
        nodes.append(
            {
                "id": name,
                "run": node_run,
                "args": node_args,
                "args_sha256": _canonical_sha256(node_args),
                "expected_updates": 12_880 if name == target_id else 1,
                "epochs": 4 if name == target_id else 1,
                "endpoint": 3 if name == target_id else 0,
                "run_directory": str(root / "artifacts" / "runs" / node_run),
                "parents": [],
            }
        )
    plan: dict[str, Any] = {
        "schema": "wnn.tl3-finite17-plan.v1",
        "root": str(root.resolve()),
        "state_dir": str(state_dir.resolve()),
        "split": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": split_sha,
            "fit_train_count": 10_325,
            "fit_valid_count": 0,
            "evaluation_valid_count": 1_873,
        },
        "source_identity": {"training_source_sha256": source_sha},
        "nodes": nodes,
    }
    plan["sha256"] = _canonical_sha256(plan)
    prefix = sorted(train)[:32]
    member_sha = hashlib.sha256(
        json.dumps([["train", group] for group in prefix], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    checkpoint_paths = ["model.pt", "model_ep3.pt"]
    run_dir = root / "artifacts" / "runs" / run
    run_dir.mkdir(parents=True)
    for name in checkpoint_paths:
        (run_dir / name).write_bytes(f"test checkpoint {name}".encode())
    hashes = {name: _sha256_file(run_dir / name) for name in checkpoint_paths}
    actual_args = {
        "name": run,
        "train_source": "train",
        "train_seqs": 10_325,
        "train_eval_seqs": 32,
        "batch": 32,
        "chunk": 2_000,
        "epochs": 4,
        "eval_source": "vfold",
        "valid_fold": 0,
        "n_valid_folds": 1,
        "checkpoint_epoch": 3,
        "checkpoint_policy": "fixed",
        "precision": "fp32",
        "full_valid": True,
        "no_eval_during_fit": False,
    }
    provenance = {
        "selected_epoch": 3,
        "split_manifest_sha256": split_sha,
        "trainer_implementation": {"combined_sha256": source_sha},
        "fit_row_groups": {"train": train, "valid": []},
        "evaluation_row_groups": valid,
        "args": actual_args,
        "checkpoint_sha256": hashes,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    history = []
    for epoch in range(4):
        history.append(
            {
                "epoch": epoch,
                "split": "vfold1873",
                "selected_rows": 3_528_273,
                "weighted_pearson": 0.2 + epoch * 0.01,
                "global_t0": 0.15 + epoch * 0.01,
                "global_t1": 0.25 + epoch * 0.01,
                "valid_wmse": 0.9 - epoch * 0.01,
                "train_eval_source": "train",
                "train_eval_mask": "all_after_warmup",
                "train_eval_sequences": 32,
                "train_eval_row_groups": prefix,
                "train_eval_members_sha256": member_sha,
                "train_eval_selected_rows": 640_000,
                "train_eval_wp": 0.4 + epoch * 0.01,
                "train_eval_wp_t0": 0.35 + epoch * 0.01,
                "train_eval_wp_t1": 0.45 + epoch * 0.01,
                "train_eval_wmse": 0.7,
                "fit_summary": {
                    "optimizer_updates": {
                        "epoch": {
                            "optimizer_attempts": 3_220,
                            "optimizer_completed_updates": 3_220,
                            "amp_skipped_updates": 0,
                        }
                    }
                },
            }
        )
    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    verified = {
        "run": run,
        "epoch": 3,
        "model_sha256": hashes["model.pt"],
        "endpoint_sha256": hashes["model_ep3.pt"],
        "provenance_sha256": _sha256_file(run_dir / "provenance.json"),
        "history_sha256": _sha256_file(history_path),
        "source_sha256": source_sha,
        "expected_updates": 12_880,
        "split_manifest_sha256": split_sha,
    }
    node_states = {
        name: {
            "state": "complete" if name == target_id else "blocked",
            "phase": "fit_complete" if name == target_id else "dependency_failed",
            "verified_run": verified if name == target_id else None,
        }
        for name in names
    }
    node_states["fin256_s3"]["state"] = "failed"
    node_states["fin256_s3"]["phase"] = "fit_failed"
    status: dict[str, Any] = {
        "schema": "wnn.tl3-finite17-state.v1",
        "status": "finished_with_failures",
        "plan_sha256": plan["sha256"],
        "fits_used": 2,
        "nodes": node_states,
        "tasks": {},
        "score": {"state": "blocked"},
    }
    (state_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    status_path = state_dir / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    terminal = {
        "schema": "wnn.tl3-finite17-terminal.v1",
        "status": status["status"],
        "plan_sha256": plan["sha256"],
        "state_sha256": _sha256_file(status_path),
        "fits_used": 2,
        "score": status["score"],
    }
    (state_dir / "terminal.json").write_text(json.dumps(terminal), encoding="utf-8")
    return root, state_dir, plan, status, train, valid
