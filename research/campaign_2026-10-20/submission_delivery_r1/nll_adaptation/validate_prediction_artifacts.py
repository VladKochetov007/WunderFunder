"""Run the frozen prediction preflight or post-export audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prediction_artifact_validation import _absent, _base, _json, _sha
from prediction_export_audit import _export


def main() -> None:
    """Run preflight or post-export audit and write a create-only receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "post-export"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--post-receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "post-export" and args.preflight_receipt is None:
        parser.error("--preflight-receipt is required for post-export")
    if args.phase == "preflight" and args.post_receipt is None:
        parser.error("--post-receipt is required for preflight")
    if args.phase == "post-export" and args.post_receipt is not None:
        parser.error("--post-receipt is only valid for preflight")
    receipt = args.receipt.resolve()
    try:
        if receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {receipt}")
        plan_path, eval_path = args.plan.resolve(), args.evaluation_manifest.resolve()
        plan, _, split, components, base = _base(plan_path, eval_path)
        audit_dir = Path(__file__).parent
        validator_files = [
            Path(__file__),
            audit_dir / "prediction_artifact_validation.py",
            audit_dir / "prediction_export_audit.py",
        ]
        validator_identity = {str(path.resolve()): _sha(path) for path in validator_files}
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "phase": args.phase,
            "created_utc": datetime.now(UTC).isoformat(),
            "validator_modules_sha256": validator_identity,
            "plan": {"path": str(plan_path), "sha256": _sha(plan_path)},
            "evaluation_manifest": {"path": str(eval_path), "sha256": _sha(eval_path)},
            **base,
        }
        if args.phase == "preflight":
            payload["new_output_paths_absent"] = _absent(plan, receipt, args.post_receipt.resolve())
            root = Path(plan["source_pins"]["project_root"])
            payload["completed_checkpoints"] = [
                _export(
                    root,
                    item,
                    split,
                    components,
                    plan["source_pins"]["split_manifest"]["sha256"],
                    check_predictions=False,
                )
                for item in plan["endpoints"]
                if item.get("argv")
            ]
        else:
            pre_path = args.preflight_receipt.resolve()
            pre = _json(pre_path)
            if (
                pre.get("status") != "PASS"
                or pre.get("phase") != "preflight"
                or pre.get("plan", {}).get("sha256") != payload["plan"]["sha256"]
                or pre.get("evaluation_manifest", {}).get("sha256")
                != payload["evaluation_manifest"]["sha256"]
                or not pre.get("new_output_paths_absent")
                or pre.get("validator_modules_sha256") != validator_identity
            ):
                raise ValueError(
                    "preflight receipt does not bind this plan and evaluation manifest"
                )
            payload["preflight_receipt"] = {"path": str(pre_path), "sha256": _sha(pre_path)}
            root = Path(plan["source_pins"]["project_root"])
            prior = {row["id"]: row for row in pre.get("completed_checkpoints", [])}
            exports = []
            for item in plan["endpoints"]:
                if not item.get("argv"):
                    continue
                current = _export(
                    root, item, split, components, plan["source_pins"]["split_manifest"]["sha256"]
                )
                earlier = prior.get(item["id"], {})
                if any(
                    current.get(field, {}).get("sha256") != earlier.get(field, {}).get("sha256")
                    for field in ("checkpoint", "provenance")
                ):
                    raise ValueError(
                        f"{item['id']} checkpoint/provenance changed since pre-export binding"
                    )
                exports.append(current)
            payload["new_exports"] = exports
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        print(
            json.dumps({"status": "FAIL", "phase": args.phase, "error": str(exc)}), file=sys.stderr
        )
        raise SystemExit(2) from exc
    print(json.dumps({"status": "PASS", "phase": args.phase, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
