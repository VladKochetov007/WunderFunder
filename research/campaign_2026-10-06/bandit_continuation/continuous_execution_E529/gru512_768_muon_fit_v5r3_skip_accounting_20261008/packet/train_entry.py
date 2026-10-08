"""Run one frozen trainer or predictor command with a verified input digest handoff."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from capacity_worker import validate_shared_input_binding
from verified_digest_handoff import FileIdentity, VerifiedDigestHandoff


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _install_input_digest_handoff(project_root: Path) -> VerifiedDigestHandoff:
    receipt_path = Path(os.environ["WNN_INPUT_IDENTITY_RECEIPT"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    sources = receipt.get("raw_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("raw input identity receipt must bind train and valid exactly")
    allowed_paths = tuple(Path(item["path"]) for item in sources)
    handoff = VerifiedDigestHandoff(allowed_paths)
    for item in sources:
        path = Path(item["path"])
        identity_payload = item["identity"]
        identity = FileIdentity(
            device=int(identity_payload["device"]),
            inode=int(identity_payload["inode"]),
            size=int(identity_payload["size"]),
            modified_ns=int(identity_payload["modified_ns"]),
            changed_ns=int(identity_payload["changed_ns"]),
            mode=int(identity_payload["mode"]),
        )
        candidate = handoff.candidate_from_prior_identity(
            path,
            prior_identity=identity,
            prior_sha256=str(item["sha256"]),
        )
        if candidate is not None:
            handoff.authorize(
                candidate,
                expected_sha256=str(item["sha256"]),
                registry_sha256=str(item["sha256"]),
            )

    import wnn.portability as portability

    handoff.install_portable_verifier(portability)
    if not project_root.is_dir():
        raise FileNotFoundError("frozen project root is unavailable")
    return handoff


def main() -> None:
    """Dispatch one child command while retaining exact input verifier behavior."""
    project_root = Path(os.environ["WNN_PROJECT_ROOT"]).resolve(strict=True)
    mode = os.environ["WNN_ENTRY_MODE"]
    if mode not in {"train", "predict"}:
        raise ValueError("unsupported frozen entry mode")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
    os.chdir(project_root)
    validate_shared_input_binding(
        project_root,
        Path(os.environ["WNN_INPUT_BINDING_PATH"]),
        os.environ["WNN_INPUT_BINDING_SHA256"],
    )
    handoff = _install_input_digest_handoff(project_root)
    args = sys.argv[1:]
    if mode == "train":
        print(
            'GRU_CAPACITY_CHILD_STAGE {"stage":"trainer_import_start"}',
            flush=True,
        )
        import train_rnn

        print(
            'GRU_CAPACITY_CHILD_STAGE {"stage":"trainer_import_done"}',
            flush=True,
        )
        sys.argv = [str(project_root / "scripts" / "train_rnn.py"), *args]
        try:
            print(
                'GRU_CAPACITY_CHILD_STAGE {"stage":"trainer_main_start"}',
                flush=True,
            )
            train_rnn.main()
        finally:
            _write_json(Path(os.environ["WNN_DIGEST_METRICS_PATH"]), handoff.metrics())
    else:
        print(
            'GRU_CAPACITY_CHILD_STAGE {"stage":"predictor_import_start"}',
            flush=True,
        )
        import predict_run

        print(
            'GRU_CAPACITY_CHILD_STAGE {"stage":"predictor_import_done"}',
            flush=True,
        )
        sys.argv = [str(project_root / "scripts" / "predict_run.py"), *args]
        try:
            print(
                'GRU_CAPACITY_CHILD_STAGE {"stage":"predictor_main_start"}',
                flush=True,
            )
            predict_run.main()
        finally:
            _write_json(Path(os.environ["WNN_DIGEST_METRICS_PATH"]), handoff.metrics())


if __name__ == "__main__":
    main()
