"""Add receipt-bound fresh full-primary refits to the existing native snapshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from multistage_github_common import canonical_json, is_sha256, safe_run_id, sha256_bytes

SOURCE = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
PRIMARY = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
LOG = logging.getLogger("multistage_github_primary")


def _record(path: Path) -> tuple[dict[str, object], str]:
    from multistage_github_source import MAX_METADATA_BYTES, _read_regular

    raw = _read_regular(path, MAX_METADATA_BYTES)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("primary-refit receipt must be a JSON object")
    return value, sha256_bytes(raw)


def _members(value: object, source: str, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(row, list)
            and len(row) == 2
            and row[0] == source
            and type(row[1]) is int
            and row[1] >= 0
            for row in value
        )
        and len({row[1] for row in value}) == count
    )


def validate_primary_snapshot(run: dict[str, object]) -> None:
    """Reject auxiliary/control data, transferred ancestry and relabelled metrics."""
    from multistage_github_common import _valid_event

    metadata = run["metadata"]
    config = metadata.get("config", {})
    data = metadata.get("provenance", {}).get("data", {})
    binding = run.get("primary_refit_binding", {})
    approval = binding.get("root_approval", {})
    if (
        run.get("stage") != 4
        or run.get("phase") != "primary_refit"
        or run.get("stage_evidence") != "approved_primary_refit"
        or run.get("source_sha256") != SOURCE
        or run.get("split_sha256") != PRIMARY
        or run.get("primary_split_sha256") != PRIMARY
        or metadata.get("source", {}).get("sha256") != SOURCE
        or data.get("split_sha256") != PRIMARY
        or data.get("primary_split_sha256") not in (None, PRIMARY)
        or data.get("evaluation_role") != "valid"
        or data.get("evaluation_source") != "valid"
        or not _members(data.get("fit_members"), "train", 10325)
        or not _members(data.get("evaluation_members"), "valid", 1873)
        or data.get("teacher") is not None
        or data.get("initializer") is not None
        or config.get("kd_weight", 0) != 0
        or config.get("hint_weight", 0) != 0
        or config.get("model", {}).get("hint_dim", 0) != 0
        or config.get("model", {}).get("architecture") != run.get("family")
        or sha256_bytes(canonical_json(config)) != run.get("config_sha256")
        or sha256_bytes(canonical_json(metadata)) != run.get("canonical_metadata_sha256")
        or binding.get("schema") != "wnn.primary-refit-monitor-binding.v1"
        or approval.get("schema") != "wnn.primary-refit-root-native-approval.v1"
        or approval.get("status") != "approved"
        or approval.get("approved") is not True
        or approval.get("approver_id") != "root"
        or approval.get("native_fit_authorized") is not True
        or approval.get("scientific_candidate_promotion") is not False
        or approval.get("primary_valid_used_for_selection") is not False
        or approval.get("source_sha256") != SOURCE
        or approval.get("primary_split_sha256") != PRIMARY
        or approval.get("fit_id") != binding.get("fit_id")
        or approval.get("request_sha256") != binding.get("request_sha256")
        or sha256_bytes(canonical_json(approval)) != binding.get("root_approval_canonical_sha256")
        or any(
            not is_sha256(binding.get(key))
            for key in (
                "selection_receipt_sha256",
                "request_sha256",
                "fit_request_sha256",
                "root_approval_sha256",
                "approved_profile_sha256",
            )
        )
    ):
        raise ValueError(
            "primary-refit metadata or approval does not bind fresh full-primary fitting"
        )
    normalization = data.get("normalization", {})
    if set(normalization.get("source_fingerprints", {})) != {"datasets/train.parquet"}:
        raise ValueError("primary-refit normalization ancestry is not official-train only")
    for name in ("event_journal", "evaluation_journal"):
        journal = run.get(name, {})
        events = journal.get("events")
        if (
            not is_sha256(journal.get("source_sha256"))
            or not isinstance(events, list)
            or journal.get("retained_event_count") != len(events)
            or any(
                type(journal.get(key)) is not int or journal[key] < 0
                for key in (
                    "source_bytes",
                    "source_complete_lines",
                    "source_partial_tail_bytes",
                )
            )
        ):
            raise ValueError("primary-refit journal source identity is incomplete")
        for event in events:
            if not _valid_event(event):
                raise ValueError("primary-refit journal contains a non-native event")
            if name == "evaluation_journal" and event["kind"] != "evaluation":
                raise ValueError("primary-refit evaluation journal has a non-evaluation event")
            if any(key == "control" or key.startswith("control/") for key in event["metrics"]):
                raise ValueError("inner-control metrics cannot masquerade as primary validation")


def _primary_run(
    prepared: Path, runtime: Path, selection_sha: str, selected: dict[str, object]
) -> dict[str, object]:
    from multistage_github_source import _journal, _preserve_metadata

    fit_id = selected.get("fit_id")
    if not safe_run_id(fit_id) or not str(fit_id).startswith("primary-refit-"):
        raise ValueError("prepared primary fit identifier is unsafe")
    request_path = prepared / "requests" / f"{fit_id}.json"
    request, request_sha = _record(request_path)
    if (
        selected.get("request_path") != str(request_path.resolve())
        or selected.get("request_sha256") != request_sha
    ):
        raise ValueError("primary request differs from frozen selection")
    attempt_id = f"stage4-primary-refit-{fit_id}"
    fit, fit_sha = _record(runtime / "requests" / f"{attempt_id}.json")
    owner, _ = _record(runtime / "receipts" / f"{attempt_id}.owner.json")
    approved_path = Path(str(request["profile"]["output"])) / "approved_profile_receipt.json"
    approved, approved_sha = _record(approved_path)
    approval_ref = approved.get("root_native_approval", {})
    approval, approval_sha = _record(Path(str(approval_ref.get("path"))))
    output = Path(str(request["fit"]["output"]))
    if (
        output.resolve() != (runtime / "runs" / str(fit_id)).resolve()
        or output.is_symlink()
        or approved_path.resolve()
        != (runtime / "profiles" / f"{fit_id}-profile" / "approved_profile_receipt.json").resolve()
        or request.get("schema") != "wnn.multistage-primary-refit-request.v1"
        or request.get("fit_id") != fit_id
        or request.get("training_source", {}).get("sha256") != SOURCE
        or request.get("primary_split", {}).get("sha256") != PRIMARY
        or request.get("primary_split", {}).get("fit_train_sequences") != 10325
        or request.get("primary_split", {}).get("fit_valid_sequences") != 0
        or request.get("primary_split", {}).get("evaluation_sequences") != 1873
        or request["fit"].get("resume") is not False
        or request["fit"].get("initializer") is not None
        or request["fit"].get("teacher_or_cache") is not None
        or request["fit"].get("transferred_auxiliary_weights") is not False
        or fit.get("stage") != 4
        or fit.get("fit_attempt_id") != attempt_id
        or fit.get("source_sha256") != SOURCE
        or fit.get("primary_split_sha256") != PRIMARY
        or fit.get("selection_receipt_sha256") != selection_sha
        or fit.get("config_sha256") != request.get("config", {}).get("sha256")
        or fit.get("command") != request["fit"].get("argv_after_approval_and_reservation")
        or fit.get("output_dir") != str(output.resolve())
        or owner.get("fit_id") != fit_id
        or owner.get("stage") != 4
        or owner.get("budget_attempt_id") != attempt_id
        or owner.get("approval_sha256") != approval_sha
        or fit.get("root_native_approval_sha256") != approval_sha
        or approval_ref.get("sha256") != approval_sha
        or approval.get("request_sha256") != request_sha
        or approved.get("approved") is not True
        or approved.get("status") != "profile_complete"
    ):
        raise ValueError(
            "primary fit is not bound to prepared selection, owned launch and root approval"
        )
    metadata, metadata_sha = _record(output / "run_metadata.json")
    config = metadata.get("config", {})
    config_sha = sha256_bytes(canonical_json(config))
    if config_sha != request.get("config", {}).get("normalized_identity_sha256"):
        raise ValueError("native primary config differs from the selected fresh recipe")
    identity = approved.get("identity", {})
    provenance = metadata.get("provenance", {})
    if (
        identity.get("config_sha256") != config_sha
        or identity.get("source_sha256") != SOURCE
        or metadata.get("source", {}).get("files") != identity.get("source_files")
        or any(
            provenance.get(key) != identity.get(key)
            for key in (
                "data",
                "data_sha256",
                "runtime",
                "world_size",
                "model_parameters",
            )
        )
        or approval.get("config_sha256") != fit.get("config_sha256")
        or approval.get("native_profile_sha256") != fit.get("profile_sha256")
        or owner.get("profile_sha256") != fit.get("profile_sha256")
        or approval.get("engineering_review_sha256") != fit.get("engineering_review_sha256")
    ):
        raise ValueError("native primary identity differs from its approved exact profile")
    run = {
        "run_id": f"s4-{fit_id}",
        "stage": 4,
        "phase": "primary_refit",
        "stage_evidence": "approved_primary_refit",
        "family": config.get("model", {}).get("architecture"),
        "source_sha256": SOURCE,
        "split_sha256": PRIMARY,
        "primary_split_sha256": PRIMARY,
        "config_sha256": config_sha,
        "metadata_sha256": metadata_sha,
        "canonical_metadata_sha256": sha256_bytes(canonical_json(metadata)),
        "metadata": _preserve_metadata(metadata),
        "primary_refit_binding": {
            "schema": "wnn.primary-refit-monitor-binding.v1",
            "fit_id": fit_id,
            "selection_receipt_sha256": selection_sha,
            "request_sha256": request_sha,
            "fit_request_sha256": fit_sha,
            "approved_profile_sha256": approved_sha,
            "root_approval_sha256": approval_sha,
            "root_approval": approval,
            "root_approval_canonical_sha256": sha256_bytes(canonical_json(approval)),
        },
        "event_journal": _journal(output / "monitor_events.jsonl", evaluations_only=False),
        "evaluation_journal": _journal(output / "monitor_evaluations.jsonl", evaluations_only=True),
    }
    validate_primary_snapshot(run)
    return run


def collect_primary_refits(prepared: Path, runtime: Path) -> list[dict[str, object]]:
    """Read at most four prepared owned fits directly; do not create mirror entries."""
    if not prepared.exists():
        return []
    selection, selection_sha = _record(prepared / "selection_receipt.json")
    selected = selection.get("selected")
    if (
        selection.get("schema") != "wnn.multistage-primary-refit-selection.v1"
        or selection.get("selection_used_primary_valid") is not False
        or selection.get("primary_split_sha256") != PRIMARY
        or not isinstance(selected, list)
        or not 2 <= len(selected) <= 4
        or not all(isinstance(row, dict) for row in selected)
    ):
        raise ValueError("primary monitoring needs the exact <=4 prepared inner-selected requests")
    runs = []
    for row in selected:
        try:
            runs.append(_primary_run(prepared, runtime, selection_sha, row))
        except FileNotFoundError:
            continue
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            LOG.warning(
                "omitting invalid primary refit fit_id=%s reason=%s", row.get("fit_id"), error
            )
    return runs
