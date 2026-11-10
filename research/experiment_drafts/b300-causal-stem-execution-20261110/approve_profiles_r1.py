"""Independently inspect all twelve native PR74 engineering receipts; start no fit."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path


def digest(raw: bytes) -> str:
    """Hash actual bytes separately from normalized recipe objects."""
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> str:
    """Use the trainer's deterministic JSON identity representation."""
    return digest(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def main() -> None:
    """Approve only bound engineering evidence, retaining the pending fit review."""
    root = Path(__file__).resolve().parent
    evidence = root / "native-profile-evidence-r1"
    authority = json.loads((root / "AUTHORITY_R2.json").read_bytes())
    split = json.loads((root / "chronological_primary.json").read_bytes())
    manifest = json.loads((root / "source-r1.manifest.json").read_bytes())
    configs = json.loads((evidence / "receipts/config-identities-r1.json").read_bytes())
    source_files = {
        k: v
        for k, v in manifest["files"].items()
        if (k.startswith("src/") and k.endswith(".py"))
        or k == "scripts/train_sequence.py"
    }
    assert canonical(source_files) == authority["source"]["source_sha256"], "source map"
    summary = json.loads(
        (evidence / "control/b300-driver-r1/profiles-complete.json").read_bytes()
    )
    assert (
        summary["status"] == "profiles_complete"
        and summary["scientific_fits_started"] == 0
    ), "phase completion"
    assert set(summary["cells"]) == set(authority["configs"]), "six cells"
    assert (evidence / "control/AUTHORITY_R2.json").read_bytes() == (
        root / "AUTHORITY_R2.json"
    ).read_bytes(), "authority bytes"
    fit = {("train", i) for i in split["fit"]["train"]}
    valid = {("valid", i) for i in split["evaluation"]["valid"]}
    assert len(fit) == 10325 and len(valid) == 1873 and not split["fit"]["valid"], (
        "official membership"
    )
    output = root / "approved-profiles-r1"
    output.mkdir(exist_ok=True)
    assert not list(output.iterdir()), "never overwrite prior approvals"
    profiles, stats = {}, []
    reviewed_at = datetime.datetime.now(datetime.UTC).isoformat()
    for cell, item in authority["configs"].items():
        config = configs[cell]["normalized"]
        assert configs[cell]["raw_sha256"] == item["raw_sha256"], cell + " raw config"
        assert canonical(config) == configs[cell]["config_sha256"], (
            cell + " normalized config"
        )
        assert (
            config["kd_weight"] == config["hint_weight"] == 0
            and config["objective_mode"] == "standalone_labels_wmse_v2"
        ), cell + " label objective"
        assert config["model"].get("input_stem_kernel", 0) == item["kernel"], (
            cell + " kernel"
        )
        production = None
        for stage in ("profile", "profile-peak"):
            receipt = json.loads(
                (evidence / "runs" / cell / stage / "profile_receipt.json").read_bytes()
            )
            identity, data = receipt["identity"], receipt["identity"]["data"]
            expected_config = dict(
                config, **({"warmup_fraction": 0.0} if stage == "profile-peak" else {})
            )
            assert (
                receipt["receipt_schema"] == "wnn.sequence-profile.v1"
                and receipt["status"] == "profile_complete"
                and receipt["approved"] is False
            ), cell + " status"
            assert (
                receipt["updates"] == 50
                and receipt["error_count"] == 0
                and receipt["errors"] == []
                and receipt["scientific_checkpoint_written"] is False
            ), cell + " work"
            assert (
                identity["source_sha256"] == authority["source"]["source_sha256"]
                and identity["source_files"] == source_files
                and identity["config_sha256"] == canonical(expected_config)
                and identity["world_size"] == 2
            ), cell + " source/config"
            assert (
                identity["data_sha256"] == canonical(data)
                and data["split_sha256"] == authority["split_sha256"]
            ), cell + " data identity"
            assert (
                len(data["fit_members"]) == 10325
                and {tuple(x) for x in data["fit_members"]} == fit
            ), cell + " exact fitting set"
            assert (
                len(data["evaluation_members"]) == 1873
                and {tuple(x) for x in data["evaluation_members"]} == valid
            ), cell + " exact evaluation set"
            assert (
                data["teacher"] is None
                and data.get("reference") is None
                and data.get("initializer") is None
                and data["hint_members"] in (None, [])
                and data["training_mask"] == "all_after_warmup"
            ), cell + " ancestry"
            norm = data["normalization"]
            assert (
                norm["scope"] == "all_fit"
                and norm["sequence_count"] == 10325
                and norm["moments_sha256"]
                == "be4935168c8b4000b5b92e26cef38c8ab44b6d8844469bf874efebf759b8031f"
                and set(norm["source_fingerprints"]) == {"datasets/train.parquet"}
            ), cell + " train-only normalization"
            runtime = identity["runtime"]
            assert (
                runtime["world_size"] == 2
                and runtime["torch"] == "2.14.0+cu130"
                and runtime["precision"]["mode"] == "bf16_bf16x9"
            ), cell + " runtime"
            assert [v["uuid"].removeprefix("GPU-") for v in runtime["devices"]] == item[
                "gpu_uuids"
            ] and all("B300" in v["name"] for v in runtime["devices"]), cell + " GPUs"
            assert (
                runtime["precision"]["parameter_dtype"]
                == runtime["precision"]["gradient_dtype"]
                == runtime["precision"]["optimizer_state_dtype"]
                == "float32"
            ), cell + " precision"
            assert (
                all(math.isfinite(float(v)) for v in receipt["losses"].values())
                and receipt["elapsed_seconds"] > 0
            ), cell + " finite loss"
            reload = receipt["engineering_reload"]
            assert reload["status"] == "passed" and [
                v["rank"] for v in reload["by_rank"]
            ] == [0, 1], cell + " reload ranks"
            assert all(
                v["model_optimizer_scheduler_rng_equal"] is True
                and v["temporary_engineering_state_removed"] is True
                and len(v["roundtrip_sha256"]) == 64
                for v in reload["by_rank"]
            ), cell + " full state"
            assert len(receipt["memory_by_rank"]) == 2 and all(
                v["peak_reserved_bytes"] < 0.9 * v["device_total_memory_bytes"]
                for v in receipt["memory_by_rank"]
            ), cell + " memory"
            stats.append(
                {
                    "cell": cell,
                    "stage": stage,
                    "seconds": receipt["elapsed_seconds"],
                    "reserved_bytes": max(
                        v["peak_reserved_bytes"] for v in receipt["memory_by_rank"]
                    ),
                    "rank_rss_bytes": max(
                        v["host_peak_rss_bytes"] for v in receipt["memory_by_rank"]
                    ),
                }
            )
            if stage == "profile":
                production = receipt
        assert production is not None
        approved = dict(
            production,
            approved=True,
            reviewer="ROOT independent native receipt/data/source/config/reload inspection",
            reviewed_at=reviewed_at,
            approval_note="Engineering only: exact train-only ancestry, config and production+peak50/both-rank full-state checks passed. Quality untested.",
        )
        path = output / (cell + ".json")
        path.write_bytes(
            (json.dumps(approved, sort_keys=True, indent=2) + "\n").encode()
        )
        profiles[cell] = {
            "approved_receipt_path": authority["native_root"]
            + "/runs/"
            + cell
            + "/approved_profile.json",
            "approved_receipt_sha256": digest(path.read_bytes()),
            "config_sha256": configs[cell]["config_sha256"],
            "raw_config_sha256": item["raw_sha256"],
            "pair": item["pair"],
            "gpu_uuids": item["gpu_uuids"],
        }
        for label, stage in (("production", "profile"), ("peak", "profile-peak")):
            profiles[cell][label + "_receipt_path"] = (
                authority["native_root"]
                + "/runs/"
                + cell
                + "/"
                + stage
                + "/profile_receipt.json"
            )
            profiles[cell][label + "_receipt_sha256"] = digest(
                (evidence / "runs" / cell / stage / "profile_receipt.json").read_bytes()
            )
    report = {
        "schema": "wnn.b300-pr74-ROOT-native-profile-review.v1",
        "status": "all12qualified",
        "profiles": profiles,
        "stats": stats,
        "source_sha256": authority["source"]["source_sha256"],
        "authority_sha256": digest((root / "AUTHORITY_R2.json").read_bytes()),
        "split_sha256": authority["split_sha256"],
        "quality_claim": None,
        "fit_admission_pending_R2_ownership_review": True,
        "reviewed_at": reviewed_at,
    }
    (root / "ROOT-profile-review-r1.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "profiles": len(stats),
                "engineering_updates": 600,
                "max_reserved_GiB": max(v["reserved_bytes"] for v in stats) / 2**30,
                "max_rank_RSS_GiB": max(v["rank_rss_bytes"] for v in stats) / 2**30,
                "fits_admitted": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
