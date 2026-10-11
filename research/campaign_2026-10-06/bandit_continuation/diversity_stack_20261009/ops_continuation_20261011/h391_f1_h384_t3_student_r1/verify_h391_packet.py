#!/usr/bin/env python3
"""Verify the frozen H391 source, recipe, and captured real producer receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
RECEIPTS = "actual_producer_receipts_r1"
EXPECTED_RUNS = {
    "vt_f1_muonall_h384_e3": {
        "seed": 3,
        "provenance": "teacher_s3.provenance.json",
        "provenance_sha256": "ca9f7c32910e5d71acddf78a474c06eee324b54d781ad9c736cd1f9a4a83deae",
        "checkpoint_sha256": "9c7d5e55fde7112dc691c80670fb17b62f24b9d759d2c90312a9ec89334843c6",
    },
    "vt_f1_muonall_h384_e3_s4": {
        "seed": 4,
        "provenance": "parent_teacher_s4.provenance.json",
        "provenance_sha256": "e11ca9b3bfb9fc302f199514eb075960933e8fd27dfa49d58a31b76d5f5011f6",
        "checkpoint_sha256": "752f462036a8e340db00de98dcb141795f3a0ae54cfffce3e95ce7ceed7f5e9e",
    },
    "vt_f1_muonall_h384_e3_s5": {
        "seed": 5,
        "provenance": "teacher_s5.provenance.json",
        "provenance_sha256": "45555945babdc6af0d56e1b5ab6a68585383bf32a73335fe8b92d84af7bdf03d",
        "checkpoint_sha256": "ee154f035720069ef26bf0fd328c61887cdbb426c49d51fb25172e5bb52b1f80",
    },
}
EXPECTED_CACHE_FILES = {
    "main_homogeneous_train": {
        "sha256": "11c2b7cc26168037d4eefe49060741c58c865653d7244df0a5308d1584f8070c",
        "bytes": 1_685_280_128,
        "shape": [10_533, 20_000, 2],
        "row_groups_sha256": "0100ea21a62be8d32f1ffecbcc0cb64ea5f681041f9fc46874c5d0a33859761d",
    },
    "main_homogeneous_valid": {
        "sha256": "8acd9443f57b57e1c4944542e28718f4592cfc89c9029461379cd2f4f255ce09",
        "bytes": 226_560_128,
        "shape": [1_416, 20_000, 2],
        "row_groups_sha256": "087c48ccdd86649d56923b0755a9f909d6ee5270158f2027fc38f0ddae81c6a8",
    },
}
EXPECTED_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
EXPECTED_CACHE_SHA256 = "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779"
EXPECTED_PORTABLE_SHA256 = "551905f3363c65608e2d5a0f49e8eeca362f574ab6d98d443e31cee0790fb5fd"
EXPECTED_PROPENSITY_PROVENANCE_SHA256 = (
    "50e442a1f2d075b88cb9e50196dc57b81bd8d586a51fa0df3149a2c5d2e8cbf7"
)
EXPECTED_PROPENSITY_CHECKPOINT_SHA256 = (
    "1fbf412df8823c703fdf79263830f341269a68e769bc4a7258d93cdf8e40dc5c"
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON with a useful path in any parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON receipt {path}: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _row_group_digest(groups: list[int]) -> str:
    payload = struct.pack("<" + "q" * len(groups), *groups)
    return hashlib.sha256(payload).hexdigest()


def _validate_cache_contract(
    cache: dict[str, Any], split: dict[str, Any], teachers: dict[str, dict[str, Any]]
) -> None:
    _require(cache.get("format") == "wnn.heterogeneous-distillation-cache", "wrong cache format")
    _require(cache.get("version") == 1 and cache.get("fold") == 1, "wrong cache version/fold")
    _require(
        cache.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256,
        "cache split does not match fold 1",
    )
    _require(
        cache.get("homogeneous_family") == "strong",
        "H391 requires the declared strong homogeneous target",
    )
    _require(
        abs(float(cache.get("coefficient_sum", -1)) - 0.8125) <= 1e-8,
        "cache coefficient mass drift",
    )
    _require(
        cache.get("evaluation_valid_groups") == split["evaluation"]["valid"],
        "cache evaluation membership differs from split",
    )
    _require(
        cache.get("fit_groups", {}).get("train") == split["fit"]["train"],
        "cache train membership differs from split",
    )
    _require(
        cache.get("fit_groups", {}).get("valid") == split["fit"]["valid"],
        "cache valid membership differs from split",
    )

    strong_coefficients = {
        row["run"]: float(row["coefficient"])
        for row in cache.get("source_coefficients", [])
        if row.get("family") == "strong"
    }
    _require(set(strong_coefficients) == set(EXPECTED_RUNS), "cache strong-teacher set drift")
    _require(
        all(abs(value - 0.1) <= 1e-8 for value in strong_coefficients.values()),
        "strong source coefficients drift",
    )
    metadata_sources = {row.get("run"): row for row in cache.get("sources", [])}
    _require(
        set(EXPECTED_RUNS).issubset(metadata_sources),
        "cache source ancestry omits a strong teacher",
    )
    for run, expected in EXPECTED_RUNS.items():
        source = metadata_sources[run]
        _require(
            source.get("family") == "strong" and source.get("epoch") == 2,
            f"cache teacher epoch/family drift: {run}",
        )
        _require(
            source.get("checkpoint_sha256") == expected["checkpoint_sha256"],
            f"cache teacher checkpoint drift: {run}",
        )
        producer = teachers[run]
        _require(
            producer["checkpoint_sha256"].get("model_ep2.pt") == source.get("checkpoint_sha256"),
            f"teacher receipt and cache disagree: {run}",
        )
        _require(
            source.get("fit_group_counts") == {"train": 10_533, "valid": 1_416},
            f"cache teacher membership drift: {run}",
        )

    files = cache.get("files", {})
    for key, expected in EXPECTED_CACHE_FILES.items():
        entry = files.get(key)
        _require(isinstance(entry, dict), f"missing cache array {key}")
        for field in ("sha256", "bytes", "shape", "row_groups_sha256"):
            _require(entry.get(field) == expected[field], f"cache {key} {field} drift")
        _require(entry.get("dtype") == "float32", f"cache {key} dtype drift")
    _require(
        len(files) == 8,
        "expected the original full cleanfold cache metadata; do not substitute another cache",
    )


def _validate_teacher_receipts(packet: Path, split: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipts = packet / RECEIPTS
    parsed: dict[str, dict[str, Any]] = {}
    expected_train = set(split["fit"]["train"])
    expected_valid = set(split["fit"]["valid"])
    expected_eval = set(split["evaluation"]["valid"])
    _require(
        len(expected_train) == 10_533
        and len(expected_valid) == 1_416
        and len(expected_eval) == 457,
        "fold-1 population count drift",
    )
    _require(
        not expected_valid.intersection(expected_eval),
        "evaluation group leaked into fitted valid pool",
    )

    for run, expected in EXPECTED_RUNS.items():
        path = receipts / expected["provenance"]
        _require(
            sha256_file(path) == expected["provenance_sha256"],
            f"actual teacher provenance hash mismatch: {run}",
        )
        producer = load_json(path)
        args = producer.get("args", {})
        _require(
            args.get("name") == run and args.get("seed") == expected["seed"],
            f"teacher run/seed mismatch: {run}",
        )
        _require(
            args.get("hidden") == 384 and args.get("layers") == 2,
            f"teacher architecture mismatch: {run}",
        )
        _require(
            args.get("epochs") == 3 and args.get("checkpoint_epoch") == 2,
            f"teacher epoch mismatch: {run}",
        )
        _require(
            args.get("train_source") == "train+valid" and args.get("valid_fold") == 1,
            f"teacher split mismatch: {run}",
        )
        _require(
            args.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256,
            f"teacher split hash mismatch: {run}",
        )
        _require(
            args.get("since_weight") == 32.0, f"teacher since weighting ancestry mismatch: {run}"
        )
        _require(
            producer.get("checkpoint_sha256", {}).get("model_ep2.pt")
            == expected["checkpoint_sha256"],
            f"teacher checkpoint hash mismatch: {run}",
        )
        _require(
            set(producer.get("fit_row_groups", {}).get("train", [])) == expected_train,
            f"teacher train groups mismatch: {run}",
        )
        _require(
            set(producer.get("fit_row_groups", {}).get("valid", [])) == expected_valid,
            f"teacher fit-valid groups mismatch: {run}",
        )
        _require(
            set(producer.get("evaluation_row_groups", [])) == expected_eval,
            f"teacher eval groups mismatch: {run}",
        )
        parsed[run] = producer
    _require(
        len({row["checkpoint_sha256"]["model_ep2.pt"] for row in parsed.values()}) == 3,
        "teacher weights are not independent",
    )
    return parsed


def _validate_recipe(packet: Path) -> None:
    recipe = load_json(packet / "RECIPE_R1.json")
    _require(recipe.get("experiment_id") == "H391", "wrong experiment ID")
    _require(recipe.get("run_id") == "vt_f1_h391_dsq_h384_t3_a9_e6_s3", "wrong run ID")
    args = recipe.get("recipe", {})
    _require(args.get("since_weight") == 0.0, "H391 student since_weight must remain zero")
    _require(
        args.get("aux_struct") == 0.3 and args.get("aux_next") == 0.3,
        "H391 auxiliary schedule drift",
    )
    _require(
        args.get("hidden") == 384 and args.get("epochs") == 6 and args.get("checkpoint_epoch") == 5,
        "H391 width/horizon drift",
    )
    _require(
        args.get("distill_alpha") == 0.9 and args.get("distill_cache_variant") == "homogeneous",
        "H391 target recipe drift",
    )
    _require(
        args.get("include_heldout_valid") is False and args.get("no_eval_during_fit") is True,
        "fit/evaluation policy drift",
    )
    _require(args.get("init") == "vt_f1_muonall_h384_e3_s4@ep2", "H391 parent drift")
    _require(
        recipe.get("data", {}).get("heldout_claim") is False,
        "H391 cannot claim a fresh heldout result",
    )
    _require(
        recipe.get("target", {}).get("cache_metadata_sha256") == EXPECTED_CACHE_SHA256,
        "H391 cache metadata pin drift",
    )
    _require(
        recipe.get("data", {}).get("split_sha256") == EXPECTED_SPLIT_SHA256, "H391 split pin drift"
    )
    _require(
        recipe.get("data", {}).get("portable_manifest_sha256") == EXPECTED_PORTABLE_SHA256,
        "portable data identity drift",
    )
    argv = recipe.get("trainer_argv", [])

    def option_value(name: str) -> str | None:
        try:
            return argv[argv.index(name) + 1]
        except (ValueError, IndexError):
            return None

    _require(option_value("--since-weight") == "0", "actual H391 argv does not set since-weight=0")
    _require(option_value("--init") == args.get("init"), "actual H391 argv parent drift")
    _require(option_value("--distill-alpha") == "0.9", "actual H391 argv alpha drift")
    _require(
        "--no-eval-during-fit" in argv and "--save-training-state" in argv,
        "actual H391 argv missing fit controls",
    )
    propensity_path = packet / RECEIPTS / "propensity.provenance.json"
    _require(
        sha256_file(propensity_path) == EXPECTED_PROPENSITY_PROVENANCE_SHA256,
        "actual propensity provenance hash mismatch",
    )
    propensity = load_json(propensity_path)
    _require(
        propensity.get("checkpoint_sha256", {}).get("model.pt")
        == EXPECTED_PROPENSITY_CHECKPOINT_SHA256,
        "actual propensity checkpoint hash mismatch",
    )
    _require(
        recipe.get("propensity", {}).get("checkpoint_sha256")
        == EXPECTED_PROPENSITY_CHECKPOINT_SHA256,
        "H391 propensity pin drift",
    )


def _packet_files(packet: Path) -> dict[str, str]:
    return {
        path.relative_to(packet).as_posix(): sha256_file(path)
        for path in sorted(packet.rglob("*"))
        if path.is_file()
        and path.name != "PACKET_MANIFEST_R1.json"
        and "__pycache__" not in path.parts
    }


def write_packet_manifest(packet: Path) -> dict[str, Any]:
    manifest = {"schema": "wnn.h391-frozen-packet.v1", "files": _packet_files(packet)}
    (packet / "PACKET_MANIFEST_R1.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_packet(packet: Path = PACKET, require_packet_manifest: bool = True) -> dict[str, Any]:
    source_manifest = load_json(packet / "SOURCE_MANIFEST_H391_R1.json")
    source_root = packet / "child_source_v1"
    files = source_manifest.get("files", {})
    _require(len(files) == 58, "H391 source manifest must bind the exact 58-file baseline snapshot")
    observed_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    _require(observed_paths == set(files), "H391 child source contains missing or extra files")
    for rel, expected in files.items():
        _require(
            sha256_file(source_root / rel) == expected, f"H391 source file hash mismatch: {rel}"
        )
    _require(
        source_manifest.get("trainer_sha256") == files.get("scripts/train_rnn.py"),
        "source trainer pin mismatch",
    )
    _require(
        source_manifest.get("origin_manifest_sha256")
        == "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc",
        "source ancestry pin mismatch",
    )
    _require(
        sha256_file(packet / "SOURCE_MANIFEST_ORIGIN.json")
        == source_manifest["origin_manifest_sha256"],
        "origin manifest byte hash mismatch",
    )

    receipts = packet / RECEIPTS
    _require(
        sha256_file(receipts / "dc_98411230_metadata.json") == EXPECTED_CACHE_SHA256,
        "actual cache metadata hash mismatch",
    )
    _require(
        sha256_file(receipts / "fold1.json") == EXPECTED_SPLIT_SHA256,
        "actual split manifest hash mismatch",
    )
    _require(
        sha256_file(receipts / "portable_inputs_fold1.json") == EXPECTED_PORTABLE_SHA256,
        "actual portable manifest hash mismatch",
    )
    split = load_json(receipts / "fold1.json")
    _require(split.get("fold") == 1 and split.get("version") == 1, "wrong captured split")
    teachers = _validate_teacher_receipts(packet, split)
    cache = load_json(receipts / "dc_98411230_metadata.json")
    _validate_cache_contract(cache, split, teachers)
    _validate_recipe(packet)

    packet_manifest_path = packet / "PACKET_MANIFEST_R1.json"
    if require_packet_manifest:
        manifest = load_json(packet_manifest_path)
        _require(
            manifest.get("schema") == "wnn.h391-frozen-packet.v1", "wrong packet manifest schema"
        )
        _require(
            manifest.get("files") == _packet_files(packet), "packet files differ from frozen hashes"
        )
    return {
        "source_files": len(files),
        "teacher_count": len(teachers),
        "fit_groups": 11_949,
        "evaluation_groups_excluded": 457,
        "cache_metadata_sha256": EXPECTED_CACHE_SHA256,
        "packet_manifest_sha256": sha256_file(packet_manifest_path)
        if packet_manifest_path.exists()
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PACKET, help="frozen H391 packet root")
    parser.add_argument(
        "--write-manifest", action="store_true", help="write/update the packet content manifest"
    )
    args = parser.parse_args()
    packet = args.root.resolve()
    if args.write_manifest:
        write_packet_manifest(packet)
    print(json.dumps(verify_packet(packet), sort_keys=True))


if __name__ == "__main__":
    main()
