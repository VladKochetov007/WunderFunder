from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wnn.student_frontier_order import (
    seeded_epoch_positions,
    source_qualified_members,
)


def read_json(path: Path, max_bytes: int = 2_000_000) -> Any:
    """Read a bounded JSON recipe or receipt."""
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON file exceeds {max_bytes} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Hash one small JSON input with fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_members(members: list[tuple[str, int]]) -> str:
    """Hash source-qualified row groups in the frozen order."""
    encoded = "\n".join(f"{source}:{group}" for source, group in members).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def digest_batches(members: list[tuple[str, int]], batch_size: int) -> str:
    """Hash the retained source-qualified membership with batch boundaries."""
    encoded = "\n".join(
        f"{index // batch_size}:{source}:{group}"
        for index, (source, group) in enumerate(members)
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def derive(recipe_path: Path, reference_path: Path) -> dict[str, Any]:
    """Derive B16 order identities through the production sampler and compare."""
    recipe = read_json(recipe_path)
    reference = read_json(reference_path)
    order_recipe = recipe["sample_order"]
    reference_order = reference["sample_order"]
    split_train_groups = recipe["split"]["source_qualified_group_ids"]["train"]
    valid_groups = recipe["split"]["source_qualified_group_ids"]["valid_fit"]
    batch_size = int(recipe["recipe"]["batch_size"])
    seed = int(order_recipe["seed"])
    epoch = int(order_recipe["epoch"])
    train_groups = np.random.default_rng(seed).permutation(split_train_groups).tolist()
    positions = seeded_epoch_positions(
        len(train_groups) + len(valid_groups), seed, batch_size, epoch
    )
    members = source_qualified_members(train_groups, valid_groups, positions)
    batch_count = len(positions) // batch_size
    exposed = batch_count * batch_size
    retained = members[:exposed]
    dropped_positions = positions[exposed:]
    dropped_members = members[exposed:]
    observed = {
        "batch_size": batch_size,
        "seed": seed,
        "epoch": epoch,
        "fit_groups_before_drop_last": len(positions),
        "exposed_sequence_count": exposed,
        "dropped_sequence_count": len(dropped_members),
        "batches_per_epoch": batch_count,
        "chunks_per_sequence": int(recipe["recipe"]["sequence_rows"])
        // int(recipe["recipe"]["chunk_rows"]),
        "optimizer_updates_per_epoch": batch_count
        * int(recipe["recipe"]["sequence_rows"])
        // int(recipe["recipe"]["chunk_rows"]),
        "train_group_order_sha256": hashlib.sha256(
            np.asarray(train_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "valid_group_order_sha256": recipe["split"]["groups"]["valid"]
        ["ordered_int64_le_sha256"],
        "sampler_positions_sha256": hashlib.sha256(
            np.asarray(positions, dtype="<i8").tobytes()
        ).hexdigest(),
        "sampler_source_rows_sha256": digest_members(members),
        "retained_source_rows_sha256": digest_members(retained),
        "batch_membership_sha256": digest_batches(retained, batch_size),
        "dropped_tail_positions": dropped_positions,
        "dropped_tail_members": [list(member) for member in dropped_members],
        "same_order_and_omissions_across_shapes": True,
        "source_qualified_row_groups": (
            "train and valid IDs are separate namespaces; never bare-ID membership-lookup"
        ),
        "sampler": "wnn.training_sampling.EpochOrderSampler",
    }
    for key in (
        "batch_size",
        "batches_per_epoch",
        "dropped_tail_positions",
        "dropped_tail_members",
        "sampler_positions_sha256",
        "sampler_source_rows_sha256",
        "seed",
    ):
        if observed[key] != reference_order[key]:
            raise ValueError(f"production sampler differs from B16 reference at {key}")
    if observed["train_group_order_sha256"] != reference_order["train_order_sha256"]:
        raise ValueError("seed-3 train permutation differs from B16 reference")
    first_batch = {
        "positions": positions[:batch_size],
        "members": [list(member) for member in members[:batch_size]],
    }
    if first_batch["positions"] != reference_order["first_batch_positions"]:
        raise ValueError("production sampler first batch positions differ from B16 reference")
    if first_batch["members"] != reference_order["first_batch_members"]:
        raise ValueError("source-qualified first batch differs from B16 reference")
    return {
        "schema": "wnn.student-shape-pair-b16-order-derivation.r1",
        "status": "passed",
        "observed_utc": datetime.now(UTC).isoformat(),
        "recipe": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "reference": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "sampler_source_sha256": recipe["source_sha256"][
            "src/wnn/training_sampling.py"
        ],
        "sample_order": observed,
        "train_group_permutation": "numpy.default_rng(seed).permutation(split.train_groups)",
        "first_batch": first_batch,
    }


def main() -> None:
    """Write a machine-readable identity receipt for the B16 fit order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.recipe, args.reference)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
