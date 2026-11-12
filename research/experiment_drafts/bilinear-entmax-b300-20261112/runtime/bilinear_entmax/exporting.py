"""Export full official-valid predictions after strict checkpoint admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bilinear_entmax.endpoint import load_endpoint


def export_endpoint(
    checkpoint: Path,
    data_root: Path,
    split_manifest: Path,
    output: Path,
    runtime_root: Path,
) -> None:
    """Write one canonical full-sequence prediction shard per official valid group."""
    import numpy as np

    from bilinear_entmax.bindings import SPLIT_SHA256
    from wnn.data import SEQUENCE_LENGTH
    from wnn.splits import file_sha256, load_campaign_split
    from wnn.training.sequence_data import SequenceRecords
    from wnn.training.sequence_distributed import (
        barrier,
        collective_output_setup,
        coordinated_call,
        initialize_distributed,
        shutdown_distributed,
    )
    from wnn.training.sequence_predict import _producer, predict_sequence

    context = initialize_distributed()
    try:
        if context.world_size != 2:
            raise ValueError("Bilinear Entmax endpoint export requires exactly two ranks")

        def load_all() -> tuple[Any, Any, list[tuple[str, int]]]:
            """Admit complete official partitions and the fixed-final checkpoint."""
            split = load_campaign_split(split_manifest, data_root)
            if split.sha256 != SPLIT_SHA256:
                raise ValueError("export split differs from the pinned Bilinear Entmax split")
            members = [("valid", group) for group in split.evaluation_groups]
            if len(members) != 1_873 or len(set(members)) != 1_873:
                raise ValueError("export requires all 1,873 official evaluation sequences")
            fit_members = [["train", group] for group in split.train_groups] + [
                ["valid", group] for group in split.valid_groups
            ]
            if len(fit_members) != 10_325:
                raise ValueError("checkpoint admission requires all 10,325 fit sequences")
            model, payload = load_endpoint(
                checkpoint, runtime_root, split.sha256, fit_members, context.device
            )
            return split, (model, payload), members

        split, loaded, members = coordinated_call(
            load_all, "Bilinear Entmax endpoint admission", context
        )
        model, payload = loaded
        collective_output_setup(output, False, context)
        barrier(context)
        dataset = coordinated_call(
            lambda: SequenceRecords(data_root, members, 0),
            "official validation sequence construction",
            context,
        )

        def export_rank() -> list[dict[str, Any]]:
            """Export only complete full-sequence predictions for this rank's rows."""
            entries: list[dict[str, Any]] = []
            for index in range(context.rank, len(dataset), context.world_size):
                source, group = members[index]
                values = np.asarray(
                    predict_sequence(
                        model,
                        dataset[index]["x"],
                        int(payload["config"]["chunk_rows"]),
                        context.device,
                    )
                )
                if (
                    values.dtype != np.float32
                    or values.shape != (SEQUENCE_LENGTH, 2)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"invalid endpoint output for {source}/{group}")
                filename = f"{source}-{group}.npy"
                destination = output / filename
                temporary = output / f"{filename}.partial"
                with temporary.open("xb") as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(destination)
                entries.append(
                    {
                        "split": source,
                        "row_group": group,
                        "file": filename,
                        "sha256": file_sha256(destination),
                    }
                )
            shard = output / f"rank-{context.rank}.json"
            partial = output / f"rank-{context.rank}.json.partial"
            with partial.open("x", encoding="utf-8") as stream:
                json.dump(entries, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            partial.replace(shard)
            return entries

        coordinated_call(export_rank, "Bilinear Entmax prediction shard", context)
        barrier(context)

        def publish_manifest() -> bool:
            """Join rank shards and atomically publish the complete scorer manifest."""
            if not context.is_primary:
                return True
            entries = [
                entry
                for rank in range(context.world_size)
                for entry in json.loads((output / f"rank-{rank}.json").read_text(encoding="utf-8"))
            ]
            by_member = {(entry["split"], entry["row_group"]): entry for entry in entries}
            if len(entries) != 1_873 or len(by_member) != 1_873 or set(by_member) != set(members):
                raise ValueError("prediction manifest has missing or duplicate valid sequences")
            manifest = {
                "schema": "sequence-predictions-v1",
                "complete": True,
                "input_schema": "raw112-v1",
                "membership": "evaluation",
                "split_sha256": split.sha256,
                "hint_dim": 0,
                "sequence_length": SEQUENCE_LENGTH,
                "producer": _producer(checkpoint, payload),
                "world_size": context.world_size,
                "sequences": [by_member[member] for member in members],
            }
            partial = output / "manifest.json.partial"
            with partial.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            partial.replace(output / "manifest.json")
            return True

        coordinated_call(publish_manifest, "Bilinear Entmax manifest publication", context)
        barrier(context)
    finally:
        shutdown_distributed(context)
