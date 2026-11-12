"""Export fixed-final Impact81 endpoints through the canonical sequence scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from impact81.endpoint import load_endpoint


def export_endpoint(
    checkpoint: Path,
    data_root: Path,
    split_manifest: Path,
    output: Path,
    runtime_root: Path,
) -> None:
    """Export complete official-valid predictions in the canonical scorer format.

    Requires the pinned runtime; exports every row without target-dependent prediction
    selection. Scoring remains in ``sequence_score.score_export``.
    """
    import numpy as np

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

    from .bindings import SPLIT_SHA256

    context = initialize_distributed()
    try:

        def _load_all() -> tuple[Any, Any, list[tuple[str, int]]]:
            split = load_campaign_split(split_manifest, data_root)
            if split.sha256 != SPLIT_SHA256:
                raise ValueError("export split differs from the pinned Impact81 primary split")
            members = [("valid", group) for group in split.evaluation_groups]
            if len(members) != 1873 or len(set(members)) != 1873:
                raise ValueError("export requires all 1,873 official validation sequences")
            fit_members = [["train", group] for group in split.train_groups] + [
                ["valid", group] for group in split.valid_groups
            ]
            if len(fit_members) != 10325:
                raise ValueError("pinned endpoint requires all 10,325 official fit sequences")
            model, payload = load_endpoint(
                checkpoint,
                runtime_root,
                split.sha256,
                fit_members,
                context.device,
            )
            model.eval()
            return split, (model, payload), members

        split, loaded, members = coordinated_call(_load_all, "Impact81 endpoint admission", context)
        model, payload = loaded
        collective_output_setup(output, False, context)
        barrier(context)

        dataset = coordinated_call(
            lambda: SequenceRecords(data_root, members, 0),
            "official-valid sequence construction",
            context,
        )

        def _export_rank() -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            for index in range(context.rank, len(dataset), context.world_size):
                source, group = members[index]
                values = predict_sequence(
                    model,
                    dataset[index]["x"],
                    int(payload["config"]["chunk_rows"]),
                    context.device,
                )
                values = np.asarray(values)
                if (
                    values.dtype != np.float32
                    or values.ndim != 2
                    or values.shape
                    != (
                        SEQUENCE_LENGTH,
                        2 + int(payload["config"]["model"].get("hint_dim", 0)),
                    )
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"invalid full-sequence output for {source}/{group}")
                values = values[:, :2].copy()
                name = f"{source}-{group}.npy"
                target = output / name
                temporary = output / f"{name}.partial"
                with temporary.open("xb") as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(target)
                entries.append(
                    {
                        "split": source,
                        "row_group": group,
                        "file": name,
                        "sha256": file_sha256(target),
                    }
                )
            part = output / f"rank-{context.rank}.json"
            temporary_part = output / f"rank-{context.rank}.json.partial"
            with temporary_part.open("x", encoding="utf-8") as stream:
                json.dump(entries, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            temporary_part.replace(part)
            return entries

        coordinated_call(_export_rank, "Impact81 prediction shard", context)
        barrier(context)

        def _publish_manifest() -> bool:
            if not context.is_primary:
                return True
            all_entries = [
                entry
                for worker in range(context.world_size)
                for entry in json.loads(
                    (output / f"rank-{worker}.json").read_text(encoding="utf-8")
                )
            ]
            entry_map = {(entry["split"], entry["row_group"]): entry for entry in all_entries}
            if len(all_entries) != 1873 or len(entry_map) != 1873 or set(entry_map) != set(members):
                raise ValueError("export manifest has missing or duplicate validation sequences")
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
                "sequences": [entry_map[member] for member in members],
            }
            temporary = output / "manifest.json.partial"
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(output / "manifest.json")
            return True

        coordinated_call(_publish_manifest, "Impact81 manifest publication", context)
        barrier(context)
    finally:
        shutdown_distributed(context)
