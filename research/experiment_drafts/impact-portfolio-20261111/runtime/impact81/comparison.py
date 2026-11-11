"""Compare complete paired Impact81 score receipts without rerunning a scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from impact81.variants import CARDS, TREATMENTS, variant_for

SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
METRIC_ID = "global_wp_clip2_v1"
EXPECTED_ROWS = 3_528_273
EXPECTED_SEQUENCES = 1_873

ARMS = ("control", "treatment")
SEEDS = (204101, 204102)
VARIANTS = TREATMENTS


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require a JSON object at a score receipt boundary."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _sha256(value: object, name: str) -> str:
    """Validate a lowercase SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str) -> float:
    """Read one finite numeric metric, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _canonical_sha256(value: object) -> str:
    """Match the runtime's stable JSON hash for config and source-file maps."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_score(path: Path) -> dict[str, Any]:
    """Read one canonical score receipt and identify its declared paired cell."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_000_000:
        raise ValueError(f"missing, unsafe or oversized score receipt: {path}")
    receipt = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    score = _mapping(receipt.get("score"), f"{path}.score")
    if (
        score.get("metric_id") != METRIC_ID
        or score.get("selected_rows") != EXPECTED_ROWS
        or receipt.get("sequences") != EXPECTED_SEQUENCES
        or receipt.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(f"{path} is not a complete pinned official-valid score receipt")
    wp = _finite(score.get("wp"), f"{path}.score.wp")
    t0 = _finite(score.get("t0"), f"{path}.score.t0")
    t1 = _finite(score.get("t1"), f"{path}.score.t1")
    producer = _mapping(receipt.get("producer"), f"{path}.producer")
    config = _mapping(producer.get("config"), f"{path}.producer.config")
    impact = _mapping(config.get("impact81"), f"{path}.producer.config.impact81")
    if impact.get("schema") != "impact81.model.v1":
        raise ValueError(f"{path} does not identify an Impact81 endpoint")
    cell = _mapping(impact.get("cell"), f"{path}.producer.config.impact81.cell")
    card, arm, variant, seed = (cell.get(key) for key in ("card", "arm", "variant", "seed"))
    if (
        cell.get("schema") != "impact81.cell.v1"
        or set(cell)
        != {"schema", "id", "card", "arm", "variant", "seed", "native_config", "native_sha256"}
        or card not in VARIANTS
        or arm not in ARMS
        or variant != variant_for(card, arm)
        or seed not in SEEDS
        or config.get("seed") != seed
        or cell.get("id") != f"ip{card}_{arm}_s{seed}"
    ):
        raise ValueError(f"{path} has a malformed or unsupported Impact81 cell")
    if config.get("epochs") != 4:
        raise ValueError(f"{path} is not a four-epoch endpoint")
    preprocessing_sha256 = _sha256(
        impact.get("preprocessing_sha256"), f"{path}.preprocessing_sha256"
    )
    provenance = _mapping(producer.get("provenance"), f"{path}.producer.provenance")
    identity = _mapping(provenance.get("identity"), f"{path}.producer.provenance.identity")
    source_sha256 = _sha256(identity.get("source_sha256"), f"{path}.source_sha256")
    source_files = _mapping(identity.get("source_files"), f"{path}.source_files")
    if _canonical_sha256(source_files) != source_sha256:
        raise ValueError(f"{path} source hash does not bind its source file map")
    if identity.get("config_sha256") != _canonical_sha256(config):
        raise ValueError(f"{path} config hash does not bind its producer config")
    checkpoint_sha256 = _sha256(producer.get("checkpoint_sha256"), f"{path}.checkpoint_sha256")
    _sha256(receipt.get("predictions_manifest_sha256"), f"{path}.predictions_manifest_sha256")
    return {
        "path": str(path.resolve()),
        "card": card,
        "arm": arm,
        "variant": variant,
        "seed": seed,
        "wp": wp,
        "t0": t0,
        "t1": t1,
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "split_sha256": receipt["split_sha256"],
        "base_config": {key: value for key, value in config.items() if key != "impact81"},
    }


def compare(
    score_paths: list[Path],
    output: Path,
    *,
    card: str | None = None,
    allow_one_card: bool = False,
) -> dict[str, Any]:
    """Persist all paired treatment-minus-control deltas from canonical score receipts.

    A full portfolio requires all twenty-four endpoints. A four-endpoint subset is
    accepted only when one card is explicitly selected and allow_one_card is true.
    No score is generated, imputed, selected, or ranked here.
    """
    if not score_paths:
        raise ValueError("at least one existing canonical score receipt is required")
    if len({path.resolve() for path in score_paths}) != len(score_paths):
        raise ValueError("score receipt paths must be unique")
    if allow_one_card:
        if card not in CARDS:
            raise ValueError(f"--allow-one-card requires an explicit card in {CARDS}")
        selected_cards = (card,)
    else:
        if card is not None:
            raise ValueError("selecting one card requires --allow-one-card")
        selected_cards = CARDS
    expected = {
        (selected_card, arm, seed)
        for selected_card in selected_cards
        for arm in ARMS
        for seed in SEEDS
    }
    if len(score_paths) != len(expected):
        raise ValueError(
            f"expected {len(expected)} score receipts for {selected_cards}, received {len(score_paths)}"
        )
    records = [_read_score(path) for path in score_paths]
    by_cell = {(row["card"], row["arm"], row["seed"]): row for row in records}
    if len(by_cell) != len(records):
        raise ValueError("duplicate score receipts for an Impact81 cell")
    observed = set(by_cell)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"score panel mismatch; missing={missing}, unexpected={unexpected}")
    if len({row["source_sha256"] for row in records}) != 1:
        raise ValueError("score receipts do not share one pinned source identity")
    if len({row["split_sha256"] for row in records}) != 1:
        raise ValueError("score receipts do not share one official split")
    if len({row["preprocessing_sha256"] for row in records}) != 1:
        raise ValueError("score receipts do not share one train-only preprocessing artifact")
    if len({row["checkpoint_sha256"] for row in records}) != len(records):
        raise ValueError("each declared cell must have its own fitted endpoint checkpoint")

    pairs: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for selected_card in selected_cards:
        card_pairs = []
        for seed in SEEDS:
            control = by_cell[(selected_card, "control", seed)]
            treatment = by_cell[(selected_card, "treatment", seed)]
            if control["base_config"] != treatment["base_config"]:
                raise ValueError(f"card {selected_card} seed {seed} is not a matched config pair")
            pair = {
                "card": selected_card,
                "variant": VARIANTS[selected_card],
                "seed": seed,
                "control": {
                    "wp": control["wp"],
                    "t0": control["t0"],
                    "t1": control["t1"],
                    "checkpoint_sha256": control["checkpoint_sha256"],
                    "score_receipt": control["path"],
                },
                "treatment": {
                    "wp": treatment["wp"],
                    "t0": treatment["t0"],
                    "t1": treatment["t1"],
                    "checkpoint_sha256": treatment["checkpoint_sha256"],
                    "score_receipt": treatment["path"],
                },
                "delta_wp": treatment["wp"] - control["wp"],
                "delta_t0": treatment["t0"] - control["t0"],
                "delta_t1": treatment["t1"] - control["t1"],
            }
            pairs.append(pair)
            card_pairs.append(pair)
        summaries[selected_card] = {
            "variant": VARIANTS[selected_card],
            "paired_seed_count": len(card_pairs),
            "mean_delta_wp": sum(item["delta_wp"] for item in card_pairs) / len(card_pairs),
            "mean_delta_t0": sum(item["delta_t0"] for item in card_pairs) / len(card_pairs),
            "mean_delta_t1": sum(item["delta_t1"] for item in card_pairs) / len(card_pairs),
        }
    result = {
        "schema": "impact81.paired-comparison.v1",
        "status": "complete_panel" if len(selected_cards) == len(CARDS) else "one_card_subset",
        "metric_id": METRIC_ID,
        "split_sha256": SPLIT_SHA256,
        "source_sha256": records[0]["source_sha256"],
        "preprocessing_sha256": records[0]["preprocessing_sha256"],
        "expected_selected_rows": EXPECTED_ROWS,
        "expected_sequences": EXPECTED_SEQUENCES,
        "endpoint_count": len(records),
        "cards": list(selected_cards),
        "card_summaries": summaries,
        "paired_deltas": pairs,
        "selection": "none; every declared pair is reported",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Compare already-scored endpoints; this command never loads a model or scores data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", choices=CARDS)
    parser.add_argument("--allow-one-card", action="store_true")
    args = parser.parse_args(argv)
    compare(args.scores, args.output, card=args.card, allow_one_card=args.allow_one_card)


if __name__ == "__main__":
    main()
