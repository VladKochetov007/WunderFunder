"""Publish durable progress for ROOT's six admitted L4 stem successor fits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR_ROOT = Path(
    r"C:\Users\admin\Documents\Codex\artifacts\l4-next70-20261110"
)
NATIVE_ROOT = "/data/lc-epanov_s/tabdata-stem-successors-20261111-r1"
SOURCE_ROOT = f"{NATIVE_ROOT}/source-r1"
SOURCE_SHA256 = "38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d"
SOURCE_COMMIT = "6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba"
SOURCE_MANIFEST_SHA256 = (
    "e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0"
)
SOURCE_ARCHIVE_SHA256 = (
    "a48b6eccbd68b8eaf2a1fc67338861301185f2ff72df6dd3c8256c3e8263af9c"
)
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
BASELINE_SOURCE_SHA256 = (
    "b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188"
)
BASELINE_PATH = Path(
    r"C:\Users\admin\Documents\Codex\artifacts\l4-causal-stem-20261110\readout-final-summary-r3.json"
)
BASELINE_FILE_SHA256 = (
    "a9f31e0458b053ddabe440f60fdae4770797d5590cea0765ac05c9b175bd7389"
)
TRAIN_SEQUENCES, VALID_SEQUENCES, VALID_ROWS = 10_325, 1_873, 3_528_273
EPOCH_UPDATES, UPDATES = 6_460, 38_760
SEEDS = (203301, 203302)
ARMS = ("k64", "gru512_k16", "gru_dropout01_k16")
CELLS = tuple(f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS)
GPU_BY_CELL = {
    "k64_s203301": 0,
    "gru512_k16_s203301": 1,
    "gru_dropout01_k16_s203301": 2,
    "k64_s203302": 3,
    "gru512_k16_s203302": 5,
    "gru_dropout01_k16_s203302": 6,
}
ARM_SPEC = {
    "k64": {"kernel": 64, "hidden": 256, "dropout": 0.0},
    "gru512_k16": {"kernel": 16, "hidden": 512, "dropout": 0.0},
    "gru_dropout01_k16": {"kernel": 16, "hidden": 256, "dropout": 0.1},
}
MAX_SECONDS, POLL_SECONDS = 5 * 60 * 60, 30
WAVE_DEADLINE_EPOCH = 1_794_377_428.5272088
HELPER = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\publish_pr77.py"
)
HELPER_SHA256 = "a65a92183c0132435d3c4d1a23b79a6890fdacd147e94a353fd23d4f0a6697e1"
CLIENT = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py"
)
CLIENT_SHA256 = "b9da652e028e5dd962df208221d7cbff80b9cc746b0a25944c23b19d870ae793"
LOG = logging.getLogger("l4_successors_publisher")


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"local helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(HELPER, "l4_successors_evidence_readers")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> None:
    if _sha(HELPER) != HELPER_SHA256:
        raise ValueError("local evidence reader differs from the reviewed helper bytes")
    source = P._read_json(ROOT / "SOURCE_R1.json", 16_384)
    manifest = P._read_json(ROOT / "CONFIG_MANIFEST.json", 32_000)
    if (
        source.get("commit") != SOURCE_COMMIT
        or source.get("source_sha256") != SOURCE_SHA256
        or source.get("manifest_sha256") != SOURCE_MANIFEST_SHA256
        or source.get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or source.get("native_root") != NATIVE_ROOT
        or manifest.get("successor_source_commit") != SOURCE_COMMIT
        or manifest.get("successor_source_identity_sha256") != SOURCE_SHA256
        or manifest.get("successor_source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or manifest.get("official_split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(
            "local source/config packet differs from the frozen successor identity"
        )


def _load_baseline() -> dict[str, object]:
    if _sha(BASELINE_PATH) != BASELINE_FILE_SHA256:
        raise ValueError(
            "historical K16 canonical summary bytes differ from the pinned file"
        )
    value = P._read_json(BASELINE_PATH, 256_000)
    scores = value.get("model_scores")
    if (
        value.get("schema") != "wnn.l4-gru-causal-stem-readout-publication-r3.v1"
        or value.get("status") != "complete"
        or value.get("source_sha256") != BASELINE_SOURCE_SHA256
        or value.get("split_sha256") != SPLIT_SHA256
        or not isinstance(scores, dict)
    ):
        raise ValueError(
            "historical K16 readout does not match the frozen comparison source"
        )
    controls: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        row = scores.get(f"k16_s{seed}")
        if (
            not isinstance(row, dict)
            or row.get("seed") != seed
            or row.get("arm") != "k16"
            or row.get("kernel") != 16
            or row.get("metric_id") != "global_wp_clip2_v1"
            or row.get("selected_rows") != VALID_ROWS
            or not all(P._finite_number(row.get(key)) for key in ("t0", "t1", "wp"))
        ):
            raise ValueError(f"historical K16 score is incomplete for seed {seed}")
        controls[seed] = {key: float(row[key]) for key in ("t0", "t1", "wp")}
    return {
        "path": str(BASELINE_PATH),
        "sha256": BASELINE_FILE_SHA256,
        "source_sha256": BASELINE_SOURCE_SHA256,
        "controls": controls,
    }


def _admission(path: Path, pinned: str, config_root: Path) -> dict[str, object]:
    _source_identity()
    if not P._is_sha256(pinned) or _sha(path) != pinned.lower():
        raise ValueError("ROOT fit admission SHA differs from --admission-sha256")
    value = P._read_json(path, 256_000)
    profiles = value.get("profiles")
    manifest = P._read_json(ROOT / "CONFIG_MANIFEST.json", 32_000)
    candidates = manifest.get("configs")
    if (
        value.get("schema") != "wnn.l4-stem-successors-root-fit-admission.v1"
        or value.get("approved") is not True
        or value.get("max_fresh_fits") != 6
        or value.get("source_root") != SOURCE_ROOT
        or value.get("source_sha256") != SOURCE_SHA256
        or value.get("split_sha256") != SPLIT_SHA256
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
        or not isinstance(candidates, list)
    ):
        raise ValueError("ROOT admission differs from the frozen six-cell contract")
    manifest_rows = {
        row.get("path"): row for row in candidates if isinstance(row, dict)
    }
    if set(manifest_rows) != {f"configs/{cell}.json" for cell in CELLS}:
        raise ValueError(
            "config manifest does not enumerate exactly the six admitted cells"
        )
    uuids: set[str] = set()
    for cell in CELLS:
        profile = profiles[cell]
        config_path = config_root / f"{cell}.json"
        raw_sha = _sha(config_path)
        manifest_row = manifest_rows[f"configs/{cell}.json"]
        if (
            not isinstance(profile, dict)
            or profile.get("gpu") != GPU_BY_CELL[cell]
            or profile.get("raw_config_sha256") != raw_sha
            or manifest_row.get("sha256") != raw_sha
            or not P._is_sha256(profile.get("config_sha256"))
            or not P._is_sha256(profile.get("receipt_sha256"))
            or not isinstance(profile.get("receipt_path"), str)
            or not profile.get("receipt_path")
        ):
            raise ValueError(f"admission/config profile binding is invalid for {cell}")
        uuid = profile.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid or _uuid(uuid) in uuids:
            raise ValueError("ROOT admission lacks six unique physical L4 UUIDs")
        uuids.add(_uuid(uuid))
    value["admission_sha256"] = pinned
    return value


def _uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def _expected_config(cell: str, config: object) -> bool:
    seed = int(cell.rsplit("s", 1)[1])
    arm = cell[: -len(f"_s{seed}")]
    model = config.get("model") if isinstance(config, dict) else None
    expected = ARM_SPEC[arm]
    return bool(
        isinstance(config, dict)
        and isinstance(model, dict)
        and config.get("seed") == seed
        and config.get("epochs") == 6
        and config.get("batch_size") == 32
        and config.get("accumulate_chunks") == 1
        and config.get("chunk_rows") == 1000
        and config.get("precision_mode") == "fp32"
        and config.get("optimizer") == "adamw"
        and config.get("lr") == 0.0003
        and config.get("weight_decay") == 0.01
        and config.get("grad_clip") == 1.0
        and config.get("warmup_fraction") == 0.05
        and config.get("schedule") == "warmup_cosine_v2"
        and config.get("min_lr_ratio") == 0.1
        and config.get("normalization_scope") == "all_fit"
        and config.get("normalization_sequences") == TRAIN_SEQUENCES
        and config.get("training_mask") == "all_after_warmup"
        and config.get("objective_mode") == "standalone_labels_wmse_v2"
        and config.get("supervised_target") is None
        and config.get("required_device") == "L4"
        and config.get("required_world_size") == 1
        and model.get("architecture") == "gru"
        and model.get("layers") == 2
        and model.get("hidden") == expected["hidden"]
        and model.get("input_stem_kernel", 0) == expected["kernel"]
        and model.get("dropout", 0.0) == expected["dropout"]
    )


def inspect_cell(
    monitor_root: Path,
    config_root: Path,
    name: str,
    admission: dict[str, object],
) -> dict[str, object]:
    """Read one mirrored fit without modifying producer files or progress state."""
    path = monitor_root / "live-dashboard" / "native" / f"l4-{name}-fit"
    cell: dict[str, object] = {
        "cell": name,
        "identity_ok": False,
        "events": [],
        "terminal": None,
        "complete": False,
    }
    try:
        profile = admission["profiles"][name]
        remote = f"{NATIVE_ROOT}/runs/{name}/fit"
        owner = P._read_json(path / ".sequence_mirror_owner.json", 8192)
        collection = P._read_json(path / "collection_receipt.json", 64_000)
        if (
            owner.get("host") != "l4"
            or owner.get("source_directory") != remote
            or collection.get("host") != "l4"
            or collection.get("source_directory") != remote
        ):
            raise ValueError(
                "mirror owner or collector receipt path differs from the admitted fit"
            )

        metadata = P._read_json(path / "run_metadata.json", 1_000_000)
        config, source, provenance = (
            metadata.get("config"),
            metadata.get("source"),
            metadata.get("provenance"),
        )
        if (
            not isinstance(config, dict)
            or not isinstance(source, dict)
            or not isinstance(provenance, dict)
        ):
            raise TypeError("run metadata lacks config, source or provenance")
        raw_config_sha = _sha(config_root / f"{name}.json")
        if (
            raw_config_sha != profile["raw_config_sha256"]
            or P.canonical_sha256(config) != profile["config_sha256"]
            or source.get("sha256") != SOURCE_SHA256
            or not _expected_config(name, config)
        ):
            raise ValueError(
                "run source/config differs from the ROOT-admitted successor recipe"
            )

        data = provenance.get("data")
        if (
            not isinstance(data, dict)
            or provenance.get("data_sha256") != P.canonical_sha256(data)
            or data.get("split_sha256") != SPLIT_SHA256
            or data.get("teacher") is not None
            or data.get("training_mask") != "all_after_warmup"
        ):
            raise ValueError(
                "data provenance differs from the official train/valid contract"
            )
        fit_members, evaluation_members = (
            data.get("fit_members"),
            data.get("evaluation_members"),
        )
        fit_ok = (
            isinstance(fit_members, list)
            and len(fit_members) == TRAIN_SEQUENCES
            and all(
                isinstance(row, list)
                and len(row) == 2
                and row[0] == "train"
                and type(row[1]) is int
                for row in fit_members
            )
            and len({(row[0], row[1]) for row in fit_members}) == TRAIN_SEQUENCES
        )
        if not fit_ok or evaluation_members != [
            ["valid", i] for i in range(VALID_SEQUENCES)
        ]:
            raise ValueError(
                "fit/full-valid membership differs from official whole-sequence counts"
            )
        normalization = data.get("normalization")
        objective = data.get("objective_contract")
        if (
            not isinstance(normalization, dict)
            or normalization.get("scope") != "all_fit"
            or normalization.get("sequence_count") != TRAIN_SEQUENCES
            or not isinstance(objective, dict)
            or objective.get("mode") != "standalone_labels_wmse_v2"
            or objective.get("reference_training_ancestry") != "none"
        ):
            raise ValueError(
                "normalizer scope or supervised objective ancestry differs"
            )

        runtime = provenance.get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, dict) else None
        expected_uuid = _uuid(str(profile["gpu_uuid"]))
        if (
            provenance.get("world_size") != 1
            or not isinstance(devices, list)
            or len(devices) != 1
            or not isinstance(devices[0], dict)
            or _uuid(str(devices[0].get("uuid", ""))) != expected_uuid
            or devices[0].get("name") != "NVIDIA L4"
        ):
            raise ValueError(
                "runtime GPU UUID/name differs from ROOT's fixed physical assignment"
            )

        cell["identity_ok"] = True
        cell["config_sha256"] = profile["config_sha256"]
        cell["data_sha256"] = provenance["data_sha256"]
        progress = P._read_json(path / "monitor_progress.json", 16_384)
        cell["phase"] = progress.get("phase", "unknown")
        cell["epoch"] = progress.get("epoch")
        rank = P._read_tail_jsonl(path / "rank0.jsonl")
        if rank is not None and rank.get("rank") == 0:
            raw_step = rank.get("step")
            cell["step"] = (
                raw_step if type(raw_step) is int and 0 <= raw_step <= UPDATES else None
            )
            cell["train"] = {
                "loss": rank.get("total_loss"),
                "lr": rank.get("learning_rate"),
                "grad": rank.get("gradient_norm_preclip"),
                "clipped": rank.get("gradient_clipped"),
            }
        else:
            cell["step"] = None
            cell["train"] = {}

        raw_events, journal_error = P._read_evaluations(
            path / "monitor_evaluations.jsonl"
        )
        checked: list[dict[str, object]] = []
        seen: set[int] = set()
        for event in raw_events:
            epoch = event.get("epoch")
            if type(epoch) is not int or epoch in seen:
                checked.append(
                    {"epoch": epoch, "ok": False, "error": "duplicate/invalid epoch"}
                )
                continue
            seen.add(epoch)
            ok, error = P._validate_eval_event(event, data, epoch)
            context = event.get("context")
            if ok and (
                not isinstance(context, dict)
                or context.get("metric_id") != "global_wp_clip2_v1"
            ):
                ok, error = (
                    False,
                    "evaluation metric is not the pinned weighted-Pearson score",
                )
            checked.append(
                {
                    "epoch": epoch,
                    "ok": ok,
                    "event": event if ok else None,
                    "error": error,
                }
            )
        cell["events"] = checked
        cell["journal_error"] = journal_error
        valid_events = [
            item
            for item in checked
            if item.get("ok") and isinstance(item.get("event"), dict)
        ]
        if valid_events:
            cell["latest_event"] = valid_events[-1]["event"]

        terminal_path = path / "terminal_status.json"
        if terminal_path.exists():
            terminal = P._read_json(terminal_path, 2_000_000)
            identity = terminal.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("source_sha256") != SOURCE_SHA256
                or identity.get("config_sha256") != profile["config_sha256"]
                or identity.get("data_sha256") != provenance.get("data_sha256")
                or identity.get("world_size") != 1
            ):
                raise ValueError("terminal identity differs from the admitted fit")
            if terminal.get("status") == "completed" and (
                terminal.get("completed_epochs") != 6
                or terminal.get("global_updates") != UPDATES
                or not P._is_sha256(terminal.get("model_sha256"))
            ):
                raise ValueError("terminal does not prove exact fixed-E6 completion")
            cell["terminal"] = {
                "status": terminal.get("status"),
                "model_sha256": terminal.get("model_sha256"),
                "sha256": _sha(terminal_path),
            }
        cell["complete"] = bool(
            isinstance(cell.get("terminal"), dict)
            and cell["terminal"].get("status") == "completed"
            and journal_error is None
            and len(checked) == 6
            and [item.get("epoch") for item in checked] == [1, 2, 3, 4, 5, 6]
            and all(item.get("ok") is True for item in checked)
            and isinstance(cell.get("latest_event"), dict)
            and cell["latest_event"].get("step") == UPDATES
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        cell["error"] = str(error)
    return cell


def _metric_scores(cell: dict[str, object]) -> dict[str, float] | None:
    event = cell.get("latest_event")
    metrics = event.get("metrics") if isinstance(event, dict) else None
    if not isinstance(metrics, dict):
        return None
    values = {key: metrics.get(f"valid/wp/{key}") for key in ("t0", "t1")}
    if not all(P._finite_number(value) for value in values.values()):
        return None
    result = {key: float(value) for key, value in values.items()}
    result["wp"] = (result["t0"] + result["t1"]) / 2.0
    return result


def paired_rows(
    cells: list[dict[str, object]], baseline: dict[str, object]
) -> list[dict[str, object]] | None:
    """Return only fixed-E6 paired monitor deltas against same-seed K16 readouts."""
    by_name = {str(cell.get("cell")): cell for cell in cells}
    if set(by_name) != set(CELLS) or not all(
        by_name[name].get("complete") is True for name in CELLS
    ):
        return None
    controls = baseline.get("controls")
    if not isinstance(controls, dict):
        return None
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        base = controls.get(seed)
        if not isinstance(base, dict):
            return None
        for arm in ARMS:
            name = f"{arm}_s{seed}"
            score = _metric_scores(by_name[name])
            if score is None:
                return None
            d0 = score["t0"] - float(base["t0"])
            d1 = score["t1"] - float(base["t1"])
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "candidate": score,
                    "baseline": base,
                    "delta_t0": d0,
                    "delta_t1": d1,
                    "delta_wp": (d0 + d1) / 2.0,
                }
            )
    return rows


def _fmt(value: object) -> str:
    return f"{float(value):.6f}" if P._finite_number(value) else "unknown"


def render_report(
    admission: dict[str, object],
    cells: list[dict[str, object]],
    clock: dict[str, object],
    baseline: dict[str, object],
    pr: int,
    title: str,
) -> str:
    now = clock.get("local_now")
    stamp = (
        datetime.fromtimestamp(float(now), UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
        if P._finite_number(now)
        else "local clock unavailable"
    )
    age = clock.get("age_seconds")
    freshness = f"{float(age):.0f}s" if P._finite_number(age) else "unknown"
    lines = [
        f"## L4 stem successors PR{pr} — {title}",
        f"Observed locally {stamp}; mirror last-applied age {freshness}. Producer timestamps are unchanged.",
        f"Admission `{admission['admission_sha256']}`; source `{SOURCE_SHA256}`; split `{SPLIT_SHA256}`.",
        f"Fit membership: {TRAIN_SEQUENCES} official train sequences; full validation: {VALID_SEQUENCES} sequences / {VALID_ROWS:,} selected rows.",
        f"Reused K16 comparison source `{BASELINE_SOURCE_SHA256}`; canonical readout `{baseline['sha256']}`.",
        f"Evidence reader SHA `{HELPER_SHA256}`.",
        "",
        "### Six fits: phase, actual rank0 step, loss, LR, preclip grad, clipped, latest full-valid WP t0/t1",
    ]
    errors = clock.get("last_errors")
    if isinstance(errors, list) and errors:
        lines.insert(
            2,
            f"Mirror collector reports {len(errors)} recent error(s); check per-file freshness.",
        )
    for cell in cells:
        name = str(cell["cell"])
        if cell.get("identity_ok") is not True:
            lines.append(
                f"- `{name}` identity unverified; metrics withheld ({cell.get('error', 'mirror metadata pending')})."
            )
            continue
        seed = int(name.rsplit("s", 1)[1])
        arm = name[: -len(f"_s{seed}")]
        terminal = cell.get("terminal")
        phase = (
            str(terminal.get("status"))
            if isinstance(terminal, dict)
            else f"{cell.get('phase', 'unknown')} {cell.get('epoch', '?')}/6E"
        )
        scores = _metric_scores(cell)
        event = cell.get("latest_event")
        event_epoch = event.get("epoch") if isinstance(event, dict) else None
        valid = (
            f"E{event_epoch} {scores['t0']:.6f}/{scores['t1']:.6f}"
            if scores is not None
            else "pending"
        )
        train = cell.get("train") if isinstance(cell.get("train"), dict) else {}
        model_sha = (
            f"; trainer model SHA `{terminal['model_sha256'][:12]}…` (not locally byte-verified)"
            if isinstance(terminal, dict) and P._is_sha256(terminal.get("model_sha256"))
            else ""
        )
        lines.append(
            f"- `{name}` GPU{GPU_BY_CELL[name]}, {arm}, seed{seed}: {phase}; "
            f"{cell.get('step', '?')}/{UPDATES:,}; loss {_fmt(train.get('loss'))}; "
            f"LR {_fmt(train.get('lr'))}; preclip grad {_fmt(train.get('grad'))}; "
            f"clipped {train.get('clipped', 'unknown')}; valid {valid}{model_sha}; "
            f"config `{cell['config_sha256']}`; data `{cell['data_sha256']}`"
        )

    lines.extend(
        [
            "",
            "### Fixed-E6 paired monitor comparisons vs same-seed reused K16 controls",
        ]
    )
    comparisons = paired_rows(cells, baseline)
    if comparisons is None:
        lines.append(
            "Withheld until all six fixed-E6 terminals and exact full-valid events pass."
        )
    else:
        for row in comparisons:
            candidate = row["candidate"]
            base = row["baseline"]
            lines.append(
                f"- {row['arm']} seed{row['seed']}: monitor t0/t1/WP "
                f"{candidate['t0']:.6f}/{candidate['t1']:.6f}/{candidate['wp']:.6f}; "
                f"canonical K16 t0/t1/WP "
                f"{base['t0']:.6f}/{base['t1']:.6f}/{base['wp']:.6f}; "
                f"delta vs canonical K16 t0/t1/WP "
                f"{row['delta_t0']:+.6f}/{row['delta_t1']:+.6f}/{row['delta_wp']:+.6f}."
            )
        lines.append(
            "These are provisional monitor comparisons. Fresh canonical readout/score verification for all six models is still pending."
        )
    lines.append(
        "Official validation has been repeatedly exposed as development evidence. Descriptive paired screen only; no significance or public-score claim."
    )
    body = "\n".join(lines)
    if len(body.encode("utf-8")) > 30_000:
        raise ValueError("progress report exceeds the 30 KB comment bound")
    return body


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema": "wnn.l4-stem-successors-publisher-state.v1", "reports": {}}
    state = P._read_json(path, 2_000_000)
    if state.get(
        "schema"
    ) != "wnn.l4-stem-successors-publisher-state.v1" or not isinstance(
        state.get("reports"), dict
    ):
        raise ValueError(
            "publisher state is malformed or belongs to a different publisher"
        )
    return state


def _save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(
        state, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    if len(payload) > 2_000_000:
        raise ValueError("publisher state exceeds 2 MB")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_once(
    client: object, state_path: Path, pr: int, marker: str, body: str, when: float
) -> str:
    """Persist a pending idempotency marker before posting or reconciling it."""
    state = _load_state(state_path)
    reports = state["reports"]
    assert isinstance(reports, dict)
    saved = reports.get(marker)
    if isinstance(saved, dict) and saved.get("status") == "posted":
        return "already-posted"
    if isinstance(saved, dict) and saved.get("status") == "pending":
        body, when = str(saved["body"]), float(saved["prepared_at"])
    found = client.find_report(pr, marker, when)  # type: ignore[attr-defined]
    if found is not None:
        reports[marker] = {
            "status": "posted",
            "body": body,
            "prepared_at": when,
            "comment_id": found.get("id"),
        }
        _save_state(state_path, state)
        return "reconciled-existing"

    def before_post() -> None:
        reports[marker] = {"status": "pending", "body": body, "prepared_at": when}
        _save_state(state_path, state)

    result = client.post_report(pr, body, before_post=before_post)  # type: ignore[attr-defined]
    reports[marker] = {
        "status": "posted",
        "body": body,
        "prepared_at": when,
        "comment_id": result.get("id"),
        "html_url": result.get("html_url"),
    }
    _save_state(state_path, state)
    return "posted"


def _client(pr: int, deadline: float) -> object:
    if not CLIENT.is_file() or _sha(CLIENT) != CLIENT_SHA256:
        raise RuntimeError(
            f"existing local GitHub progress client is unavailable: {CLIENT}"
        )
    module = _module(CLIENT, "sequence_progress_github_l4_successors")
    return module.ProgressGitHub({pr}, deadline=deadline)


def run(args: argparse.Namespace) -> int:
    if args.pr <= 0 or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError(
            "invalid PR number or --max-seconds outside the five-hour bound"
        )
    config_root = args.config_root or ROOT / "configs"
    admission_path = args.admission or ROOT / "ROOT-fit-admission-r1.json"
    args.admission_sha256 = args.admission_sha256.lower()
    admission = _admission(admission_path, args.admission_sha256, config_root)
    baseline = _load_baseline()
    state_path = (
        args.state_file or ROOT / "reporting" / f"publisher-state-pr{args.pr}.json"
    )
    wall_remaining = max(0, WAVE_DEADLINE_EPOCH - time.time())
    wall_seconds = min(args.max_seconds, int(wall_remaining))
    deadline = time.monotonic() + max(0, wall_seconds)
    client = None if args.dry_run else _client(args.pr, deadline)
    while True:
        if _sha(admission_path) != args.admission_sha256:
            raise ValueError("ROOT admission bytes changed while publishing")
        cells = [
            inspect_cell(args.monitor_root, config_root, name, admission)
            for name in CELLS
        ]
        clock = P._source_clock(ROOT)
        now = (
            float(clock["local_now"])
            if P._finite_number(clock.get("local_now"))
            else time.time()
        )
        bucket = datetime.fromtimestamp(int(now // 1800) * 1800, UTC).strftime(
            "%Y%m%dT%H%M"
        )
        due: list[tuple[str, str]] = [
            (
                f"<!-- wnn-l4-successors-pr{args.pr}-30m-{bucket}Z -->",
                "30-minute progress",
            )
        ]
        for cell in cells:
            terminal = cell.get("terminal")
            if isinstance(terminal, dict):
                digest = str(terminal.get("sha256", "unknown"))[:16]
                due.append(
                    (
                        f"<!-- wnn-l4-successors-pr{args.pr}-terminal-{cell['cell']}-{digest} -->",
                        f"terminal {cell['cell']} ({terminal.get('status')})",
                    )
                )

        all_terminal = len(cells) == 6 and all(
            isinstance(cell.get("terminal"), dict) for cell in cells
        )
        complete = len(cells) == 6 and all(
            cell.get("complete") is True for cell in cells
        )
        terminal_failure = all_terminal and any(
            cell["terminal"].get("status") != "completed" for cell in cells
        )
        deadline_reached = (
            time.time() >= WAVE_DEADLINE_EPOCH or time.monotonic() >= deadline
        )
        final_due = complete or terminal_failure or deadline_reached
        if final_due:
            digest = P.canonical_sha256(
                [
                    (
                        cell.get("cell"),
                        cell.get("terminal", {}).get("sha256")
                        if isinstance(cell.get("terminal"), dict)
                        else None,
                        cell.get("step"),
                        cell.get("complete"),
                    )
                    for cell in cells
                ]
            )[:16]
            due.append(
                (
                    f"<!-- wnn-l4-successors-pr{args.pr}-final-{digest} -->",
                    "final six-fit panel",
                )
            )

        delivered = True
        for marker, title in due:
            body = (
                render_report(admission, cells, clock, baseline, args.pr, title)
                + f"\n\n{marker}"
            )
            if args.dry_run:
                print(body)
                continue
            try:
                outcome = publish_once(client, state_path, args.pr, marker, body, now)
                print(f"{outcome}: {marker}")
            except (RuntimeError, ValueError) as error:
                delivered = False
                LOG.warning("%s: %s", marker, type(error).__name__)
        if final_due and delivered:
            return 0
        if args.once or args.dry_run:
            return 0
        if deadline_reached:
            LOG.error("publisher deadline reached; final panel could not be confirmed")
            return 0
        remaining = min(deadline - time.monotonic(), WAVE_DEADLINE_EPOCH - time.time())
        if remaining <= 0:
            LOG.error("publisher deadline reached; final panel could not be confirmed")
            return 0
        time.sleep(min(POLL_SECONDS, remaining))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--monitor-root", type=Path, default=DEFAULT_MONITOR_ROOT)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
