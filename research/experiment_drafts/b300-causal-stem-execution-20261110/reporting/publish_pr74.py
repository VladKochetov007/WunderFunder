"""Publish bounded progress for ROOT-admitted PR74 B300 input-stem fits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NB = Path(r"C:\Users\admin\Documents\Codex\artifacts\b300-causal-stem-20261110")
OLDN = Path(r"C:\Users\admin\Documents\Codex\artifacts\l4-next70-20261110")
HELPER = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\publish_pr77.py"
)
CLIENT_PATH = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py"
)
ADMISSION_SHA = "fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18"
AUTHORITY_SHA = "74663454d5c8ace8f7c1cdeb6c651e3d9a93fb9b2e1c4cc756bd03fb45df3f77"
SOURCE_SHA = "959fc8a335f97a83de31c12dfa6a24f5db4776b78b70bf45f7890cf09b96b5d6"
SOURCE_COMMIT = "8a8afec964985a6ba8b13d1c9af1ab00e14c6504"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TRAIN_N, VALID_N, VALID_ROWS = 10_325, 1_873, 3_528_273
UPDATES_PER_EPOCH, UPDATES, EPOCHS = 820, 3_280, 4
CELLS = (
    "plain_s203201",
    "k1_s203201",
    "k16_s203201",
    "plain_s203202",
    "k1_s203202",
    "k16_s203202",
)
SPEC = {
    "plain_s203201": (203201, 0),
    "k1_s203201": (203201, 1),
    "k16_s203201": (203201, 16),
    "plain_s203202": (203202, 0),
    "k1_s203202": (203202, 1),
    "k16_s203202": (203202, 16),
}
POLL_SECONDS, MAX_SECONDS = 30, 21_600
LOG = logging.getLogger("b300_pr74_publisher")


def _module(path: Path, name: str) -> Any:
    """Load one existing local helper without invoking its command line."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"local helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(HELPER, "next70_report_readers")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _admission(
    path: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify ROOT admission, amended authority, configs and exact split bytes."""
    if _sha(path) != expected_sha:
        raise ValueError("ROOT fit-admission bytes differ from the CLI SHA")
    admission = P._read_json(path, 256_000)
    authority_path, split_path = NB / "AUTHORITY_R2.json", NB / "chronological_primary.json"
    if _sha(authority_path) != AUTHORITY_SHA or _sha(split_path) != SPLIT_SHA:
        raise ValueError("frozen B300 authority or official split bytes changed")
    authority, split = P._read_json(authority_path, 512_000), P._read_json(split_path, 8_000_000)
    profiles, recipes = admission.get("profiles"), authority.get("configs")
    if (
        admission.get("schema") != "wnn.b300-pr74-fourpairs-fit-admission.v1"
        or admission.get("approved") is not True
        or admission.get("max_fresh_fits") != 6
        or admission.get("authority_sha256") != AUTHORITY_SHA
        or admission.get("source_sha256") != SOURCE_SHA
        or admission.get("source_root")
        != "/data/epanov_s/tabdata-causal-stem-b300-20261110-r1/source-r1"
        or admission.get("split_sha256") != SPLIT_SHA
        or authority.get("source", {}).get("commit") != SOURCE_COMMIT
        or authority.get("source", {}).get("source_sha256") != SOURCE_SHA
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
        or not isinstance(recipes, dict)
    ):
        raise ValueError("ROOT admission differs from the frozen six-fit PR74 contract")
    for cell in CELLS:
        profile, recipe = profiles[cell], recipes[cell]
        config_path = NB / "configs" / f"{cell}.json"
        if (
            not isinstance(profile, dict)
            or not isinstance(recipe, dict)
            or not config_path.is_file()
        ):
            raise ValueError(f"admission or local config missing for {cell}")
        config = P._read_json(config_path, 64_000)
        seed, kernel = SPEC[cell]
        if (
            not P._is_sha256(profile.get("config_sha256"))
            or profile.get("raw_config_sha256") != recipe.get("raw_sha256")
            or profile.get("pair") != recipe.get("pair")
            or profile.get("gpu_uuids") != recipe.get("gpu_uuids")
            or config.get("seed") != seed
            or config.get("epochs") != EPOCHS
            or config.get("model", {}).get("input_stem_kernel", 0) != kernel
            or not all(
                P._is_sha256(profile.get(name))
                for name in (
                    "approved_receipt_sha256",
                    "production_receipt_sha256",
                    "peak_receipt_sha256",
                )
            )
        ):
            raise ValueError(f"config/profile receipt binding differs for {cell}")
    admission["admission_sha256"] = expected_sha
    return admission, authority, split


def _members_match(data: dict[str, Any], split: dict[str, Any]) -> bool:
    """Compare exact manifest membership, including the sparse training IDs."""
    fit = split.get("fit", {}).get("train")
    valid = split.get("evaluation", {}).get("valid")
    if not isinstance(fit, list) or not isinstance(valid, list):
        return False
    expected_fit = [["train", index] for index in fit]
    expected_valid = [["valid", index] for index in valid]
    return (
        data.get("fit_members") == expected_fit and data.get("evaluation_members") == expected_valid
    )


def _eval_ok(event: dict[str, Any], data: dict[str, Any], epoch: int) -> tuple[bool, str | None]:
    """Validate one complete official-valid event at its fixed B300 update boundary."""
    metrics, context = event.get("metrics"), event.get("context")
    members = data.get("evaluation_members")
    if (
        not isinstance(metrics, dict)
        or not isinstance(context, dict)
        or not isinstance(members, list)
    ):
        return False, "missing metrics or evaluation context"
    member_sha = hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest()
    if (
        event.get("schema") != "wnn.monitor.v1"
        or event.get("kind") != "evaluation"
        or type(event.get("epoch")) is not int
        or event.get("epoch") != epoch
        or type(event.get("step")) is not int
        or event.get("step") != epoch * UPDATES_PER_EPOCH
        or context.get("split_sha256") != SPLIT_SHA
        or context.get("evaluation_member_sha256") != member_sha
        or context.get("evaluation_member_count") != VALID_N
        or context.get("endpoint") != "raw standalone z; no reference or blend"
        or metrics.get("valid/sequence_count") != VALID_N
        or metrics.get("valid/selected_rows") != VALID_ROWS
        or not all(
            P._finite_number(metrics.get(key)) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1")
        )
    ):
        return False, "epoch boundary, full-valid coverage or identity differs"
    return True, None


def _observe_cell(
    monitor_root: Path, cell: str, admission: dict[str, Any], split: dict[str, Any]
) -> dict[str, Any]:
    """Read one mirror; prefer actual rank-0 update records over heartbeat step snapshots."""
    directory = monitor_root / "live-dashboard" / "native" / f"b300-{cell}-fit"
    view: dict[str, Any] = {"cell": cell, "status": "queued", "complete": False, "terminal": None}
    if not directory.exists():
        return view
    try:
        owner = P._read_json(directory / ".sequence_mirror_owner.json", 8_192)
        collection = P._read_json(directory / "collection_receipt.json", 64_000)
        expected_dir = f"{admission['source_root'].rsplit('/source-r1', 1)[0]}/runs/{cell}/fit"
        if any(
            record.get("host") != "b300" or record.get("source_directory") != expected_dir
            for record in (owner, collection)
        ):
            raise ValueError("native mirror owner/collection path differs from the admitted cell")
        metadata = P._read_json(directory / "run_metadata.json", 1_000_000)
        config, source, provenance = (
            metadata.get("config"),
            metadata.get("source"),
            metadata.get("provenance"),
        )
        if not all(isinstance(value, dict) for value in (config, source, provenance)):
            raise ValueError("run metadata lacks config/source/provenance")
        profile = admission["profiles"][cell]
        authority_cell = SPEC[cell]
        seed, kernel = authority_cell
        model = config.get("model")
        data = provenance.get("data")
        runtime = provenance.get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, dict) else None
        precision = runtime.get("precision") if isinstance(runtime, dict) else None
        if (
            P.canonical_sha256(config) != profile["config_sha256"]
            or source.get("sha256") != SOURCE_SHA
            or config.get("seed") != seed
            or config.get("epochs") != EPOCHS
            or config.get("objective_mode") != "standalone_labels_wmse_v2"
            or config.get("required_device") != "B300"
            or config.get("required_world_size") != 2
            or config.get("precision_mode") != "bf16_bf16x9"
            or config.get("checkpoint_endpoint_policy") != "configured_final_epoch"
            or config.get("normalization_scope") != "all_fit"
            or config.get("training_mask") != "all_after_warmup"
            or not isinstance(model, dict)
            or model.get("architecture") != "transformer"
            or model.get("hidden") != 768
            or model.get("layers") != 24
            or model.get("input_stem_kernel", 0) != kernel
            or not isinstance(data, dict)
            or provenance.get("data_sha256") != P.canonical_sha256(data)
            or data.get("split_sha256") != SPLIT_SHA
            or data.get("teacher") is not None
            or not isinstance(data.get("normalization"), dict)
            or data["normalization"].get("scope") != "all_fit"
            or data["normalization"].get("sequence_count") != TRAIN_N
            or data.get("training_mask") != "all_after_warmup"
            or not _members_match(data, split)
            or not isinstance(devices, list)
            or [str(item.get("uuid", "")).removeprefix("GPU-") for item in devices]
            != profile["gpu_uuids"]
            or not isinstance(precision, dict)
            or precision.get("mode") != "bf16_bf16x9"
        ):
            raise ValueError("fit metadata differs from admitted source/config/split/runtime")
        view.update(status="running", identity_ok=True, metadata=metadata, data=data)
        progress_path = directory / "monitor_progress.json"
        progress = P._read_json(progress_path, 16_384) if progress_path.exists() else {}
        rank_path = directory / "rank0.jsonl"
        rank = P._read_tail_jsonl(rank_path) if rank_path.exists() else None
        view["phase"] = progress.get("phase", "unknown")
        view["snapshot_step"] = progress.get("step")
        view["rank0"] = rank
        view["rank0_mtime"] = rank_path.stat().st_mtime if rank_path.exists() else None
        observer_path = directory / "monitor_observer.jsonl"
        view["observer"] = P._read_tail_jsonl(observer_path) if observer_path.exists() else None
        events, event_error = P._read_evaluations(directory / "monitor_evaluations.jsonl")
        checked = []
        for event in events:
            epoch = event.get("epoch")
            ok, error = (
                _eval_ok(event, data, epoch) if type(epoch) is int else (False, "invalid epoch")
            )
            checked.append(
                {"epoch": epoch, "ok": ok, "event": event if ok else None, "error": error}
            )
        view["evaluations"], view["evaluation_error"] = checked, event_error
        terminal_path = directory / "terminal_status.json"
        if terminal_path.exists():
            terminal = P._read_json(terminal_path, 2_000_000)
            identity = terminal.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("source_sha256") != SOURCE_SHA
                or identity.get("config_sha256") != profile["config_sha256"]
                or identity.get("data_sha256") != provenance.get("data_sha256")
                or identity.get("world_size") != 2
            ):
                raise ValueError("terminal identity differs from admitted fit")
            if terminal.get("status") == "completed" and (
                terminal.get("completed_epochs") != EPOCHS
                or terminal.get("global_updates") != UPDATES
                or not P._is_sha256(terminal.get("model_sha256"))
            ):
                raise ValueError("terminal is not the fixed-E4 endpoint")
            view["terminal"] = {
                "status": terminal.get("status"),
                "model_sha256": terminal.get("model_sha256"),
                "sha": _sha(terminal_path),
            }
        view["complete"] = (
            isinstance(view["terminal"], dict)
            and view["terminal"].get("status") == "completed"
            and [item.get("epoch") for item in checked] == [1, 2, 3, 4]
            and all(item.get("ok") for item in checked)
            and checked[-1]["event"].get("step") == UPDATES
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        view["status"], view["error"] = "unverified", str(error)
    return view


def _bridge(root: Path) -> dict[str, Any]:
    """Report local transport freshness without rewriting native timestamps."""
    now = time.time()
    result: dict[str, Any] = {"now": now, "age": None, "errors": []}
    try:
        bridge = P._read_json(root / "live-dashboard" / "bridge-state.json", 8_000_000)
        host = bridge.get("hosts", {}).get("b300", {})
        applied = host.get("last_applied_unix")
        if P._finite_number(applied):
            age = now - float(applied)
            result["age"] = age if age >= 0 else None
        errors = host.get("last_errors", [])
        result["errors"] = errors if isinstance(errors, list) else []
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return result


def _metric(cell: dict[str, Any], name: str) -> float | None:
    events = cell.get("evaluations", [])
    for item in reversed(events):
        if item.get("ok") and isinstance(item.get("event"), dict):
            value = item["event"].get("metrics", {}).get(name)
            return float(value) if P._finite_number(value) else None
    return None


def _report(
    cells: list[dict[str, Any]], bridge: dict[str, Any], admission: dict[str, Any], title: str
) -> str:
    now = (
        datetime.fromtimestamp(bridge["now"], UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    age = bridge.get("age")
    age_text = f"{float(age):.0f}s" if P._finite_number(age) else "unknown"
    lines = [
        f"## PR74 B300 causal input-stem — {title}",
        f"Local observation {now}; mirror age {age_text}; native timestamps unchanged.",
        f"Admission `{admission['admission_sha256']}`; source `{SOURCE_SHA}` ({SOURCE_COMMIT}); split `{SPLIT_SHA}`.",
        "Six fresh fixed-E4 fits, world size 2, BF16/BF16x9; exact sparse official training membership from the frozen manifest.",
        "Update count comes from the latest complete rank0.jsonl record; monitor_progress.step is phase-snapshot metadata only.",
        "",
        "### Cell status: phase, actual update, loss, preclip gradient, latest full-valid WP",
    ]
    for cell in cells:
        name = cell["cell"]
        profile = admission["profiles"][name]
        pair = ",".join(str(x) for x in profile["pair"])
        seed, kernel = SPEC[name]
        if cell["status"] == "queued":
            lines.append(
                f"- `{name}` GPUs {pair}, K{kernel}, seed {seed}: admitted/queued; native fit mirror not created yet."
            )
            continue
        if cell["status"] == "unverified":
            lines.append(
                f"- `{name}`: identity unverified; metrics withheld ({cell.get('error', 'incomplete mirror')})."
            )
            continue
        rank, observer = cell.get("rank0"), cell.get("observer")
        step = (
            rank.get("step") if isinstance(rank, dict) and type(rank.get("step")) is int else None
        )
        loss = rank.get("total_loss") if isinstance(rank, dict) else None
        grad = rank.get("gradient_norm_preclip") if isinstance(rank, dict) else None
        clipped = rank.get("gradient_clipped", "unknown") if isinstance(rank, dict) else "unknown"
        phase = cell.get("phase", "unknown")
        mtime = cell.get("rank0_mtime")
        log_time = (
            datetime.fromtimestamp(float(mtime), UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
            if P._finite_number(mtime)
            else "unknown"
        )
        valid = _metric(cell, "valid/wp")
        t0, t1 = _metric(cell, "valid/wp/t0"), _metric(cell, "valid/wp/t1")
        score = (
            f"t0/t1/wp {t0:.6f}/{t1:.6f}/{valid:.6f}"
            if t0 is not None and t1 is not None and valid is not None
            else "full-valid pending"
        )
        train_loss = f"{float(loss):.6f}" if P._finite_number(loss) else "unknown"
        grad_text = f"{float(grad):.6f}" if P._finite_number(grad) else "unknown"
        update = (
            f"{step:,}/{UPDATES:,} from rank0.jsonl"
            if step is not None
            else f"unknown/{UPDATES:,} from rank0.jsonl"
        )
        snapshot = cell.get("snapshot_step")
        snapshot_text = (
            f"snapshot step {snapshot}" if type(snapshot) is int else "snapshot step unknown"
        )
        gpu_text = ""
        if isinstance(observer, dict) and isinstance(observer.get("metrics"), dict):
            metrics = observer["metrics"]
            util = [metrics.get(f"gpu/device_{gpu}/utilization_pct") for gpu in profile["pair"]]
            if all(P._finite_number(value) for value in util):
                gpu_text = f"; assigned GPU util {util[0]:.0f}/{util[1]:.0f}%"
        terminal = cell.get("terminal")
        terminal_text = ""
        if isinstance(terminal, dict):
            model_sha = terminal.get("model_sha256")
            model_text = (
                f"trainer model SHA `{model_sha[:12]}…`, byte verification is ROOT-side"
                if isinstance(model_sha, str) and P._is_sha256(model_sha)
                else "model SHA unavailable for incomplete terminal"
            )
            terminal_text = f"; {terminal.get('status', 'unknown')} ({model_text})"
        lines.append(
            f"- `{name}` GPUs {pair}, K{kernel}, seed {seed}: {phase}; {update}; {snapshot_text}; loss {train_loss}; preclip grad {grad_text}, clipped {clipped}; rank0 log mtime {log_time}; {score}{gpu_text}{terminal_text}."
        )
    if bridge.get("errors"):
        latest = bridge["errors"][-1]
        lines.append(
            f"\nLatest bridge error entry: `{str(latest)[:300]}`; older last_rpc_error is not used as freshness evidence."
        )
    lines.append("\n### Fixed-E4 paired score differences")
    complete = len(cells) == 6 and all(cell.get("complete") for cell in cells)
    if complete:
        by_name = {cell["cell"]: cell for cell in cells}
        for comparator, label in (("k1", "K16−K1"), ("plain", "K16−plain")):
            for metric, target in (
                ("valid/wp/t0", "t0"),
                ("valid/wp/t1", "t1"),
                ("valid/wp", "wp"),
            ):
                deltas = []
                for seed in (203201, 203202):
                    left = _metric(by_name[f"k16_s{seed}"], metric)
                    right = _metric(by_name[f"{comparator}_s{seed}"], metric)
                    deltas.append(float(left) - float(right))
                lines.append(
                    f"- {label} {target}: seed203201 {deltas[0]:+.6f}; seed203202 {deltas[1]:+.6f}; descriptive mean {sum(deltas) / 2:+.6f}."
                )
    else:
        lines.append(
            "Withheld until all six fixed-E4 terminals and all four exact full-valid events per fit pass."
        )
    lines.append(
        "Scores use the repeatedly exposed official validation set (1,873 sequences / 3,528,273 rows); no fitted blend, significance, or public-score claim."
    )
    return "\n".join(lines)


class _PR74Client:
    """Adapt the existing P77 marker helper to ROOT's sole allowlisted PR74 client."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def find_report(self, _number: int, marker: str, prepared_at: float) -> Any:
        return self.client.find_report(74, marker, prepared_at)

    def post_report(self, _number: int, body: str, *, before_post: Any) -> Any:
        return self.client.post_report(74, body, before_post=before_post)


def run(args: argparse.Namespace) -> int:
    """Poll verified mirrors and publish 30-minute, terminal and final updates."""
    if not 1 <= args.maxseconds <= MAX_SECONDS:
        raise ValueError("--maxseconds must be in [1, 21600]")
    admission, _authority, split = _admission(
        args.root / "ROOT-fit-admission-r1.json", args.admission_sha256
    )
    deadline = time.monotonic() + args.maxseconds
    client = None
    if not args.dryrun:
        github = _module(CLIENT_PATH, "sequence_progress_github_pr74").ProgressGitHub(
            {74}, deadline=deadline
        )
        client = _PR74Client(github)
    state = args.state_file or args.root / "reporting" / "publisher-state-pr74.json"
    while True:
        cells = [_observe_cell(args.monitor_root, cell, admission, split) for cell in CELLS]
        bridge = _bridge(args.root)
        bucket = datetime.fromtimestamp(int(bridge["now"] // 1800) * 1800, UTC).strftime(
            "%Y%m%dT%H%M"
        )
        due = [(f"<!-- wnn-b300-pr74-progress-30m-{bucket}Z -->", "30-minute progress")]
        terminal_cells = [cell for cell in cells if isinstance(cell.get("terminal"), dict)]
        for cell in terminal_cells:
            terminal = cell["terminal"]
            due.append(
                (
                    f"<!-- wnn-b300-pr74-terminal-{cell['cell']}-{terminal['sha'][:16]} -->",
                    f"terminal {cell['cell']}",
                )
            )
        all_terminal = len(terminal_cells) == len(CELLS)
        complete = all_terminal and all(cell.get("complete") for cell in cells)
        if all_terminal:
            digest = P.canonical_sha256(
                [(cell["cell"], cell["terminal"]["sha"]) for cell in cells]
            )[:16]
            due.append(
                (
                    f"<!-- wnn-b300-pr74-cohort-{digest} -->",
                    "fixed-E4 cohort complete"
                    if complete
                    else "cohort terminal; endpoint incomplete",
                )
            )
        delivered = True
        for marker, title in due:
            body = _report(cells, bridge, admission, title) + f"\n\n{marker}"
            if args.dryrun:
                logging.info("%s", body)
                continue
            try:
                outcome = P.publish_once(client, state, marker, body, bridge["now"])
                if outcome not in {"posted", "reconciled-existing", "already-posted"}:
                    delivered = False
                    LOG.warning("%s: unconfirmed outcome %r", marker, outcome)
            except Exception as error:
                delivered = False
                LOG.warning("%s: %s", marker, type(error).__name__)
        if args.once or (all_terminal and delivered):
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        time.sleep(min(POLL_SECONDS, remaining))


def main() -> int:
    """Run one local snapshot or ROOT's bounded credentialed publisher."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=NB)
    parser.add_argument("--monitor-root", type=Path, default=OLDN)
    parser.add_argument("--admission-sha256", default=ADMISSION_SHA)
    parser.add_argument("--maxseconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--state-file", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dryrun", action="store_true", help="read local mirrors and print only")
    modes.add_argument("--once", action="store_true", help="publish one snapshot and exit")
    args = parser.parse_args()
    if args.dryrun:
        args.once = True
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
