"""Publish bounded local progress for six ROOT-admitted L4 causal-stem fits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # The existing local helper/client APIs are dynamically loaded.

PACKET = Path(__file__).resolve().parents[1]
NATIVE = "/data/lc-epanov_s/tabdata-stem-20261110-r1"
SOURCE = f"{NATIVE}/source-r1"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
ARMS = (("plain", 0), ("k1", 1), ("k16", 16))
SEEDS = (203301, 203302)
CELLS = tuple(f"{arm}_s{seed}" for arm, _ in ARMS for seed in SEEDS)
SPEC = {
    f"{arm}_s{seed}": (seed, kernel, gpu)
    for gpu, (arm, kernel, seed) in enumerate(((a, k, s) for a, k in ARMS for s in SEEDS))
}
UPDATES, EPOCH_UPDATES, NSEQS, NROWS = 38_760, 6_460, 1_873, 3_528_273
MAX_SECONDS, POLL = 14_400, 30
HELPER = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\publish_pr77.py"
)
CLIENT = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py"
)
LOG = logging.getLogger("l4_stem_publisher")


def _module(path: Path, name: str) -> Any:
    """Load a local helper module without invoking its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"local helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(HELPER, "next70_local_readers")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _admission(path: Path, pinned: str, configs: Path) -> dict[str, object]:
    """Bind ROOT's six-cell admission and its config bytes before reading metrics."""
    if _sha(path) != pinned:
        raise ValueError("ROOT admission SHA differs from the CLI pin")
    value = P._read_json(path, 256_000)
    profiles = value.get("profiles")
    if (
        value.get("schema") != "wnn.l4-stem-root-fit-admission.v1"
        or value.get("approved") is not True
        or value.get("max_fresh_fits") != 6
        or value.get("source_root") != SOURCE
        or value.get("split_sha256") != SPLIT
        or not P._is_sha256(value.get("source_sha256"))
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("ROOT admission differs from the frozen six-cell contract")
    uuids = set()
    for cell, (_, _, gpu) in SPEC.items():
        profile, config = profiles[cell], configs / f"{cell}.json"
        if (
            not isinstance(profile, dict)
            or not config.is_file()
            or profile.get("gpu") != gpu
            or profile.get("raw_config_sha256") != _sha(config)
            or not P._is_sha256(profile.get("config_sha256"))
            or not P._is_sha256(profile.get("receipt_sha256"))
            or not profile.get("receipt_path")
        ):
            raise ValueError(f"invalid profile binding for {cell}")
        uuid = profile.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid or uuid in uuids:
            raise ValueError("ROOT admission lacks six unique GPU UUIDs")
        uuids.add(uuid)
    value["admission_sha256"] = pinned
    return value


def inspect_cell(monitor: Path, name: str, admission: dict[str, object]) -> dict[str, object]:
    """Check source/config/split identity and read one bounded mirror cell."""
    path = monitor / "live-dashboard" / "native" / f"l4-{name}-fit"
    cell: dict[str, object] = {"cell": name, "ok": False, "events": [], "terminal": None}
    try:
        profile = admission["profiles"][name]
        owner = P._read_json(path / ".sequence_mirror_owner.json", 8192)
        if (
            owner.get("host") != "l4"
            or owner.get("source_directory") != f"{NATIVE}/runs/{name}/fit"
        ):
            raise ValueError("mirror owner path mismatch")
        meta = P._read_json(path / "run_metadata.json", 1_000_000)
        config, source, prov = meta.get("config"), meta.get("source"), meta.get("provenance")
        if (
            not isinstance(config, dict)
            or not isinstance(source, dict)
            or not isinstance(prov, dict)
        ):
            raise ValueError("run metadata is incomplete")
        seed, kernel, _ = SPEC[name]
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            raise ValueError("run model config is incomplete")
        effective_kernel = model_config.get("input_stem_kernel")
        if effective_kernel is None:
            effective_kernel = 0
        if (
            P.canonical_sha256(config) != profile["config_sha256"]
            or source.get("sha256") != admission["source_sha256"]
            or config.get("seed") != seed
            or effective_kernel != kernel
        ):
            raise ValueError("run source/config differs from ROOT admission")
        data = prov.get("data")
        if (
            not isinstance(data, dict)
            or prov.get("data_sha256") != P.canonical_sha256(data)
            or data.get("split_sha256") != SPLIT
            or data.get("teacher") is not None
        ):
            raise ValueError("data provenance or split differs from admission")
        fit, valid = data.get("fit_members"), data.get("evaluation_members")
        if (
            not isinstance(fit, list)
            or len(fit) != 10_325
            or any(not isinstance(row, list) or len(row) != 2 or row[0] != "train" for row in fit)
            or len({tuple(row) for row in fit}) != 10_325
            or valid != [["valid", i] for i in range(NSEQS)]
        ):
            raise ValueError("fit/full-valid membership differs from official split")
        cell["ok"] = True
        progress = P._read_json(path / "monitor_progress.json", 16_384)
        cell["phase"], cell["epoch"] = progress.get("phase", "unknown"), progress.get("epoch")
        step = progress.get("last_successful_update", progress.get("step"))
        cell["step"] = step if type(step) is int and step >= 0 else None
        rank = P._read_tail_jsonl(path / "rank0.jsonl") or {}
        cell["train"] = {
            "loss": rank.get("total_loss"),
            "grad": rank.get("gradient_norm_preclip"),
            "clip": rank.get("gradient_clipped"),
        }
        events, _ = P._read_evaluations(path / "monitor_evaluations.jsonl")
        checked = []
        for event in events:
            epoch = event.get("epoch")
            ok, error = (
                P._validate_eval_event(event, data, epoch)
                if type(epoch) is int
                else (False, "invalid epoch")
            )
            if ok and event.get("step") != epoch * EPOCH_UPDATES:
                ok, error = False, "update boundary mismatch"
            checked.append(
                {"epoch": epoch, "ok": ok, "event": event if ok else None, "error": error}
            )
        cell["events"] = checked
        terminal_path = path / "terminal_status.json"
        if terminal_path.exists():
            terminal = P._read_json(terminal_path, 2_000_000)
            identity = terminal.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("source_sha256") != admission["source_sha256"]
                or identity.get("config_sha256") != profile["config_sha256"]
                or identity.get("data_sha256") != prov.get("data_sha256")
            ):
                raise ValueError("terminal identity mismatch")
            if terminal.get("status") == "completed" and (
                terminal.get("completed_epochs") != 6
                or terminal.get("global_updates") != UPDATES
                or not P._is_sha256(terminal.get("model_sha256"))
            ):
                raise ValueError("terminal is not fixed6E")
            cell["terminal"] = {
                "status": terminal.get("status"),
                "model_sha256": terminal.get("model_sha256"),
                "sha": _sha(terminal_path),
            }
        events = cell["events"]
        cell["complete"] = (
            isinstance(cell["terminal"], dict)
            and cell["terminal"].get("status") == "completed"
            and [event.get("epoch") for event in events] == [1, 2, 3, 4, 5, 6]
            and all(event.get("ok") for event in events)
            and events[-1]["event"].get("step") == UPDATES
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


def _score(cell: dict[str, object]) -> dict[str, float] | None:
    valid = [event for event in cell.get("events", []) if event.get("ok")]
    metrics = valid[-1]["event"].get("metrics") if valid else None
    if not isinstance(metrics, dict):
        return None
    scores = {f"t{i}": metrics.get(f"valid/wp/t{i}") for i in (0, 1)}
    return scores if all(P._finite_number(x) for x in scores.values()) else None


def _report(
    admission: dict[str, object],
    cells: list[dict[str, object]],
    clock: dict[str, object],
    pr: int,
    title: str,
) -> str:
    now = (
        datetime.fromtimestamp(float(clock["local_now"]), UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    age = clock.get("age_seconds")
    freshness = f"{float(age):.0f}s" if P._finite_number(age) else "unknown"
    lines = [
        f"## L4 GRU causal stem PR{pr} — {title}",
        f"Local observation {now}; mirror age {freshness}; native timestamps unchanged.",
        f"Admission `{admission['admission_sha256']}`; source `{admission['source_sha256']}`; split `{SPLIT}`.",
        "",
        "### Six fits: phase, step, loss, preclip grad, latest full-valid WP t0/t1",
    ]
    for cell in cells:
        name = str(cell["cell"])
        if not cell.get("ok"):
            lines.append(
                f"- `{name}` identity unverified; metrics withheld ({cell.get('error', 'mirror incomplete')})."
            )
            continue
        seed, kernel, gpu = SPEC[name]
        train, terminal = cell.get("train", {}), cell.get("terminal")
        phase = (
            terminal.get("status")
            if isinstance(terminal, dict)
            else f"{cell.get('phase', 'unknown')} {cell.get('epoch', '?')}/6E"
        )
        score = _score(cell)
        valid = (
            f"E{cell['events'][-1]['epoch']} {score['t0']:.6f}/{score['t1']:.6f}"
            if score
            else "pending"
        )
        loss = f"{train['loss']:.6f}" if P._finite_number(train.get("loss")) else "unknown"
        grad = f"{train['grad']:.6f}" if P._finite_number(train.get("grad")) else "unknown"
        model = (
            f"; trainer model SHA `{terminal['model_sha256'][:12]}…` (not byte-verified)"
            if isinstance(terminal, dict) and P._is_sha256(terminal.get("model_sha256"))
            else ""
        )
        lines.append(
            f"- `{name}` GPU{gpu}, K{kernel}, seed{seed}: {phase}; {cell.get('step', '?')}/38,760; loss {loss}; preclip grad {grad}, clipped {train.get('clip', 'unknown')}; valid {valid}{model}"
        )
    lines.append("\n### Fixed-E6 paired target deltas")
    complete = len(cells) == 6 and all(cell.get("complete") for cell in cells)
    if complete:
        by_name = {cell["cell"]: cell for cell in cells}
        for left, right, label in (
            ("k16", "k1", "K16−K1"),
            ("k16", "plain", "K16−K0"),
            ("k1", "plain", "K1−K0"),
        ):
            for target in ("t0", "t1"):
                delta = [
                    float(_score(by_name[f"{left}_s{s}"])[target])
                    - float(_score(by_name[f"{right}_s{s}"])[target])
                    for s in SEEDS
                ]
                lines.append(
                    f"- {label} {target}: seed203301 {delta[0]:+.6f}; seed203302 {delta[1]:+.6f}; mean {sum(delta) / 2:+.6f}."
                )
    else:
        lines.append("Withheld until all six fixed-E6 terminals and exact full-valid events pass.")
    lines.append(
        "Complete-model target scores, no blend/routing. Validation is repeatedly exposed; no significance/public-score claim."
    )
    return "\n".join(lines)


class _PRClient:
    """Bind PR77's idempotent local publisher to ROOT's single selected PR."""

    def __init__(self, client: Any, number: int) -> None:
        self.client, self.number = client, number

    def find_report(self, _number: int, marker: str, when: float) -> dict[str, object] | None:
        return self.client.find_report(self.number, marker, when)

    def post_report(self, _number: int, body: str, *, before_post: Any) -> dict[str, object]:
        return self.client.post_report(self.number, body, before_post=before_post)


def run(args: argparse.Namespace) -> int:
    """Poll only local mirrors and publish half-hour, per-cell, and cohort updates."""
    if not 1 <= args.maxseconds <= MAX_SECONDS or args.pr <= 0:
        raise ValueError("invalid --maxseconds or --pr")
    configs = args.config_root or PACKET / "configs"
    admission = _admission(args.profile_admission, args.admission_sha256, configs)
    deadline = time.monotonic() + args.maxseconds
    client = (
        None
        if args.dryrun
        else _PRClient(
            _module(CLIENT, "sequence_progress_github").ProgressGitHub(
                {args.pr}, deadline=deadline
            ),
            args.pr,
        )
    )
    state = args.state_file or args.root / "reporting" / f"publisher-state-pr{args.pr}.json"
    while True:
        cells = [inspect_cell(args.monitor_root, name, admission) for name in CELLS]
        clock = P._source_clock(args.root)
        now = float(clock["local_now"])
        bucket = datetime.fromtimestamp(int(now // 1800) * 1800, UTC).strftime("%Y%m%dT%H%M")
        due = [(f"<!-- wnn-l4-stem-pr{args.pr}-30m-{bucket}Z -->", "30-minute progress")]
        for cell in cells:
            terminal = cell.get("terminal")
            if isinstance(terminal, dict):
                marker = f"<!-- wnn-l4-stem-pr{args.pr}-terminal-{cell['cell']}-{terminal['sha'][:16]} -->"
                due.append((marker, f"terminal {cell['cell']}"))
        complete = len(cells) == 6 and all(cell.get("complete") for cell in cells)
        failed = (
            len(cells) == 6
            and all(isinstance(cell.get("terminal"), dict) for cell in cells)
            and any(cell["terminal"].get("status") != "completed" for cell in cells)
        )
        if complete or failed:
            marker = f"<!-- wnn-l4-stem-pr{args.pr}-cohort-{P.canonical_sha256([c['terminal']['sha'] for c in cells])[:16]} -->"
            due.append(
                (
                    marker,
                    "fixed-E6 cohort complete"
                    if complete
                    else "cohort terminal; endpoint incomplete",
                )
            )
        delivered = True
        for marker, title in due:
            body = _report(admission, cells, clock, args.pr, title) + f"\n\n{marker}"
            if args.dryrun:
                logging.info("%s", body)
            else:
                try:
                    outcome = P.publish_once(client, state, marker, body, now)
                    if outcome not in {"posted", "reconciled-existing", "already-posted"}:
                        delivered = False
                        LOG.warning("%s: unconfirmed publish outcome %r", marker, outcome)
                except Exception as error:  # The fixed progress client sanitizes API failures.
                    delivered = False
                    LOG.warning("%s: %s", marker, type(error).__name__)
        if args.once or ((complete or failed) and delivered):
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        time.sleep(min(POLL, remaining))


def main() -> int:
    """Run local dry-run reporting or ROOT's bounded credentialed publisher."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--monitor-root", type=Path, required=True)
    cli.add_argument("--profile-admission", type=Path, required=True)
    cli.add_argument("--admission-sha256", required=True)
    cli.add_argument("--config-root", type=Path, default=PACKET / "configs")
    cli.add_argument("--pr", type=int, required=True)
    cli.add_argument("--maxseconds", type=int, default=MAX_SECONDS)
    cli.add_argument("--state-file", type=Path)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--dryrun", action="store_true", help="read local evidence and print only")
    modes.add_argument("--once", action="store_true", help="publish one snapshot and exit")
    args = cli.parse_args()
    if args.dryrun:
        args.once = True
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
