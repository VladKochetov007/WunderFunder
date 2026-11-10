"""Publish bounded local progress and fixed-endpoint comparisons for PR78."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SOURCE = "b917b83edd463760d83829e44f87b57af5473507302e558d36fad3d1034cd01b"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
REMOTE = "/data/lc-epanov_s/tabdata-next70-20261110-r1"
CLIENT = Path(r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py")
PR77 = Path(__file__).resolve().parents[2] / "l4-next70-20261110" / "reporting" / "publish_pr77.py"
STAGES = {
    "ssl_pretrain_s203201": (2, 203201, 6, "004cdb0f-3bf7-0003-702e-0e2fb87215e8"),
    "ssl_pretrain_s203202": (2, 203202, 7, "cd069633-c451-769e-3727-cb1fe310e407"),
    "ssl_finetune_s203201": (4, 203201, 6, "004cdb0f-3bf7-0003-702e-0e2fb87215e8"),
    "ssl_finetune_s203202": (4, 203202, 7, "cd069633-c451-769e-3727-cb1fe310e407"),
}
UPDATES, NVALID, ROWS, MAX_SECONDS, POLL = 6460, 1873, 3_528_273, 43_200, 30
STATE_SCHEMA = "wnn.l4-input-dynamics-pr78-publisher-state.v1"

def _module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"local reporting module is unavailable: {name}")  # noqa: E701
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

P77 = _module("l4_pr77_progress", PR77)
_json, _finite, _sha, _digest = P77._read_json, P77._finite_number, P77._is_sha256, P77.canonical_sha256

def _fit(root: Path, name: str, source: str, config_sha: str, epochs: int, seed: int, uuid: str | None, pretrain: bool = False) -> dict[str, object]:
    directory = root / "live-dashboard" / "native" / f"l4-{name}-fit"
    result: dict[str, object] = {"name": name, "identity_ok": False, "evaluations": []}
    try:
        owner = _json(directory / ".sequence_mirror_owner.json", 8192)
        collected = _json(directory / "collection_receipt.json", 64_000)
        remote = f"{REMOTE}/runs/{name}/fit"
        if any(x.get("host") != "l4" or x.get("source_directory") != remote for x in (owner, collected)):
            raise ValueError("mirror owner/collection path differs")
        meta = _json(directory / "run_metadata.json")
        cfg, src, prov = meta.get("config"), meta.get("source"), meta.get("provenance")
        if not all(isinstance(x, dict) for x in (cfg, src, prov)): raise ValueError("run metadata lacks config, source, or provenance")  # noqa: E701
        data = prov.get("data")
        if not isinstance(data, dict): raise ValueError("run data provenance is incomplete")  # noqa: E701
        members, valid = data.get("fit_members"), data.get("evaluation_members")
        runtime = prov.get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, dict) else None
        if (
            _digest(cfg) != config_sha or src.get("sha256") != source
            or prov.get("data_sha256") != _digest(data) or data.get("split_sha256") != SPLIT
            or data.get("teacher") is not None or prov.get("world_size") != 1
            or cfg.get("seed") != seed or cfg.get("epochs") != epochs
            or not isinstance(members, list) or len(members) != 10_325
            or any(not isinstance(x, list) or len(x) != 2 or x[0] != "train" for x in members)
            or len({tuple(x) for x in members}) != 10_325
            or valid != [["valid", i] for i in range(NVALID)]
            or not isinstance(devices, list) or len(devices) != 1
            or devices[0].get("name") != "NVIDIA L4"
            or (uuid is not None and devices[0].get("uuid") != uuid)
        ):
            raise ValueError("admitted source/config/fit-only/runtime identity differs")
        if pretrain:
            pre = data.get("future_input_innovations")
            if (
                not isinstance(pre, dict) or pre.get("fit_sequence_count") != 10_325
                or pre.get("validation_members_used") is not False or pre.get("label_values_read") is not False
                or cfg.get("pretraining") != {"kind": "future_input_innovations_v1", "horizons": [16, 64, 256]}
            ):
                raise ValueError("pretext ancestry is not train-only and label-free")
        elif cfg.get("objective_mode") != "standalone_labels_wmse_v2" or cfg.get("supervised_target") is not None:
            raise ValueError("child is not the shared two-target recipe")
        result.update(identity_ok=True, data=data)
        progress = _json(directory / "monitor_progress.json", 16_384)
        step = progress.get("last_successful_update", progress.get("step"))
        result.update(phase=progress.get("phase", "unknown"), step=step if type(step) is int else None, epoch=progress.get("epoch"))
        row = P77._read_tail_jsonl(directory / "rank0.jsonl")
        if row:
            result["train"] = {k: row.get(k) for k in ("step", "hint_loss", "pretext_loss", "learning_rate", "gradient_norm_preclip", "gradient_clipped")}
        events, journal_error = P77._read_evaluations(directory / "monitor_evaluations.jsonl")
        if journal_error and not (pretrain and journal_error == "evaluation journal is missing"): raise ValueError(journal_error)  # noqa: E701
        if pretrain and events:
            raise ValueError("pretext parent unexpectedly has task-validation events")
        checked = [{"ok": P77._validate_eval_event(e, data, i)[0], "event": e} for i, e in enumerate(events, 1)]
        result["evaluations"] = checked
        terminal_path = directory / "terminal_status.json"
        if terminal_path.exists():
            terminal = _json(terminal_path, 2_000_000)
            tid = terminal.get("identity")
            expected = {"source_sha256": source, "config_sha256": config_sha, "data_sha256": prov["data_sha256"], "world_size": 1}
            if not isinstance(tid, dict) or any(tid.get(k) != v for k, v in expected.items()): raise ValueError("terminal identity differs from run metadata/admission")  # noqa: E701
            result["terminal"] = {
                "status": terminal.get("status"), "completed_epochs": terminal.get("completed_epochs"),
                "global_updates": terminal.get("global_updates"), "model_sha256": terminal.get("model_sha256"),
                "sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(), "native_byte_verified": False,
            }
        terminal = result.get("terminal")
        terminal_ok = (
            isinstance(terminal, dict) and terminal.get("status") == "completed"
            and terminal.get("completed_epochs") == epochs and terminal.get("global_updates") == epochs * UPDATES
            and _sha(terminal.get("model_sha256"))
        )
        result["endpoint_ok"] = bool(terminal_ok and (pretrain or len(checked) == epochs and all(x["ok"] for x in checked)))
        if not pretrain and isinstance(data.get("initializer"), dict):
            result["parent_model_sha256"] = data["initializer"].get("checkpoint_sha256")
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        result["identity_ok"] = False
        result["error"] = str(error)
        return result

def _pr77_controls(root: Path) -> dict[int, dict[str, object]]:
    admission = P77.load_admission(root)
    controls = {}
    for seed in (203201, 203202):
        cell = P77.inspect_cell(root, f"shared_control_s{seed}", admission)
        ev = cell.get("evaluations", [])
        controls[seed] = ev[-1]["event"]["metrics"] if cell.get("endpoint_ok") and ev and ev[-1].get("ok") else {}
    return controls

def _snapshot(root: Path) -> dict[str, object]:
    adm = _json(root / "ROOT-ssl-conditional-admission-r2.json", 32_000)
    if adm.get("schema") != "wnn.l4-next70-ssl-conditional-admission.v1" or adm.get("approved") is not True or adm.get("source_sha256") != SOURCE or adm.get("split_sha256") != SPLIT or adm.get("physical_gpus") != [6, 7]:
        raise ValueError("PR78 conditional admission differs from the frozen contract")
    configs = adm.get("configs")
    if not isinstance(configs, dict) or set(configs) != set(STAGES): raise ValueError("admission does not bind exactly four stages")  # noqa: E701
    stages = []
    for name, (epochs, seed, gpu, uuid) in STAGES.items():
        entry = configs[name]
        if not isinstance(entry, dict) or not _sha(entry.get("normalized_sha256")): raise ValueError(f"normalized config hash missing: {name}")  # noqa: E701
        stage = _fit(root, name, SOURCE, entry["normalized_sha256"], epochs, seed, uuid, pretrain=epochs == 2)
        stage.update(epochs_expected=epochs, gpu=gpu)
        stages.append(stage)
    try:
        host = _json(root / "live-dashboard" / "bridge-state.json", 8_000_000)["hosts"]["l4"]
        applied = host.get("last_applied_unix")
        clock = {"now": time.time(), "age": max(0.0, time.time() - float(applied)) if _finite(applied) else None, "errors": host.get("last_errors", [])}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        clock = {"now": time.time(), "age": None, "errors": []}
    return {"admission": adm, "stages": stages, "clock": clock}

def _comparison(stages: list[dict[str, object]], controls: dict[int, dict[str, object]]) -> list[dict[str, object]] | None:
    by_name = {str(x["name"]): x for x in stages}
    rows = []
    for seed in (203201, 203202):
        parent, child = by_name[f"ssl_pretrain_s{seed}"], by_name[f"ssl_finetune_s{seed}"]
        baseline = controls.get(seed)
        if not parent.get("endpoint_ok") or not child.get("endpoint_ok") or child.get("parent_model_sha256") != parent.get("terminal", {}).get("model_sha256") or not baseline:
            return None
        cm, bm = child["evaluations"][-1]["event"]["metrics"], baseline
        for target in (0, 1):
            key = f"valid/wp/t{target}"
            rows.append({"seed": seed, "target": f"t{target}", "child": float(cm[key]), "control": float(bm[key]), "delta": float(cm[key]) - float(bm[key])})
        rows.append({"seed": seed, "target": "mean", "child": float(cm["valid/wp"]), "control": float(bm["valid/wp"]), "delta": float(cm["valid/wp"]) - float(bm["valid/wp"])})
    return rows
def _fmt(x: object) -> str:
    return f"{float(x):.6f}" if _finite(x) else "unknown"

def _body(snap: dict[str, object], controls: dict[int, dict[str, object]], heading: str) -> str:
    adm, stages, clock = snap["admission"], snap["stages"], snap["clock"]
    now = datetime.fromtimestamp(float(clock["now"]), UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    age = "unknown" if not _finite(clock.get("age")) else f"{float(clock['age']):.0f}s"
    lines = [
        f"## Input-dynamics PR78 - {heading}",
        f"Local observation time {now}; mirror last-applied age {age}. Native timestamps are unchanged.",
        f"Source {SOURCE[:16]}...; split {SPLIT[:16]}...; admission {adm.get('receiving_attempt')}.",
        "Pretext loss is training-only; it has no task-validation score. Child WP is on the repeatedly exposed official valid partition.",
        "", "### Four admitted stages",
    ]
    errors = clock.get("errors")
    if isinstance(errors, list) and errors:
        lines.append(f"Mirror collector has {len(errors)} recent error(s); freshness alone is not a completeness proof.")
    for stage in stages:
        name = str(stage["name"])
        if not stage.get("identity_ok"):
            lines.append(f"- {name}: identity unverified; metrics withheld ({stage.get('error', 'awaiting mirror')}).")
            continue
        term, step = stage.get("terminal"), stage.get("step", "unknown")
        total = int(stage["epochs_expected"]) * UPDATES
        phase = f"terminal {term.get('status')}" if isinstance(term, dict) else f"{stage.get('phase', 'unknown')} {stage.get('epoch', 'unknown')} E"
        line = f"- {name} GPU{stage['gpu']}: {phase}; update {step}/{total}"
        if stage["epochs_expected"] == 2:
            train = stage.get("train") if isinstance(stage.get("train"), dict) else {}
            loss = train.get("pretext_loss") if _finite(train.get("pretext_loss")) else train.get("hint_loss")
            line += f"; last rank0 training pretext MSE/loss {_fmt(loss)}; no task-valid metric"
        else:
            checked = [x["event"] for x in stage["evaluations"] if x.get("ok")]
            if checked:
                m = checked[-1]["metrics"]
                line += f"; latest full-valid WP t0/t1 {_fmt(m.get('valid/wp/t0'))}/{_fmt(m.get('valid/wp/t1'))}"
            else:
                line += "; full-valid WP pending"
        if isinstance(term, dict) and _sha(term.get("model_sha256")):
            line += f"; trainer-reported model SHA {term['model_sha256'][:12]}... (not byte-hashed locally)"
        lines.append(line)
    pairs = _comparison(stages, controls)
    lines.extend(["", "### Fixed child endpoints versus matched PR77 controls"])
    if pairs is None:
        lines.append("Pending both fixed4E children, parent receipt joins, complete validation, and same-seed fixed6E controls.")
    else:
        lines.append("Both paired comparisons pass fixed-epoch and full-valid checks. Direct shared outputs; no fitted blend.")
        lines.extend(f"- Seed {r['seed']} {r['target']}: child4E={r['child']:.6f}, control6E={r['control']:.6f}, delta={r['delta']:+.6f}." for r in pairs)
    lines.append("Same exposed validation evidence; not fresh confirmation or a public-score claim.")
    body = "\n".join(lines)
    if len(body.encode()) > 24_000:
        raise ValueError("PR78 report exceeds 24 KB")
    return body

def _state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "reports": {}}
    value = _json(path, 2_000_000)
    if value.get("schema") != STATE_SCHEMA or not isinstance(value.get("reports"), dict):
        raise ValueError("publisher state is malformed")
    return value
def _save(path: Path, value: dict[str, object]) -> None:
    P77._save_state(path, value)

def _publish(client: object, path: Path, marker: str, body: str, when: float) -> str:
    state, reports = _state(path), None
    reports = state["reports"]
    saved = reports.get(marker)
    if isinstance(saved, dict) and saved.get("status") == "posted":
        return "already-posted"
    if isinstance(saved, dict) and saved.get("status") == "pending":
        body, when = str(saved["body"]), float(saved["prepared_at"])
    found = client.find_report(78, marker, when)
    if found:
        reports[marker] = {"status": "posted", "body": body, "prepared_at": when, "comment_id": found.get("id")}
        _save(path, state)
        return "reconciled-existing"
    def before_post() -> None:
        reports[marker] = {"status": "pending", "body": body, "prepared_at": when}
        _save(path, state)
    posted = client.post_report(78, body + f"\n\n{marker}", before_post=before_post)
    reports[marker] = {"status": "posted", "body": body, "prepared_at": when, "comment_id": posted.get("id")}
    _save(path, state)
    return "posted"
def _client(deadline: float) -> object:
    return _module("sequence_progress_github", CLIENT).ProgressGitHub({78}, deadline=deadline)

def run(root: Path, state_path: Path, max_seconds: int, *, dryrun: bool, once: bool) -> int:
    if not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError(f"--maxseconds must be in [1, {MAX_SECONDS}]")
    deadline = time.monotonic() + max_seconds
    client = None if dryrun else _client(deadline)
    errors: set[str] = set()
    while True:
        snap = _snapshot(root)
        controls = {}
        if all(s.get("endpoint_ok") for s in snap["stages"] if s["epochs_expected"] == 4):
            controls = _pr77_controls(root)
        now = float(snap["clock"]["now"])
        bucket = datetime.fromtimestamp(int(now // 1800) * 1800, UTC).strftime("%Y%m%dT%H%M")
        due = [(f"<!-- wnn-l4-input-dynamics-pr78-30m-{bucket}Z -->", _body(snap, controls, "30-minute progress"))]
        for stage in snap["stages"]:
            term = stage.get("terminal")
            if stage.get("identity_ok") and isinstance(term, dict) and (stage.get("endpoint_ok") or term.get("status") != "completed"):
                marker = f"<!-- wnn-l4-input-dynamics-pr78-stage-{stage['name']}-{term['sha256'][:16]} -->"
                due.append((marker, _body(snap, controls, f"stage terminal: {stage['name']} ({term['status']})")))
        pairs = _comparison(snap["stages"], controls)
        if pairs is not None:
            digest = hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
            marker = f"<!-- wnn-l4-input-dynamics-pr78-paired-endpoints-{digest} -->"
            due.append((marker, _body(snap, controls, "fixed child/control comparisons")))
        for marker, body in due:
            if dryrun:
                print(body + f"\n\n{marker}")
                continue
            try:
                print(f"{_publish(client, state_path, marker, body, now)}: {marker}")
            except Exception as error:
                message = f"{marker}: {type(error).__name__}: {error}"
                if message not in errors:
                    print(message, file=sys.stderr)
                    errors.add(message)
        ended = all(s.get("endpoint_ok") or isinstance(s.get("terminal"), dict) and s["terminal"].get("status") != "completed" for s in snap["stages"])
        if once or (ended and pairs is not None) or time.monotonic() >= deadline:
            return 0
        time.sleep(min(POLL, max(0.0, deadline - time.monotonic())))

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pr78", action="store_true", required=True)
    p.add_argument("--maxseconds", type=int, default=MAX_SECONDS)
    p.add_argument("--state-file", type=Path)
    p.add_argument("--dryrun", action="store_true")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    state = a.state_file or a.root / "reporting" / "publisher-state-pr78.json"
    return run(a.root, state, a.maxseconds, dryrun=a.dryrun, once=a.once or a.dryrun)

if __name__ == "__main__":
    raise SystemExit(main())
