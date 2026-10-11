"""Queue a fold-1, seed-S, 3-epoch local experiment (fit + export) cloned from the matched control item, with extra trainer arguments.

Usage: scripts/queue_local_experiment.py --tag fcast832 --seed 3 --extra "--aux-fcast 0.3 --aux-fcast-k 8,32" [--note text]
Uses the v6 trainer (all research flags) and the control contract r3-ctrl as the template, so results compare with r3_ctrl s3/s4 (0.66605 / 0.66578).
Item numbers continue after the highest existing one in pending/done.
"""
from __future__ import annotations
import argparse, datetime, json, re, shlex
from pathlib import Path

Q = Path(__file__).resolve().parents[1] / "research/campaign_2026-10-06/bandit_continuation/local_queue_r1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", default="3")
    ap.add_argument("--extra", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--export-only", default="", help="existing run name: queue only an export of --epoch N of it (output pred name = <run>_x<N>)")
    ap.add_argument("--epoch", default="", help="export epoch index (default: --checkpoint-epoch from --extra, else template)")
    a = ap.parse_args()
    nums = [int(m.group(1)) for d in ("pending", "done", "failed") for p in (Q / d).glob("*.json") if (m := re.match(r"(\d+)_", p.name))]
    num = max(nums) + 1
    fit_t, exp_t = json.loads((Q / "done/77_r3-ctrl-fit-r1.json").read_text()), json.loads((Q / "done/78_r3-ctrl-export-r1.json").read_text())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    name = f"r3_{a.tag}_f1_s{a.seed}_e3"
    pairs = [("r3_ctrl_f1_s3_e3", name), ("r3-ctrl-fit-r1", f"r3-{a.tag}-s{a.seed}-fit-r1"), ("r3-ctrl-export-r1", f"r3-{a.tag}-s{a.seed}-export-r1")]
    def sub(o):
        s = json.dumps(o)
        for x, y in pairs:
            s = s.replace(x, y)
        return json.loads(s)
    toks = shlex.split(a.extra)
    exp_epoch = a.epoch or (toks[toks.index("--checkpoint-epoch") + 1] if "--checkpoint-epoch" in toks else "")
    for kind, tpl, n in (("fit", fit_t, num), ("export", exp_t, num + 1)):
        if a.export_only and kind == "fit":
            continue
        item = sub(tpl)
        contract = json.loads((Q / "contracts" / f"r3-ctrl-{kind}-r1.json").read_text())
        contract = sub(contract)
        if kind == "fit" and "384" in a.extra:
            item["reserve_bytes"] = 10500000000  # h384 fits peak ~9 GB incl. cache: 9 GB reserve killed r3-muonall384prop0-s7-a2
        if kind == "fit":
            for o in (item, contract):
                o["argv"][o["argv"].index("--seed") + 1] = a.seed
                o["argv"] += shlex.split(a.extra)
        if kind == "export" and exp_epoch:
            for o in (item, contract):
                if "--epoch" in o["argv"]:
                    o["argv"][o["argv"].index("--epoch") + 1] = exp_epoch
        if kind == "export" and a.export_only:
            for o in (item, contract):
                o["argv"] = [a.export_only if x == name else x for x in o["argv"]]
                if "--output-name" in o["argv"]:
                    o["argv"][o["argv"].index("--output-name") + 1] = f"{a.export_only}_x{exp_epoch}"
        contract["created_utc"] = now
        contract["note"] = a.note or f"{a.tag}: extra args {a.extra}"
        item["after"] = [] if (kind == "fit" or a.export_only) else [f"r3-{a.tag}-s{a.seed}-fit-r1"]
        (Q / "contracts" / f"{item['attempt']}.json").write_text(json.dumps(contract, indent=1))
        item["contract"] = f"research/campaign_2026-10-06/bandit_continuation/local_queue_r1/contracts/{item['attempt']}.json"
        (Q / "pending" / f"{n}_{item['attempt']}.json").write_text(json.dumps(item, indent=1))
    print(f"queued {name}: items {num}/{num + 1}  extra: {a.extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
