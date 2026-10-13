from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
JOBS = [(name, "original") for name in ("pair_ftt2", "xteam_h4", "C", "E", "e6a", "p_c3zero", "p_c3only")]
JOBS.append(("pair_ftt2", "rust"))


async def run_one(name: str, mode: str, slots: asyncio.Semaphore) -> dict[str, object]:
    async with slots:
        admission = json.loads((HERE / "FULL_REPLAY_ADMISSION_R1.json").read_text())
        source = HERE / "score_archive.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != admission["source_sha256"]:
            raise ValueError("admitted worker source changed")
        out = HERE / "scores" / f"{name}_{mode}_r1"
        if out.exists():
            raise FileExistsError(out)
        env = dict(os.environ)
        env.update(WNN_SLICE="wnn-campaign.slice", WNN_CPU_QUOTA="100%", WNN_THREADS="1",
                   WNN_MEM_MAX="1100000000", WNN_RUNTIME_MAX="12000s",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   WNN_SCOPE_UNIT=f"wnn-team-{name.lower().replace('_', '-')}-{mode}-r1")
        command = [str(ROOT / "scripts/limited"), str(ROOT / ".venv-py311-profile/bin/python"),
                   str(source), "--candidate", name, "--mode", mode, "--count", "1873", "--out", str(out)]
        receipt = {"candidate": name, "mode": mode, "command": command,
                   "started_utc": datetime.now(UTC).isoformat(), "environment": {k: env[k] for k in env if k.startswith("WNN_")}}
        (HERE / "scores").mkdir(exist_ok=True)
        log_path = HERE / "scores" / f"{name}_{mode}_r1.log"
        print(json.dumps({"event": "started", **receipt}), flush=True)
        with log_path.open("xb") as log:
            process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
            receipt["exit_code"] = await process.wait()
        receipt["finished_utc"] = datetime.now(UTC).isoformat()
        (HERE / "scores" / f"{name}_{mode}_ATTEMPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"event": "finished", **receipt}), flush=True)
        return receipt


async def main() -> None:
    slots = asyncio.Semaphore(4)
    results = await asyncio.gather(*(run_one(name, mode, slots) for name, mode in JOBS))
    (HERE / "FULL_REPLAY_TERMINAL_R1.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
