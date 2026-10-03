"""Write scoped report and freeze hashes; read prior review solely to verify preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    assert results["passed"]
    prior = HERE.parents[1] / "round4/combo_confirmation_review"
    frozen = json.loads((prior / "HASHES.json").read_text())
    verified = {name: sha(prior / name) == expected for name, expected in frozen.items()}
    assert all(verified.values())
    assert all(sha(ROOT / p) == h for p, h in results["source_hashes"].items())
    (HERE / "preservation.json").write_text(json.dumps(verified, indent=2) + "\n")
    records = results["records"]
    error = max(arm["oracle_error"] for record in records for arm in record["arms"])
    cost = results["cost"]
    state = {
        "state": "FROZEN_CPU_CONTROL_PROOF_PASS",
        "checks": len(results["checks"]),
        "prior_combo_review_unchanged": True,
        "source_hashes_verified": True,
        "command": "bash research/campaign_2026-10-03/continuation/round5/credit_control/run.sh",
        "resources": "run_cpu.sh round2-critic: CPU1core2GiB900s",
        "cost": cost,
        "no_real_training_data": True,
        "scientific_checkpoint_replayed": False,
        "gpu_or_fused_or_amp_skip_validated": False,
        "empirical_credit_benefit": False,
        "recipe_attribution": False,
        "public_score_claim": False,
        "next": "controller review; no fit or GPU admission implied",
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (HERE / "COST.json").write_text(
        json.dumps(
            {
                "numerical_run": cost,
                "measurement": "main-function wall/CPU time excludes Python/import startup; peak RSS covers process lifetime",
                "wrapper_measurement": "resource_cost.txt includes admission and process startup",
                "development": "two failed exploratory runs (namespace key collision; no actual D-change in fixture), corrected before freeze; several bounded successful reruns for capture and preservation additions",
                "resources": "one core / 2GiB / 900s per command; no GPU",
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    for record in records:
        a, b = record["arms"]
        rows.append(
            f"| H{record['horizon']} {record['control']} | {record['loss']:.9g} | {record['gradient_max_difference']:.9g} | {a['input_first_norm']:.9g} / {b['input_first_norm']:.9g} | {a['mid_norm']:.9g} / {b['mid_norm']:.9g} |"
        )
    report = f"""# Frozen common-macrostep credit control

PASS: {len(results["checks"])} executable checks. Combo review REPORT/STATE and all {len(verified)} previously hashed artifacts are unchanged. This is a synthetic CPU control proof, without a full fit, real targets, empirical credit benefit, recipe attribution or public-score claim.

Both arms execute the actual `RNNModel(3, hidden=6, layers=2, dropout=0, n_out=23)` through the actual trainer `forward_chunk`: two 10-row calls under identical unchanged parameters, concatenate, then evaluate ONE 20-row loss. Only the raw midpoint `.detach()` differs. The encoder is linear; convolutional/stateful encoder alternatives are unsupported. Twenty active synthetic rows are a scaled macrostep interface, not a replay of 2000 real training rows or the warmup/propensity/fit-mask pipeline.

The trainer's four selected loss bodies are compiled from its actual AST, with hashes/line identities in results.json. No separate arm-specific loss implementation exists. Cached teacher inputs are explicitly synthetic detached trainer-facing tensors, not cached exports or reconstructed actual scientific EMA state. Three prior synthetic updates generate genuine lagged EMA weights and distinct detached raw/EMA incoming hidden states under their respective historical weights. These, warm Muon/AdamW state, scheduler, disabled scaler, labels and features are serialized and roundtrip validated. `synthetic_capture.pt` is explicitly synthetic.

The independent output derivative is: main `2w(f-B)/sum(w)` with `w=v*abs(clip(Y))`; EMA `2v(f-U)/sum(v)` broadcast to both targets (no extra divide by two); structural each of 12/3/4 heads `0.3v(softmax-onehot)/sum(v)` (sum three CEs, no divide by three); next `0.6nw(u-(log1p(distance)-3)/2)/sum(nw)` and `0.3nw(sigmoid-a)/sum(nw)`. Denominators use the source floor. Independently loop-reconstructed teacher targets and a VJP of this oracle agree with source autograd: maximum parameter-gradient error {error:.9g}, threshold 2e-6. This checks coefficient/broadcast/teacher-clipping identities, not a statistical optimum.

H0 and strict H8 cached-distillation semantics are source-bound. The fixture independently checks first future RAW PAIR change, strictly `next_change > t+8`, sequence-end exclusion, and pair changes that preserve the target difference (the auxiliary next-loss helper differs). This validates synthetic branch arithmetic only; it does not endorse the source comment's unbiased-stopping-law interpretation.

Every forward/EMA output, teacher target and common loss is finite and BITEXACT across detach. Planted second-half-only loss gives nonzero first-half input and midpoint credit in the connected arm, exactly zero in the detached arm, and different parameter/optimizer updates. First-half-only loss produces BITEXACT parameter gradients, optimizer state and updated weights. Mean-of-normalized-halves is measurably unequal to the common full-macrostep loss for this unequal-mass fixture and is rejected.

| Case | Common loss | max gradient difference | first-half input gradient norm: connected / detached | midpoint gradient norm: connected / detached |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Each arm reloads exactly the same warm optimizer/scheduler/EMA/scaler state and advances once: scheduler epoch 3→4, EMA counter 3→4, Adam state steps 3→4; Muon momentum is retained, not reset. Optimizer group ordering/names are recorded. Actual source clip=1 executes before each actual MuonAdamW step, including cases with preclip norm above one. `synthetic_replays.pt` stores preclip/postclip gradients and post-update raw/EMA/optimizer/scheduler/scaler state for independent inspection. Same clock does not imply same updates: the positive and null controls discriminate this explicitly.

CPU GradScaler is disabled: state is empty, scale operation is identity, and one actual step/update call occurs per arm. There is NO active overflow/skip or fused/GPU parity evidence. Other trainer branches, actual teacher-cache binding, native inference, full data exposures and scientific checkpoint replay remain unsupported. No 32-fit launcher, GPU request, live trainer edit or queue/ledger mutation is included. This foundation is informative for checking a future common-clock gradient instrument; it cannot approve its own mechanistic candidate or authorize a fit.

Reproduce: `bash research/campaign_2026-10-03/continuation/round5/credit_control/run.sh`. It invokes `run_cpu.sh round2-critic` with one core/2GiB/900s. Numerical run: {cost["wall_seconds"]:.3f}s wall, {cost["cpu_seconds"]:.3f}s CPU inside main, peak RSS {cost["peak_rss_kib"]} KiB. See resource_cost.txt for whole wrapper command cost and checks.log for executed results. Initial fixture construction errors were corrected before this freeze; they were not trainer bugs. Source files were read/compiled in place, not copied or modified.
"""
    (HERE / "REPORT.md").write_text(report)
    hashes = {
        p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "HASHES.json"
    }
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {"frozen": True, "checks": len(results["checks"]), "prior_preserved": len(verified)}
        )
    )


if __name__ == "__main__":
    main()
