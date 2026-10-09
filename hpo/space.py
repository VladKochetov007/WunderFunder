"""Search space (v2, narrowed by the 46-model evidence report: epochs above 4, the corr/huber losses, TCN and widths above 384
are excluded; the continuous knobs are kept narrow because single-fold noise hides small effects) for teacher hyper-parameter optimization and its mapping to trainer arguments.

Two modes share the parameter names: ``core`` suggests the dozen parameters with the strongest prior (learning-rate
family, schedule, budget, width/depth, initialization, loss and target weighting, weight averaging); ``full`` also
suggests the secondary ones.  Parameters not suggested keep the recipe defaults of ``build_bundle.BASE``.  Seeds are never
searched.  The space is versioned (``SPACE_VERSION``): trials of different versions are never mixed in one study.
"""

from __future__ import annotations

SPACE_VERSION = "v2"
COST_CAP_HOURS = 2.6  # trials estimated above this are infeasible for the sampler (TPE constraint), not skipped silently

DEFAULTS = {
    "optim": "muon_rnn", "lr": 1e-3, "muon_mult": 1.0, "wd": 1e-2, "head_lr_mult": 1.0, "clip": 1.0, "sched": "onecycle",
    "warmup_frac": 0.05, "wsd_decay": 0.2, "wsd_final": 0.01, "restarts": 2, "epochs": 4, "batch": 32, "chunk": 1000,
    "hidden": 256, "layers": 2, "cell": "gru", "dropout": 0.0, "readout": "gelu", "enc": "linear", "rnn_init": "default",
    "chrono_tmax": 2000.0, "loss": "wmse", "gcorr_weight": 0.5, "lw_power": 1.0, "lw_floor": 0.0, "class3_weight": 1.0,
    "since_weight": 32.0, "prop_power": 2.0, "aux_struct": 0.3, "aux_next": 0.3, "avg": "none", "swa_frac": 0.25,
}

CORE = ("optim", "lr", "muon_mult", "wd", "sched", "epochs", "hidden", "layers", "rnn_init", "loss", "class3_weight",
        "avg", "dropout")


def suggest(trial, mode: str = "core") -> dict:
    """Suggest a parameter dict (conditional parameters only when their parent value makes them meaningful)."""
    full = mode == "full"
    p = dict(DEFAULTS)
    p["optim"] = trial.suggest_categorical("optim", ["muon_rnn", "muon_hidden"])
    p["lr"] = trial.suggest_float("lr", 6e-4, 2e-3, log=True)
    p["muon_mult"] = trial.suggest_float("muon_mult", 0.5, 2.0, log=True)
    p["wd"] = trial.suggest_float("wd", 3e-3, 6e-2, log=True)
    p["sched"] = trial.suggest_categorical("sched", ["onecycle", "wsd_sqrt", "wsd_linear", "sgdr"])
    if p["sched"].startswith("wsd"):
        p["wsd_decay"] = trial.suggest_float("wsd_decay", 0.1, 0.5)
        p["wsd_final"] = trial.suggest_float("wsd_final", 0.0, 0.05)
    if p["sched"] == "sgdr":
        p["restarts"] = trial.suggest_int("restarts", 1, 4)
    p["epochs"] = trial.suggest_categorical("epochs", [2, 3, 4, 5])
    p["hidden"] = trial.suggest_categorical("hidden", [256, 320, 384])
    p["layers"] = trial.suggest_categorical("layers", [2, 3])
    p["rnn_init"] = trial.suggest_categorical("rnn_init", ["default", "chrono", "orth", "chrono_orth"])
    if "chrono" in p["rnn_init"]:
        p["chrono_tmax"] = trial.suggest_float("chrono_tmax", 500.0, 10000.0, log=True)
    p["loss"] = trial.suggest_categorical("loss", ["wmse", "wmse_gcorr", "mix"])
    if p["loss"] == "wmse_gcorr":
        p["gcorr_weight"] = trial.suggest_float("gcorr_weight", 0.05, 1.0)
    if p["loss"] == "wmse":
        p["lw_power"] = trial.suggest_float("lw_power", 0.8, 1.6)
        p["lw_floor"] = trial.suggest_float("lw_floor", 0.0, 0.2)
    p["class3_weight"] = trial.suggest_float("class3_weight", 0.7, 2.5)
    p["avg"] = trial.suggest_categorical("avg", ["none", "ema", "swa"])
    if p["avg"] == "swa":
        p["swa_frac"] = trial.suggest_float("swa_frac", 0.1, 0.4)
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.25)
    if full:
        p["head_lr_mult"] = trial.suggest_float("head_lr_mult", 0.3, 3.0, log=True)
        p["clip"] = trial.suggest_float("clip", 0.3, 3.0, log=True)
        p["warmup_frac"] = trial.suggest_float("warmup_frac", 0.02, 0.2)
        p["batch"] = trial.suggest_categorical("batch", [16, 32, 48])
        p["chunk"] = trial.suggest_categorical("chunk", [500, 1000, 2000])
        p["cell"] = trial.suggest_categorical("cell", ["gru", "lstm"])
        p["readout"] = trial.suggest_categorical("readout", ["gelu", "fixed_mix"])
        p["enc"] = trial.suggest_categorical("enc", ["linear", "mlp"])
        p["since_weight"] = trial.suggest_float("since_weight", 8.0, 256.0, log=True)
        p["prop_power"] = trial.suggest_float("prop_power", 0.5, 3.0)
        p["aux_struct"] = trial.suggest_float("aux_struct", 0.0, 0.6)
        p["aux_next"] = trial.suggest_float("aux_next", 0.0, 0.6)
    return p


def estimate_hours(p: dict) -> float:
    """Wall-clock hours of one fold run at 6 concurrent jobs per 2-GPU host (calibrated on GRU256x2, 4 epochs, 1.2 h)."""
    width = (p["hidden"] / 256) ** 2
    depth = p["layers"] / 2
    cell = 1.33 if p["cell"] == "lstm" else 1.0
    chunk = {500: 1.15, 1000: 1.0, 2000: 0.9}.get(p["chunk"], 1.0)
    batch = {16: 1.25, 32: 1.0, 48: 0.9}.get(p["batch"], 1.0)
    enc = 1.1 if p["enc"] == "mlp" else 1.0
    return round(1.2 * (p["epochs"] / 4) * (0.35 + 0.65 * width * depth * cell) * chunk * batch * enc, 2)


def constraint(p: dict) -> float:
    """TPE constraint value: <= 0 is feasible."""
    return estimate_hours(p) - COST_CAP_HOURS


def to_change(p: dict, fold: int) -> dict:
    """Trainer-argument delta for ``make_refill_jobs.build_job`` (keys of BASE are replaced, ``+`` flags appended)."""
    change: dict = {"fold": fold}
    extra: list[str] = []
    scope = "hidden" if p["optim"] == "muon_hidden" else "rnn"
    change.update({"--optim": "muon", "--muon-scope": scope, "--lr": f"{p['lr']:.6g}", "--wd": f"{p['wd']:.6g}",
                   "--epochs": str(p["epochs"]), "--checkpoint-epoch": str(p["epochs"] - 1), "--batch": str(p["batch"]),
                   "--chunk": str(p["chunk"]), "--hidden": str(p["hidden"]), "--layers": str(p["layers"]),
                   "--cell": p["cell"], "--dropout": f"{p['dropout']:.4g}", "--enc": p["enc"],
                   "--loss": "wmse" if p["loss"] == "wmse_gcorr" else p["loss"],
                   "--since-weight": f"{p['since_weight']:.6g}", "--prop-power": f"{p['prop_power']:.4g}",
                   "--aux-struct": f"{p['aux_struct']:.4g}", "--aux-next": f"{p['aux_next']:.4g}"})
    if p["loss"] == "wmse_gcorr":
        change["--loss"] = "wmse_gcorr"
        extra += ["--gcorr-weight", f"{p['gcorr_weight']:.4g}"]
    if p["muon_mult"] != 1.0:
        extra += ["--muon-lr-mult", f"{p['muon_mult']:.4g}"]
    if p["head_lr_mult"] != 1.0:
        extra += ["--head-lr-mult", f"{p['head_lr_mult']:.4g}"]
    if p["clip"] != 1.0:
        extra += ["--clip", f"{p['clip']:.4g}"]
    if p["warmup_frac"] != 0.05:
        extra += ["--warmup-frac", f"{p['warmup_frac']:.4g}"]
    if p["sched"].startswith("wsd"):
        change["--sched"] = "wsd"
        extra += ["--wsd-decay", f"{p['wsd_decay']:.4g}", "--wsd-final", f"{p['wsd_final']:.4g}"]
        if p["sched"] == "wsd_linear":
            extra += ["--wsd-shape", "linear"]
    elif p["sched"] == "sgdr":
        change["--sched"] = "sgdr"
        extra += ["--restarts", str(p["restarts"]), "--wsd-final", f"{p['wsd_final']:.4g}"]
    if p["readout"] != "gelu":
        extra += ["--readout-activation", p["readout"]]
    if p["rnn_init"] != "default":
        extra += ["--rnn-init", p["rnn_init"]]
        if "chrono" in p["rnn_init"]:
            extra += ["--chrono-tmax", f"{p['chrono_tmax']:.5g}"]
    if p["loss"] == "wmse" and (p["lw_power"] != 1.0 or p["lw_floor"] > 0.0):
        extra += ["--label-weight-power", f"{p['lw_power']:.4g}", "--label-weight-floor", f"{p['lw_floor']:.4g}"]
    if abs(p["class3_weight"] - 1.0) > 1e-9:
        extra += ["--class-weight", f"1,1,1,{p['class3_weight']:.4g}"]
    if p["avg"] == "ema":
        extra += ["--ema-teacher", "1.0", "--eval-ema"]
    elif p["avg"] == "swa":
        extra += ["--swa-frac", f"{p['swa_frac']:.4g}"]
    if extra:
        change["+"] = extra
    return change


# First trials of every study: known-good anchors so the sampler starts from evidence, not only random draws.
BASELINES = [
    {"epochs": 3},  # evidence: 6 and 8 epochs are far worse (-0.017, -0.036), shorter training is untested
    {"epochs": 2},
    {"epochs": 3, "hidden": 384},
    {"hidden": 384, "dropout": 0.2},
    {"epochs": 3, "sched": "wsd_sqrt"},
]
