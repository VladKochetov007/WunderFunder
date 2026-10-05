"""Render the preregistered E476/E480 three-seed activation-schedule comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E476_SCORE = ROOT / "research/campaign_2026-10-05/activation_schedule/eda/twelve_v1/scores.json"
E476_EXPORT = (
    ROOT / "research/campaign_2026-10-05/activation_schedule/eda/twelve_v1/selected_exports.npz"
)
E480_SCORE = HERE / "eda/twentyfour_v1/scores.json"
E480_EXPORT = HERE / "eda/twentyfour_v1/selected_exports.npz"
E480_GRADE_SPEC = HERE / "e480_cpu_grade_v1.sealed.json"
E480_GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e480-activation-schedule-replication-cpu-test-grade-v1.json"
)
E480_FIT_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e480-activation-schedule-replication-twentyfour-fit-v1.json"
)
NUMPY_REPRODUCTION = (
    ROOT / "research/campaign_2026-10-05/checks/activation_export_reproduction.json"
)
NUMPY_REPRODUCTION_SOURCE = ROOT / "research/campaign_2026-10-05/reproduce_activation_exports.py"
OUTPUT_DIR = HERE / "eda/three_seed_v1"
SEEDS = (271_828, 271_829, 271_830)
ACTIVATIONS = ("base", "sine_embed")
OPTIMIZERS = ("adamw", "muon_adamw")
SCHEDULES = ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
BOOTSTRAP_COUNT = 2_000
EXPECTED_NUMPY_REPRODUCTION_SOURCE_SHA256 = (
    "a7ba0ceaa435ec9b2b6e7b935f0796c896c4fbc8b6f80774b8d3575e084e3121"
)


@dataclass(frozen=True)
class ArmScore:
    """One fixed-seed, activation, optimizer, and schedule score."""

    seed: int
    activation: str
    optimizer: str
    schedule: str
    wp: float
    t0: float
    t1: float
    experiment: str

    @property
    def key(self) -> tuple[int, str, str, str]:
        """Return the unique registered cell key."""
        return self.seed, self.activation, self.optimizer, self.schedule


@dataclass(frozen=True)
class PrimaryEffect:
    """Paired warm-cosine minus constant-mean effect and interval."""

    seed: int
    activation: str
    optimizer: str
    delta: float
    low: float
    high: float
    p_le_zero: float
    experiment: str

    @property
    def label(self) -> str:
        """Return a display label for the matched candidate pair."""
        activation = "Sine embed" if self.activation == "sine_embed" else "Base GELU"
        optimizer = "MuonAdamW" if self.optimizer == "muon_adamw" else "AdamW"
        return f"{activation} / {optimizer}"


@dataclass(frozen=True)
class ContrastRow:
    """One E480 registered paired contrast for the report table."""

    seed: int
    name: str
    family: str
    candidate: str
    reference: str
    delta: float
    low: float
    high: float
    p_le_zero: float


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object and reject malformed roots."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], value)


def record(value: object, label: str) -> dict[str, object]:
    """Narrow a JSON object after an explicit runtime type check."""
    if not isinstance(value, dict):
        raise ValueError(f"expected object for {label}")
    return cast(dict[str, object], value)


def validate_grade_receipt(score: dict[str, object]) -> dict[str, object]:
    """Validate the official E480 CPU-grade spec, attempt, runtime, and outputs."""
    sys_path = str(ROOT / "src")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from wnn.campaign import RUNTIME_FILES, runtime_identity
    from wnn.campaign_spec import digest, validate_spec

    spec = validate_spec(read_json(E480_GRADE_SPEC))
    attempt = read_json(E480_GRADE_ATTEMPT)
    if spec.get("name") != "e480_independent_cpu_export_grade_v1":
        raise ValueError("E480 score-grade spec identity changed")
    if attempt.get("id") != "e480-activation-schedule-replication-cpu-test-grade-v1":
        raise ValueError("E480 score-grade attempt identity changed")
    if attempt.get("spec") != spec or attempt.get("spec_sha256") != digest(spec):
        raise ValueError("E480 score-grade runner receipt does not bind its sealed spec")
    if (
        attempt.get("status") != "finished"
        or attempt.get("exit_code") != 0
        or attempt.get("signals") != []
        or attempt.get("teardown_verified") is not True
        or any(
            attempt.get(key) is not None for key in ("failure", "failure_scope", "failure_category")
        )
    ):
        raise ValueError("E480 score-grade attempt did not finish successfully")
    unit = record(attempt.get("unit_properties"), "grade unit properties")
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("E480 score-grade systemd result or exit status is not successful")
    runtime = attempt.get("runtime_sha256")
    current_runtime = runtime_identity(ROOT)
    if (
        not isinstance(runtime, dict)
        or set(runtime) != set(RUNTIME_FILES)
        or runtime != current_runtime
    ):
        raise ValueError("E480 score-grade runtime hashes differ from this checkout")
    for initial_key, final_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        initial = record(attempt.get(initial_key), initial_key)
        final = record(attempt.get(final_key), final_key)
        for counter in ("oom", "oom_kill", "oom_group_kill"):
            if initial.get(counter) != final.get(counter):
                raise ValueError(f"E480 grade has a new OOM counter: {initial_key}/{counter}")
    artifacts = attempt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("E480 score-grade receipt must contain scores and selected exports")
    checked_artifacts = []
    for artifact_value in artifacts:
        artifact = record(artifact_value, "grade artifact")
        path = ROOT / str(artifact["path"])
        if not path.is_file():
            raise ValueError(f"E480 grade output missing: {path}")
        if sha256_file(path) != artifact.get("sha256") or path.stat().st_size != artifact.get(
            "size"
        ):
            raise ValueError(f"E480 grade output differs from runner inventory: {path}")
        checked_artifacts.append(dict(artifact))
    spec_artifacts = list(spec.get("artifacts", []))
    if [artifact["path"] for artifact in checked_artifacts] != spec_artifacts:
        raise ValueError("E480 score-grade artifact order differs from sealed spec")
    if sha256_file(E480_SCORE) != checked_artifacts[0]["sha256"]:
        raise ValueError("E480 scores file differs from successful grade attempt")
    if sha256_file(E480_EXPORT) != checked_artifacts[1]["sha256"]:
        raise ValueError("E480 selected export differs from successful grade attempt")
    fit_receipt = record(score.get("runner_attempt_receipt"), "E480 fit receipt")
    fit_attempt = read_json(E480_FIT_ATTEMPT)
    if (
        fit_receipt.get("attempt_id") != fit_attempt.get("id")
        or fit_receipt.get("attempt_sha256") != sha256_file(E480_FIT_ATTEMPT)
        or fit_attempt.get("status") != "finished"
        or fit_attempt.get("exit_code") != 0
        or fit_attempt.get("teardown_verified") is not True
    ):
        raise ValueError("E480 fit runner receipt in score report is not independently verified")
    fit_manifest_artifacts = fit_receipt.get("artifacts")
    if not isinstance(fit_manifest_artifacts, list) or not fit_manifest_artifacts:
        raise ValueError("E480 scorer omitted its fit artifact receipt")
    fit_manifest_hash = record(fit_manifest_artifacts[0], "fit manifest artifact").get("sha256")
    if score.get("fit_manifest_sha256") != fit_manifest_hash:
        raise ValueError("E480 score report fit manifest does not match the fit runner receipt")
    return {
        "score_grade_attempt_id": attempt["id"],
        "score_grade_attempt_sha256": sha256_file(E480_GRADE_ATTEMPT),
        "score_grade_spec_sha256": digest(spec),
        "status": attempt["status"],
        "exit_code": attempt["exit_code"],
        "signals": attempt["signals"],
        "teardown_verified": attempt["teardown_verified"],
        "unit_result": unit.get("Result"),
        "unit_exit_code": unit.get("ExecMainStatus"),
        "elapsed_seconds": float(attempt.get("elapsed_seconds", float("nan"))),
        "peak_ram_bytes": int(attempt.get("peak_ram_bytes", -1)),
        "peak_observed_vram_bytes": int(attempt.get("peak_observed_vram_bytes", -1)),
        "worker_oom_counters_unchanged": True,
        "parent_oom_counters_unchanged": True,
        "stored_unit_active_state": unit.get("ActiveState"),
        "artifact_receipts": checked_artifacts,
        "fit_attempt_id": fit_attempt.get("id"),
        "fit_attempt_sha256": fit_receipt.get("attempt_sha256"),
    }


def validate_numpy_reproduction(e480_doc: dict[str, object]) -> dict[str, object]:
    """Verify the independent pure-NumPy reconstruction of all E480 values."""
    rows = json.loads(NUMPY_REPRODUCTION.read_text())
    if not isinstance(rows, list):
        raise ValueError("activation export reproduction receipt must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("study") == "E480"]
    if len(matches) != 1:
        raise ValueError("independent E480 NumPy reproduction entry is missing/duplicated")
    row = matches[0]
    if (
        row.get("status") != "all_checked_values_match"
        or row.get("arms") != 24
        or row.get("global_contrasts") != 44
        or float(row.get("maximum_absolute_error", float("inf"))) > 1e-10
        or row.get("result_sha256") != sha256_file(E480_SCORE)
        or row.get("export_sha256") != sha256_file(E480_EXPORT)
        or sha256_file(NUMPY_REPRODUCTION_SOURCE) != EXPECTED_NUMPY_REPRODUCTION_SOURCE_SHA256
    ):
        raise ValueError("independent NumPy reproduction does not match E480 scores/exports")
    return cast(dict[str, object], row)


def load_arm_scores(score_doc: dict[str, object], experiment: str) -> list[ArmScore]:
    """Parse canonical E476 or E480 arm score maps into fixed typed records."""
    raw_arms = record(score_doc.get("arm_scores"), f"{experiment} arm scores")
    results = []
    fit_seed = int(score_doc["fit_seed"]) if experiment == "E476" else None
    for raw_key, raw_value in raw_arms.items():
        arm = record(raw_value, f"{experiment}/{raw_key}")
        if experiment == "E476":
            activation, optimizer, schedule = str(raw_key).split("__")
            seed = cast(int, fit_seed)
        else:
            seed = int(arm["seed"])
            activation, optimizer, schedule = (
                str(arm["activation"]),
                str(arm["optimizer"]),
                str(arm["schedule"]),
            )
            if str(raw_key) != f"seed{seed}__{activation}__{optimizer}__{schedule}":
                raise ValueError(f"E480 cell label is inconsistent: {raw_key}")
        if (
            activation not in ACTIVATIONS
            or optimizer not in OPTIMIZERS
            or schedule not in SCHEDULES
        ):
            raise ValueError(f"unexpected score cell: {raw_key}")
        results.append(
            ArmScore(
                seed=seed,
                activation=activation,
                optimizer=optimizer,
                schedule=schedule,
                wp=float(arm["wp"]),
                t0=float(arm["t0"]),
                t1=float(arm["t1"]),
                experiment=experiment,
            )
        )
    expected = 12 if experiment == "E476" else 24
    if len(results) != expected or len({item.key for item in results}) != expected:
        raise ValueError(f"{experiment} score map must have exactly {expected} unique cells")
    return results


def parse_ci(value: object, label: str) -> tuple[float, float]:
    """Validate a two-sided percentile interval."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"malformed confidence interval: {label}")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"invalid confidence interval: {label}")
    return low, high


def load_primary_effects(
    e476_doc: dict[str, object], e480_doc: dict[str, object]
) -> list[PrimaryEffect]:
    """Read all four schedule primary contrasts from each fixed fit seed."""
    effects = []
    e476_pairs = record(e476_doc.get("pair_contrasts"), "E476 paired effects")
    for name, raw_value in e476_pairs.items():
        value = record(raw_value, f"E476 contrast {name}")
        if value.get("family") != "warmcos_minus_constant_mean":
            continue
        candidate = str(value["candidate"]).split("__")
        if len(candidate) != 3:
            raise ValueError(f"malformed E476 primary candidate: {name}")
        low, high = parse_ci(value.get("ci95"), str(name))
        effects.append(
            PrimaryEffect(
                seed=int(e476_doc["fit_seed"]),
                activation=candidate[0],
                optimizer=candidate[1],
                delta=float(value["mean_diff"]),
                low=low,
                high=high,
                p_le_zero=float(value["p_le_0"]),
                experiment="E476 original",
            )
        )
    primary_by_seed = record(
        e480_doc.get("primary_warmcos_minus_constant_mean_by_seed"), "E480 primary contrasts"
    )
    for seed_value, raw_seed_effects in primary_by_seed.items():
        seed_effects = record(raw_seed_effects, f"E480 seed {seed_value} primary contrasts")
        if len(seed_effects) != 4:
            raise ValueError(f"E480 seed {seed_value} must have four primary effects")
        for name, raw_value in seed_effects.items():
            value = record(raw_value, f"E480 primary {name}")
            if value.get("family") != "warmcos_minus_constant_mean":
                raise ValueError(f"unexpected E480 primary family: {name}")
            activation = str(value["candidate"]).split("__")[0]
            optimizer = str(value["candidate"]).split("__")[1]
            low, high = parse_ci(value.get("ci95"), str(name))
            effects.append(
                PrimaryEffect(
                    seed=int(seed_value),
                    activation=activation,
                    optimizer=optimizer,
                    delta=float(value["mean_diff"]),
                    low=low,
                    high=high,
                    p_le_zero=float(value["p_le_0"]),
                    experiment="E480 replication",
                )
            )
    if len(effects) != 12 or {effect.seed for effect in effects} != set(SEEDS):
        raise ValueError("expected all 12 primary effects across the three fixed seeds")
    return sorted(
        effects,
        key=lambda item: (
            item.seed,
            ACTIVATIONS.index(item.activation),
            OPTIMIZERS.index(item.optimizer),
        ),
    )


def load_e480_contrasts(score_doc: dict[str, object]) -> list[ContrastRow]:
    """Read the complete 44-contrast E480 result table."""
    rows = []
    for top_key, expected_per_seed in (
        ("pair_contrasts_by_seed", 20),
        ("joint_interactions_by_seed", 2),
    ):
        per_seed = record(score_doc.get(top_key), top_key)
        if set(per_seed) != {"271829", "271830"}:
            raise ValueError(f"unexpected E480 seed set in {top_key}")
        for seed_text, raw_effects in per_seed.items():
            effects = record(raw_effects, f"{top_key}/{seed_text}")
            if len(effects) != expected_per_seed:
                raise ValueError(f"{top_key}/{seed_text} must have {expected_per_seed} effects")
            for name, raw_value in effects.items():
                value = record(raw_value, f"E480 effect {name}")
                low, high = parse_ci(value.get("ci95"), str(name))
                rows.append(
                    ContrastRow(
                        seed=int(seed_text),
                        name=str(name),
                        family=str(value["family"]),
                        candidate=str(value.get("candidate", value.get("contrast", ""))),
                        reference=str(value.get("reference", "joint interaction")),
                        delta=float(value["mean_diff"]),
                        low=low,
                        high=high,
                        p_le_zero=float(value["p_le_0"]),
                    )
                )
    if len(rows) != 44:
        raise ValueError(f"expected 44 E480 contrasts, got {len(rows)}")
    return sorted(rows, key=lambda row: (row.seed, row.family, row.name))


def display_cell(arm: ArmScore) -> str:
    """Compact cell code used in the score panels."""
    activation = "B" if arm.activation == "base" else "S"
    optimizer = "A" if arm.optimizer == "adamw" else "M"
    schedule = {"constant_peak": "P", "warmcos_47_427": "W", "constant_mean_warmcos": "C"}[
        arm.schedule
    ]
    return f"{activation}/{optimizer}/{schedule}"


def render_scores(arms: list[ArmScore], path: Path) -> None:
    """Render all 36 global-WP scores on one shared axis scale."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = {"base": "#2878b5", "sine_embed": "#e07a24"}
    markers = {"adamw": "o", "muon_adamw": "s"}
    all_wp = np.asarray([arm.wp for arm in arms])
    lower = min(0.35, float(all_wp.min()) - 0.015)
    upper = max(0.56, float(all_wp.max()) + 0.015)
    canonical = [
        (activation, optimizer, schedule)
        for activation in ACTIVATIONS
        for optimizer in OPTIMIZERS
        for schedule in SCHEDULES
    ]
    for axis, seed in zip(axes, SEEDS, strict=True):
        by_key = {arm.key[1:]: arm for arm in arms if arm.seed == seed}
        if set(by_key) != set(canonical):
            raise ValueError(f"score facet for seed {seed} does not have all 12 cells")
        for index, key in enumerate(canonical):
            arm = by_key[key]
            axis.scatter(
                index,
                arm.wp,
                color=colors[arm.activation],
                marker=markers[arm.optimizer],
                s=62,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_title(
            f"Seed {seed}" + (" (original E476)" if seed == 271828 else " (E480 replication)")
        )
        axis.set_xticks(
            range(len(canonical)),
            [display_cell(by_key[key]) for key in canonical],
            rotation=55,
            ha="right",
        )
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Activation / optimizer / schedule")
    axes[0].set_ylabel("Official global weighted Pearson (WP)")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            label=label,
            markersize=8,
        )
        for label, color in (
            ("Base GELU", colors["base"]),
            ("Sine embedding", colors["sine_embed"]),
        )
    ] + [
        plt.Line2D(
            [0], [0], marker=marker, color="black", linestyle="None", label=label, markersize=7
        )
        for label, marker in (("AdamW", markers["adamw"]), ("MuonAdamW", markers["muon_adamw"]))
    ]
    fig.suptitle(
        "E476 and E480 scores on one fixed 16-component exposed evaluation cohort",
        y=0.985,
        fontsize=15,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.27, top=0.80, wspace=0.04)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_primary(effects: list[PrimaryEffect], path: Path) -> None:
    """Render all 12 paired primary effects with a common delta axis."""
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = {271_828: "#2878b5", 271_829: "#e07a24", 271_830: "#2a9d66"}
    combos = [(activation, optimizer) for activation in ACTIVATIONS for optimizer in OPTIMIZERS]
    offsets = {271_828: -0.21, 271_829: 0.0, 271_830: 0.21}
    labels = []
    for index, (activation, optimizer) in enumerate(combos):
        label = (
            ("Sine embed" if activation == "sine_embed" else "Base GELU")
            + " / "
            + ("MuonAdamW" if optimizer == "muon_adamw" else "AdamW")
        )
        labels.append(label)
        for seed in SEEDS:
            effect = next(
                item
                for item in effects
                if item.seed == seed
                and item.activation == activation
                and item.optimizer == optimizer
            )
            y = index + offsets[seed]
            axis.errorbar(
                effect.delta,
                y,
                xerr=np.asarray([[effect.delta - effect.low], [effect.high - effect.delta]]),
                fmt="o",
                color=colors[seed],
                capsize=3,
                markersize=6,
                linewidth=1.4,
            )
    axis.axvline(0.0, color="black", linewidth=1, alpha=0.7)
    axis.set_yticks(range(len(combos)), labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.14, 0.14)
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlabel(
        "Warm-cosine minus constant-mean schedule (paired Δ global WP; 95% component bootstrap CI)"
    )
    axis.set_title("Primary schedule effect reverses across fixed initialization seeds")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color=color, linestyle="None", label=f"Seed {seed}", markersize=7
        )
        for seed, color in colors.items()
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    """Format WP or contrast values compactly."""
    return f"{value:+.5f}"


def make_report(
    e476_doc: dict[str, object],
    e480_doc: dict[str, object],
    arms: list[ArmScore],
    effects: list[PrimaryEffect],
    contrasts: list[ContrastRow],
    receipt: dict[str, object],
    numpy_reproduction: dict[str, object],
    plot_scores: Path,
    plot_primary: Path,
) -> str:
    """Create the three-seed result report with complete registered tables."""
    if len(arms) != 36 or len(effects) != 12 or len(contrasts) != 44:
        raise ValueError("report inputs do not include every preregistered result")
    replication = [item for item in effects if item.seed in (271_829, 271_830)]
    negative_points = sum(item.delta < 0 for item in replication)
    negative_ci = sum(item.high < 0 for item in replication)
    if (negative_points, negative_ci) != (8, 6):
        raise ValueError(
            f"E480 primary summary differs from expected evidence: {negative_points}/{negative_ci}"
        )
    original_positive_points = sum(item.delta > 0 for item in effects if item.seed == 271_828)
    if original_positive_points != 4:
        raise ValueError("original E476 seed no longer has all four positive primary effects")
    score_values = [item.wp for item in arms]
    contrast_rows = []
    for row in contrasts:
        contrast_rows.append(
            f"| {row.seed} | {row.family} | `{row.name}` | {fmt(row.delta)} | [{fmt(row.low)}, {fmt(row.high)}] | {row.p_le_zero:.3f} |"
        )
    primary_rows = []
    for effect in effects:
        primary_rows.append(
            f"| {effect.seed} | {effect.experiment} | {effect.label} | {fmt(effect.delta)} | [{fmt(effect.low)}, {fmt(effect.high)}] | {effect.p_le_zero:.3f} |"
        )
    score_rows = []
    for item in sorted(
        arms,
        key=lambda arm: (
            arm.seed,
            ACTIVATIONS.index(arm.activation),
            OPTIMIZERS.index(arm.optimizer),
            SCHEDULES.index(arm.schedule),
        ),
    ):
        score_rows.append(
            f"| {item.seed} | {item.experiment} | {item.activation} | {item.optimizer} | {item.schedule} | {item.wp:.5f} | {item.t0:.5f} | {item.t1:.5f} |"
        )
    grade_unit_state = (
        "The saved attempt snapshot listed `ActiveState=activating`; a fresh unit query showed `inactive/dead`, "
        "with `Result=success` and `ExecMainStatus=0`."
    )
    return f"""# Three-seed activation and schedule results

E480's independent CPU scoring attempt finished with exit code 0, no signals,
verified teardown, and successful systemd result. The sealed grade spec and
runtime hashes match; both output files match the runner's path, SHA-256, and
size inventory. {grade_unit_state} The fit receipt also binds the completed
two-seed E480 training artifacts. Receipt summary: `{json.dumps(receipt, sort_keys=True)}`.

E476's original seed 271828 is shown next to the two fixed E480 replication
seeds 271829 and 271830. All 36 arms use the same exposed 16-component
evaluation cohort and `global_wp_clip2_v1`; this is a paired diagnostic, not a
fresh holdout or public-score estimate. The E480 scorer reports 40 paired
effects, four joint interactions, and eight primary warm-cosine-minus-
constant-mean effects, separately by seed. This report includes every one of
the 44 new E480 contrasts and all 12 primary effects across the three seeds.

## Main finding

The original seed's four primary schedule effects are positive. Both AdamW
effects have intervals above zero; the two MuonAdamW intervals include zero.
In E480's two new seeds, **all eight primary point estimates are negative**,
and **six of their eight 95% component-bootstrap intervals lie below zero**.
Seed 271829 has two negative intervals (both AdamW) and two Muon intervals
that cross zero. Seed 271830 has all four intervals below zero. The sign
reversal across fixed seeds is the main result: the first-seed warm-cosine
advantage did not replicate, so these runs do not support a seed-robust claim
that the warm-cosine schedule helps.

An independent pure-NumPy reconstruction also matched all 24 E480 arm scores
and all 44 global contrasts from the selected exports. Its maximum absolute
numerical difference was {float(numpy_reproduction["maximum_absolute_error"]):.3g}
(tolerance {float(numpy_reproduction["tolerance"]):.1g}); it reused exports and
did not refit, rerun inference, or inspect public results.

The 36 global-WP scores range from {min(score_values):.5f} to {max(score_values):.5f} across these same exposed components. Initialization-to-initialization
variation is large relative to many schedule contrasts. There are only three
fit initializations, so these seed-specific results characterize instability;
they do not estimate a population distribution over seeds. The component
bootstrap conditions on each fitted seed and resamples the same 16 previously
exposed components. It does not account for seed search, architecture or
schedule selection, training randomness beyond these three seeds, or transfer
to the competition's unseen public period. Do not pool the three seed scores
## Primary paired effects

Positive values favor warm-cosine; negative values favor the constant-mean
schedule. Intervals are the registered 2,000-resample paired component
bootstrap intervals.

| Seed | Run | Activation / optimizer | Δ global WP | 95% CI | P(Δ ≤ 0) |
|---:|---|---|---:|---:|---:|
{chr(10).join(primary_rows)}

## All 36 official arm scores

| Seed | Run | Activation | Optimizer | Schedule | Global WP | t0 | t1 |
|---:|---|---|---|---|---:|---:|---:|
{chr(10).join(score_rows)}

![All 36 score estimates on the same vertical scale]({plot_scores.relative_to(HERE)})

In the score panels, `B/S` denote base GELU/sine embedding, `A/M` denote
AdamW/MuonAdamW, and `P/W/C` denote constant-peak/warm-cosine/constant-mean.

![All 12 primary effects on the same horizontal scale]({plot_primary.relative_to(HERE)})

## All 44 E480 registered contrasts

The table is reported separately by replication seed. Percentile intervals are
component-resampling intervals over the same fixed 16 components; they are not
seed-level uncertainty intervals. For interactions the candidate column
contains the full difference-in-differences expression.

| Seed | Family | Contrast | Δ global WP | 95% CI | P(Δ ≤ 0) |
|---:|---|---|---:|---:|---:|
{chr(10).join(contrast_rows)}

## Provenance and limits

- E476 score file SHA-256: `{sha256_file(E476_SCORE)}`; its selected-export SHA-256: `{sha256_file(E476_EXPORT)}`.
- E480 score file SHA-256: `{sha256_file(E480_SCORE)}`; selected-export SHA-256: `{sha256_file(E480_EXPORT)}`.
- E480 grade attempt SHA-256: `{receipt["score_grade_attempt_sha256"]}`; sealed grade-spec digest: `{receipt["score_grade_spec_sha256"]}`.
- Independent NumPy reproduction source SHA-256: `{numpy_reproduction["source_sha256"]}`; maximum absolute error: `{numpy_reproduction["maximum_absolute_error"]}`.
- Metric: `{e480_doc["metric_id"]}`; fit seeds: `{", ".join(str(seed) for seed in SEEDS)}`; each seed is kept separate.
- E480's bootstrap generator seed is 271828; it is distinct from the model initialization seeds 271829 and 271830.
- Each component had prior exposure. These scores are descriptive paired diagnostics, not generalization evidence.
- The seed-271828 E476 run was the motivating result. Its positive direction is retained; it was not omitted after opposite replications.
"""


def main() -> None:
    """Verify grade proofs, render shared-scale figures, and write the report."""
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite three-seed render directory: {OUTPUT_DIR}")
    e476_doc, e480_doc = read_json(E476_SCORE), read_json(E480_SCORE)
    if (
        e476_doc.get("metric") != "global_wp_clip2_v1"
        or e480_doc.get("metric_id") != "global_wp_clip2_v1"
    ):
        raise ValueError("E476/E480 metric identities differ")
    if int(e476_doc.get("fit_seed", -1)) != 271_828:
        raise ValueError("original E476 fit seed changed")
    if tuple(e480_doc.get("fit_seeds", ())) != (271_829, 271_830):
        raise ValueError("E480 replication seed set changed")
    e480_arms = e480_doc.get("arm_count")
    if (
        e480_arms != 24
        or e480_doc.get("pair_contrast_count") != 40
        or e480_doc.get("joint_interaction_count") != 4
    ):
        raise ValueError("E480 registered score/contrast counts differ")
    if e480_doc.get("primary_warmcos_minus_constant_mean_count") != 8:
        raise ValueError("E480 primary contrast count differs")
    receipt = validate_grade_receipt(e480_doc)
    numpy_reproduction = validate_numpy_reproduction(e480_doc)
    arms = load_arm_scores(e476_doc, "E476") + load_arm_scores(e480_doc, "E480")
    effects = load_primary_effects(e476_doc, e480_doc)
    contrasts = load_e480_contrasts(e480_doc)
    OUTPUT_DIR.mkdir(parents=True)
    score_plot = OUTPUT_DIR / "three_seed_scores.png"
    primary_plot = OUTPUT_DIR / "three_seed_primary_contrasts.png"
    render_scores(arms, score_plot)
    render_primary(effects, primary_plot)
    report = make_report(
        e476_doc,
        e480_doc,
        arms,
        effects,
        contrasts,
        receipt,
        numpy_reproduction,
        score_plot,
        primary_plot,
    )
    report_path = HERE / "THREE_SEED_REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite three-seed report: {report_path}")
    report_path.write_text(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "plots": [str(score_plot), str(primary_plot)],
                "arms": len(arms),
                "primary_effects": len(effects),
                "e480_contrasts": len(contrasts),
                "grade_status": receipt["status"],
                "grade_exit_code": receipt["exit_code"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
