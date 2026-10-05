"""E489 fit-only H128 recurrent candidate duration experiment using E487 FP32 Muon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E471_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E487_DIR = ROOT / "research/campaign_2026-10-05/muon_orthogonalization_precision"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E471_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E487_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import precision_replay as e487  # noqa: E402
from fit_recurrent_candidate import scripted_candidate_scan  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import MuonAdamW  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271836, 271837, 271838)
ALPHAS = (0.0, 0.1)
HIDDEN = 128
FEATURES = 42
EXPECTED_PARAMETERS = 236_802
CHUNK_ROWS = 256
ROWS_PER_GROUP = 20_000
BATCH_SIZE = 4
UPDATES_PER_EPOCH = math.ceil(ROWS_PER_GROUP / CHUNK_ROWS) * (24 // BATCH_SIZE)
EPOCHS = 4
SNAPSHOT_EPOCHS = (1, 2, 4)
UPDATES_PER_TRAJECTORY = UPDATES_PER_EPOCH * EPOCHS
MUON_NAMES = {
    f"gru.{prefix}_l{layer}" for prefix in ("weight_ih", "weight_hh") for layer in range(2)
}
TELEMETRY_STEPS = (0, 46, 236, 473, 947, 1895)
CLIP_NORM = 1.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
E470_SOURCE_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E471_SOURCE_SHA256 = "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf"
E474_SOURCE_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_OPTIM_SHA256 = "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
DESIGN_SHA256 = "0859e6ac528b850452183afc46ab28358cac13c97ace5cb0350e413c2b9a7090"
CRITIC_SHA256 = "e4a8cae780b1e49739dd9474e3c13af459ef75db4bd39eb6e224cffe0ec76e70"
ADDENDUM_SHA256 = "9a1803680163d556037cabecfbc5cdfb923b30039140e64cda783493bfe2ae68"
E487_REPLAY_SHA256 = "480f2ae6d3aa09a3f972a3ad9fc682e7a92e07843d6e002434b2130ec75415d2"
E487_RESULT_SHA256 = "46b1edfc5b63d2da93ddc444b8c40142d4e57e16b6ceb4f80f33bc4fffd89714"
E487_EXPORT_SHA256 = "591a2eb248249f1925eca71c85031639b0085fbf58b148144f32bfd3b9a7c152"
E487_VERIFIER_SHA256 = "f717d0d30d1c6e1456b80bdd31d76896633a588ea22e5b9243652af5f92391e6"
E487_VERIFIER_REVIEW_SHA256 = "0685ce65335c956a94003700827798c8d3080d746bb876362074cc19545f9132"
E487_GATE_SHA256 = "4fced259c38beae9a74652cf486ccd952d62c830e5927e5316be161aba6c6cac"
E487_VERIFICATION_RECEIPT_SHA256 = (
    "8820f1eb3edf954f1d2279843be85d418a8883fad39fff3f9a9689396eeae6a9"
)
E487_ACTUAL_ATTEMPT_SHA256 = "03178e19bc242f698c7ffceb911b972299799ec9b909db6f964ec5cbc6ccc5af"
E487_DESIGN_SHA256 = "c937a9d6772f70cd56d83e7c9d6bfdde47a52c0e4bb9c66b0ca05a4e29fe8bd6"
E485_PRELIGHT_SHA256 = "bf23c9bdfb27a3f9367e8d305bcfa632bf410e6c0990fa2670e6538062bf3d8f"
E485_FAILURE_SHA256 = "1347acc2cb27f5b1a10c6a5aeda1962e7da77973a70a0865e203c7a64dde9adf"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"


def sha256_file(path: Path) -> str:
    """Return a streaming-safe SHA-256 digest for a bound receipt or source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DurationCandidateRNN(e474.PlacementRNN):
    """H128 incumbent topology with a fixed candidate-only sine perturbation."""

    def __init__(self, alpha: float) -> None:
        if alpha not in ALPHAS:
            raise ValueError(f"unregistered alpha: {alpha}")
        super().__init__("gelu", "gelu", hidden=HIDDEN)
        self.alpha = float(alpha)

    def forward_details(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("input must have shape (batch, time, 42)")
        encoded = self.encode((x - self.mean) / self.std)
        batch = encoded.shape[0]
        if state is None:
            hidden_state = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("state must have shape (2, batch, 128)")
            hidden_state = state
        layer_input = encoded
        final_states: list[Tensor] = []
        for layer in range(2):
            layer_output, final_hidden = scripted_candidate_scan(
                layer_input,
                hidden_state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.alpha,
            )
            layer_input = layer_output
            final_states.append(final_hidden)
        recurrent_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, recurrent_state, layer_input

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Keep the registered public model output/state API unchanged."""
        prediction, recurrent_state, _candidate = self.forward_details(x, state)
        return prediction, recurrent_state


def parent_source_hashes() -> dict[str, str]:
    """Check E485 topology, E487 FP32 optimizer, data and preregistrations."""
    files = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SOURCE_SHA256),
        "e471_scan": (E471_DIR / "fit_recurrent_candidate.py", E471_SOURCE_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SOURCE_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.optim": (ROOT / "src/wnn/optim.py", WNN_OPTIM_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
        "e487_fp32_replay": (E487_DIR / "precision_replay.py", E487_REPLAY_SHA256),
        "e487_probe_result": (E487_DIR / "gpu_probe_v2/result.json", E487_RESULT_SHA256),
        "e487_replay_export": (E487_DIR / "gpu_probe_v2/replay_deltas.npz", E487_EXPORT_SHA256),
        "e487_verifier_v4": (E487_DIR / "verify_probe_v4.py", E487_VERIFIER_SHA256),
        "e487_root_gate": (
            ROOT / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json",
            E487_GATE_SHA256,
        ),
        "e487_verification_receipt": (
            ROOT / "research/campaign_2026-10-05/checks/e487_saved_delta_verification_v4.json",
            E487_VERIFICATION_RECEIPT_SHA256,
        ),
    }
    actual = {name: e474.file_hash(path) for name, (path, _) in files.items()}
    for name, (_, expected) in files.items():
        if actual[name] != expected:
            raise ValueError(f"frozen parent source changed: {name}={actual[name]}")
    docs = {
        "e489_design": (HERE / "E489_DESIGN.md", DESIGN_SHA256),
        "e489_critic": (HERE / "SPEC_CRITIC.md", CRITIC_SHA256),
        "e489_addendum": (HERE / "PROVENANCE_ADMISSION_ADDENDUM.md", ADDENDUM_SHA256),
        "e485_design": (
            ROOT / "research/campaign_2026-10-05/activation_duration/E485_DESIGN.md",
            "bcbf17ec7b632909e9e8792cdf11436822dd4dcac5326befa506c8dcb96e70cc",
        ),
        "e487_design": (E487_DIR / "E487_DESIGN.md", E487_DESIGN_SHA256),
        "e487_critic": (
            E487_DIR / "SPEC_CRITIC.md",
            "a65ce0d723b0857972983ed37742c87bb884764221705c5c69d4931d1c3ca60c",
        ),
        "e487_verifier_review": (
            E487_DIR / "VERIFY_PROBE_REVIEW_V4.md",
            E487_VERIFIER_REVIEW_SHA256,
        ),
        "e485_preflight": (
            ROOT
            / "research/campaign_2026-10-05/activation_duration/cpu_preflight_v7/fit_manifest.json",
            E485_PRELIGHT_SHA256,
        ),
        "e485_failure": (
            ROOT / "research/campaign_2026-10-05/activation_duration/ACTUAL_PROBE_V3_FAILURE.md",
            E485_FAILURE_SHA256,
        ),
    }
    for name, (path, expected) in docs.items():
        if e474.file_hash(path) != expected:
            raise ValueError(f"registered E489 dependency changed: {name}")
    data_hash = e474.file_hash(e470.DATA_PATH)
    cohort_hash = e474.file_hash(e470.COHORT_PATH)
    if data_hash != e474.DATASET_SHA256 or cohort_hash != COHORT_SHA256:
        raise ValueError("E489 frozen data or cohort manifest changed")
    preflight = json.loads(
        (
            ROOT
            / "research/campaign_2026-10-05/activation_duration/cpu_preflight_v7/fit_manifest.json"
        ).read_text()
    )
    e487_result = json.loads((E487_DIR / "gpu_probe_v2/result.json").read_text())
    if (
        preflight.get("seeds") != list(SEEDS)
        or preflight.get("test_rows_loaded") is not False
        or preflight.get("status") != "preflight_passed"
    ):
        raise ValueError("E485 seed prior-exposure receipt is invalid")
    if not e487_result.get("fit_only") or e487_result.get("test_rows_loaded") is not False:
        raise ValueError("E487 optimizer precision diagnostic read evaluation rows")
    return {**actual, "dataset": data_hash, "cohort": cohort_hash}


def validate_e487_saved_delta_gate(path: Path) -> dict:
    """Require root's independent saved-delta arithmetic receipt before CUDA."""
    proof = json.loads(path.read_text())
    if (
        proof.get("study") != "E487"
        or proof.get("status") != "all_checked_values_match"
        or proof.get("source_sha256") != E487_REPLAY_SHA256
        or proof.get("result_sha256") != E487_RESULT_SHA256
        or proof.get("export_sha256") != E487_EXPORT_SHA256
        or proof.get("verifier_source_sha256") != E487_VERIFIER_SHA256
        or proof.get("verifier_review_sha256") != E487_VERIFIER_REVIEW_SHA256
        or proof.get("actual_attempt_sha256") != E487_ACTUAL_ATTEMPT_SHA256
        or proof.get("strict_gate_fp32_pass") is not True
        or proof.get("strict_gate_bf16_pass") is not False
        or proof.get("all_delta_arrays_finite") is not True
        or proof.get("source_identities_current_match_actual_sealed_probe") is not True
        or proof.get("actual_teardown_and_unchanged_oom_counters_verified") is not True
        or proof.get("new_training_fits") != 0
    ):
        raise ValueError("E489 requires the exact independent E487 saved-delta reproduction gate")
    receipt_relative = proof.get("verification_receipt_path")
    receipt_hash = proof.get("verification_receipt_sha256")
    if not isinstance(receipt_relative, str) or not isinstance(receipt_hash, str):
        raise ValueError("E487 root gate omitted the verification receipt identity")
    receipt_path = (ROOT / receipt_relative).resolve()
    checks_root = (ROOT / "research/campaign_2026-10-05/checks").resolve()
    if (
        receipt_relative
        != "research/campaign_2026-10-05/checks/e487_saved_delta_verification_v4.json"
        or receipt_hash != E487_VERIFICATION_RECEIPT_SHA256
        or checks_root not in receipt_path.parents
        or sha256_file(receipt_path) != receipt_hash
    ):
        raise ValueError("E487 verification receipt path/hash is invalid")
    return proof


def make_initial_states() -> dict[int, dict[str, Tensor]]:
    """Create one native RNNModel initialization per seed for both shapes."""
    states: dict[int, dict[str, Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
            raise ValueError("native model parameter count changed")
        states[seed] = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return states


def initialize_arm(alpha: float, state: dict[str, Tensor]) -> DurationCandidateRNN:
    """Load the shared native initialization into a parameter-free variant."""
    model = DurationCandidateRNN(alpha)
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("candidate parameter count differs from registered H128 model")
    return model


def make_optimizer(model: DurationCandidateRNN) -> MuonAdamW:
    """Use the frozen E487 FP32 orthogonalizer over exact E485 groups."""
    optimizer = e487.make_optimizer(model, precision="fp32_diagnostic")
    groups = [group for group in optimizer.param_groups if group.get("muon")]
    names = {name for group in groups for name in group.get("_e487_names", [])}
    if names != MUON_NAMES or len(groups) != 1 or groups[0].get("blocks") != 3:
        raise ValueError(f"unexpected E487 recurrent Muon assignment: {names}")
    if len(optimizer.param_groups) != 2:
        raise ValueError("E487 optimizer must have one Muon and one AdamW group")
    for group in optimizer.param_groups:
        if (group["lr"], group["weight_decay"], tuple(group["betas"])) != (
            LEARNING_RATE,
            WEIGHT_DECAY,
            BETAS,
        ):
            raise ValueError("E487 optimizer hyperparameters differ from E489 registration")
    return optimizer


def canonical_epoch_order(seed: int, epoch_index: int) -> np.ndarray:
    """Shuffle indices in the cohort's canonical 24-fit-role order."""
    if epoch_index not in range(EPOCHS):
        raise ValueError(f"invalid epoch index: {epoch_index}")
    order = np.arange(24)
    np.random.default_rng(seed + epoch_index).shuffle(order)
    return order


def optimizer_update(
    model: DurationCandidateRNN,
    optimizer: MuonAdamW,
    update_index: int,
    epoch_index: int,
    owner: str,
    batch_index: int = -1,
    chunk_start: int = -1,
    alpha: float = 0.0,
    state: Tensor | None = None,
    prediction: Tensor | None = None,
    candidate: Tensor | None = None,
) -> tuple[bool, dict | None]:
    """Check raw gradients, clip globally, step once, and report fixed samples."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if len(gradients) != len(list(model.parameters())):
        raise ValueError("every trainable parameter must receive a gradient")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise ValueError(f"nonfinite raw gradient in {owner}")
    raw_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient.detach(), 2) for gradient in gradients]), 2
    )
    if not torch.isfinite(raw_norm):
        raise ValueError(f"nonfinite global gradient norm in {owner}")
    clipped = bool(raw_norm.item() > CLIP_NORM)
    sampled = update_index in TELEMETRY_STEPS
    named = dict(model.named_parameters())
    families: dict[str, list[Tensor]] = {
        f"matrix:{name}": [named[name]] for name in sorted(MUON_NAMES)
    }
    families["adamw"] = [parameter for name, parameter in named.items() if name not in MUON_NAMES]
    gradient_rms = None
    parameter_before = None
    if sampled:
        gradient_rms = e474.family_rms(
            {
                family: [parameter.grad for parameter in parameters]
                for family, parameters in families.items()
            }
        )
        parameter_before = {
            family: [parameter.detach().clone() for parameter in parameters]
            for family, parameters in families.items()
        }
    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    e470.ensure_finite_parameters(model, owner)
    e487.require_optimizer_state_finite(optimizer, owner)
    telemetry = None
    if sampled:
        if gradient_rms is None or parameter_before is None:
            raise AssertionError("sampled telemetry snapshots are missing")
        updates = {
            family: [
                parameter.detach() - previous
                for parameter, previous in zip(
                    families[family], parameter_before[family], strict=True
                )
            ]
            for family in families
        }
        telemetry = {
            "global_update_index_zero_based": update_index,
            "epoch_index_zero_based": epoch_index,
            "batch_index_zero_based": batch_index,
            "chunk_start": chunk_start,
            "alpha": alpha,
            "learning_rate": LEARNING_RATE,
            "preclip_gradient_rms": gradient_rms,
            "raw_gradient_global_norm": float(raw_norm.detach().cpu()),
            "parameter_update_rms": e474.family_rms(updates),
            "matrix_adam_family_update_rms": e474.family_rms(updates),
            "clip_factor": min(1.0, CLIP_NORM / (float(raw_norm.detach().cpu()) + 1e-6)),
            "state_abs_max": None if state is None else float(state.detach().abs().max().cpu()),
            "prediction_abs_max": None
            if prediction is None
            else float(prediction.detach().abs().max().cpu()),
            "candidate_recurrent_output_abs_max": None
            if candidate is None
            else float(candidate.detach().abs().max().cpu()),
        }
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise ValueError(f"optimizer produced nonfinite parameters in {owner}")
    return clipped, telemetry


def fit_only_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Make one actual B4×256 update for each seed and fixed shape."""
    if x.shape != (24, ROWS_PER_GROUP, FEATURES):
        raise ValueError(f"unexpected fit array: {x.shape}")
    if y.shape != (24, ROWS_PER_GROUP, 2) or need.shape != (24, ROWS_PER_GROUP):
        raise ValueError("unexpected labels or mask shape")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("preflight must contain only the canonical 24 fit roles")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only cohort inputs")
    receipts: list[dict] = []
    for seed in SEEDS:
        ids = canonical_epoch_order(seed, 0)[:BATCH_SIZE].tolist()
        for alpha in ALPHAS:
            model = initialize_arm(alpha, initial_states[seed])
            optimizer = make_optimizer(model)
            xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).float()
            yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).float()
            mask = torch.from_numpy(need[ids, :CHUNK_ROWS])
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError("nonfinite raw preflight batch")
            optimizer.zero_grad(set_to_none=True)
            prediction, state, candidate = model.forward_details(xx)
            if not (
                torch.isfinite(prediction).all()
                and torch.isfinite(state).all()
                and torch.isfinite(candidate).all()
            ):
                raise ValueError(f"nonfinite preflight forward at {seed}/alpha={alpha}")
            loss = e470.loss_for_batch(prediction, yy, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite fit-only loss")
            loss.backward()
            before = [parameter.detach().clone() for parameter in model.parameters()]
            clipped, telemetry = optimizer_update(
                model,
                optimizer,
                0,
                0,
                f"preflight/{seed}/alpha={alpha}",
                batch_index=0,
                chunk_start=0,
                alpha=alpha,
                state=state,
                prediction=prediction,
                candidate=candidate,
            )
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if not changed:
                raise ValueError(f"preflight optimizer made no change: {seed}/alpha={alpha}")
            receipts.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "batch": BATCH_SIZE,
                    "chunk_rows": CHUNK_ROWS,
                    "updates": 1,
                    "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                    "loss": float(loss.detach()),
                    "clip_applied": clipped,
                    "parameters_changed": changed,
                    "parameters_finite": True,
                    "telemetry": telemetry,
                }
            )
    return receipts


def save_checkpoint(
    model: DurationCandidateRNN, output: Path, seed: int, alpha: float, epoch: int
) -> tuple[str, str]:
    """Write an immutable CPU checkpoint for one registered trajectory snapshot."""
    path = output / f"seed{seed}_alpha{alpha:.1f}_epoch{epoch}.pt"
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return path.name, e474.file_hash(path)


def run_trajectory(
    seed: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    initial_state: dict[str, Tensor],
    active_precision: dict,
) -> dict:
    """Train four continuous epochs and save the three registered snapshots."""
    model = initialize_arm(alpha, initial_state).to(device).float()
    optimizer = make_optimizer(model)
    records: list[dict] = []
    global_step = 0
    total_clipped = 0
    started = time.monotonic()
    for epoch_index in range(EPOCHS):
        order = canonical_epoch_order(seed, epoch_index)
        epoch_groups: list[str] = []
        clipped_this_epoch = 0
        epoch_telemetry: list[dict] = []
        for batch_start in range(0, 24, BATCH_SIZE):
            ids = order[batch_start : batch_start + BATCH_SIZE].tolist()
            epoch_groups.extend(nodes[index]["row_group"] for index in ids)
            state: Tensor | None = None
            for position in range(0, ROWS_PER_GROUP, CHUNK_ROWS):
                xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(
                    device=device, dtype=torch.float32
                )
                yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(
                    device=device, dtype=torch.float32
                )
                mask = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(
                    device=device
                )
                if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                    raise ValueError(f"nonfinite raw fit batch at {seed}/{alpha}/{global_step}")
                optimizer.zero_grad(set_to_none=True)
                prediction, state, candidate = model.forward_details(xx, state)
                if not (
                    torch.isfinite(prediction).all()
                    and torch.isfinite(state).all()
                    and torch.isfinite(candidate).all()
                ):
                    raise ValueError(f"nonfinite forward at {seed}/{alpha}/{global_step}")
                loss = e470.loss_for_batch(prediction, yy, mask)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at {seed}/{alpha}/{global_step}")
                loss.backward()
                clipped, telemetry = optimizer_update(
                    model,
                    optimizer,
                    global_step,
                    epoch_index,
                    f"fit/{seed}/alpha={alpha}",
                    batch_index=batch_start // BATCH_SIZE,
                    chunk_start=position,
                    alpha=alpha,
                    state=state,
                    prediction=prediction,
                    candidate=candidate,
                )
                clipped_this_epoch += int(clipped)
                if telemetry is not None:
                    telemetry.update(
                        {
                            "row_groups": [nodes[index]["row_group"] for index in ids],
                            "chunk_start": position,
                        }
                    )
                    epoch_telemetry.append(telemetry)
                state = state.detach()
                global_step += 1
        if len(epoch_groups) != 24 or len(set(epoch_groups)) != 24:
            raise ValueError("epoch order must contain each fit role exactly once")
        if global_step != (epoch_index + 1) * UPDATES_PER_EPOCH:
            raise ValueError("unexpected continuous trajectory update count")
        adam_steps = sorted(
            {
                int(state["step"].item() if isinstance(state["step"], Tensor) else state["step"])
                for state in optimizer.state.values()
                if "step" in state
            }
        )
        if adam_steps != [global_step]:
            raise ValueError(f"E489 AdamW moments did not continue at {global_step}: {adam_steps}")
        total_clipped += clipped_this_epoch
        snapshot: dict | None = None
        if epoch_index + 1 in SNAPSHOT_EPOCHS:
            filename, digest = save_checkpoint(model, output, seed, alpha, epoch_index + 1)
            snapshot = {"checkpoint": filename, "sha256": digest}
        records.append(
            {
                "epoch": epoch_index + 1,
                "updates_this_epoch": UPDATES_PER_EPOCH,
                "total_updates": global_step,
                "fit_row_groups_in_order": epoch_groups,
                "clipped_updates": clipped_this_epoch,
                "cumulative_clipped_updates": total_clipped,
                "telemetry": epoch_telemetry,
                "snapshot": snapshot,
            }
        )
        write_progress_manifest(
            output, seed, alpha, records, total_clipped, global_step, active_precision
        )
    if global_step != UPDATES_PER_TRAJECTORY or len(records) != EPOCHS:
        raise ValueError("trajectory did not complete all four epochs")
    return {
        "seed": seed,
        "alpha": alpha,
        "status": "finished",
        "parameters": EXPECTED_PARAMETERS,
        "updates": global_step,
        "clipped_updates": total_clipped,
        "fit_seconds": time.monotonic() - started,
        "epochs": records,
    }


def write_progress_manifest(
    output: Path,
    seed: int,
    alpha: float,
    epochs: list[dict],
    clipped: int,
    updates: int,
    active_precision: dict,
) -> None:
    """Persist per-trajectory progress after each completed epoch."""
    path = output / f"seed{seed}_alpha{alpha:.1f}_progress.json"
    payload = {
        "seed": seed,
        "alpha": alpha,
        "completed_epochs": len(epochs),
        "updates": updates,
        "clipped_updates": clipped,
        "active_precision": active_precision,
        "epochs": epochs,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    """Run only a fit-only preflight or separately admitted E489 fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-real", action="store_true")
    modes.add_argument("--run-fits", action="store_true")
    parser.add_argument(
        "--e487-root-proof",
        type=Path,
        default=ROOT / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("E489 real-data fit-only preflight is CPU-only")
    if args.run_fits and args.device != "cuda":
        raise ValueError("E489 full trajectories require separately admitted CUDA")
    torch.set_num_threads(1)
    hashes = parent_source_hashes()
    gate = None
    if args.run_fits:
        gate = validate_e487_saved_delta_gate(args.e487_root_proof)
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    hashes = parent_source_hashes()
    initial_states = make_initial_states()
    cohort, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E485 can load only the 24 fit-role row groups")
    if x.shape != (24, ROWS_PER_GROUP, FEATURES):
        raise ValueError(f"unexpected fit cohort shape: {x.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit cohort data")
    result = {
        "kind": "E489 H128 activation duration with E487 FP32 Muon",
        "status": "preflight" if args.preflight_real else "fitting",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "alphas": list(ALPHAS),
        "hidden": HIDDEN,
        "parameters_per_trajectory": EXPECTED_PARAMETERS,
        "epochs": EPOCHS,
        "snapshots_per_trajectory": list(SNAPSHOT_EPOCHS),
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "chunk_rows": CHUNK_ROWS,
        "candidate": "u=tanh(v); n=u+alpha*sin(pi*u); h'=(1-z)*n+z*h",
        "optimizer": {
            "name": "E487 FP32MuonAdamW diagnostic implementation",
            "muon_names": sorted(MUON_NAMES),
            "muon_blocks": 3,
            "muon_beta1": 0.9,
            "adam_betas": list(BETAS),
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "schedule": "constant",
            "global_clip_norm": CLIP_NORM,
            "state_continues_across_epochs": True,
            "cuda_muon_orthogonalization": "E487 FP32 five-step Newton-Schulz diagnostic path",
        },
        "shuffle": "canonical np.arange(24), default_rng(seed+epoch_index).shuffle",
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "parent_source_sha256": hashes,
        "design_sha256": DESIGN_SHA256,
        "blind_critic_sha256": CRITIC_SHA256,
        "provenance_addendum_sha256": ADDENDUM_SHA256,
        "e487_saved_delta_gate": gate,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in (
                "fit_fp32_duration.py",
                "tests_fp32_duration.py",
                "gpu_probe_fp32_duration.py",
                "PROVENANCE_ADMISSION_ADDENDUM.md",
            )
            if (HERE / name).is_file()
        },
        "cohort_manifest": cohort,
        "trajectories": [],
    }
    args.output.mkdir(parents=True)
    (args.output / "fit_only_cohort.json").write_text(json.dumps(cohort, indent=2))
    if args.preflight_real:
        with e474.full_ieee_float32():
            result["active_precision"] = e474.precision_metadata()
            result["preflight_arms"] = fit_only_preflight(x, y, need, nodes, initial_states)
        result["status"] = "preflight_passed"
        (args.output / "fit_manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    else:
        with e474.full_ieee_float32():
            result["active_precision"] = e474.precision_metadata()
            for seed in SEEDS:
                for alpha in ALPHAS:
                    trajectory = run_trajectory(
                        seed,
                        alpha,
                        x,
                        y,
                        need,
                        nodes,
                        device,
                        args.output,
                        initial_states[seed],
                        result["active_precision"],
                    )
                    result["trajectories"].append(trajectory)
                    temporary = args.output / "fit_manifest.json.tmp"
                    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    os.replace(temporary, args.output / "fit_manifest.json")
        if len(result["trajectories"]) != 6:
            raise ValueError("all six registered trajectories must finish")
        result["status"] = "all_six_trajectories_complete_test_unread"
        temporary = args.output / "fit_manifest.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, args.output / "fit_manifest.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
