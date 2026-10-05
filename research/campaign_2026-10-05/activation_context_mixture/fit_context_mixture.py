"""Fit the preregistered E484 causal context-conditioned head mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E477_DIR = ROOT / "research/campaign_2026-10-05/activation_head_mixture"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E477_DIR))

import fit_activation_head_mixture as e477  # noqa: E402
import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271834, 271835)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix", "projection_mix", "self_mix")
MIXTURE_ARMS = ARMS[1:]
CONTEXT_MODES = {
    "recurrent_mix": "recurrent",
    "projection_mix": "projection",
    "self_mix": "head_self",
}
CHUNK_ROWS, UPDATES, FEATURES, HIDDEN = 256, 474, 42, 64
BASE_PARAMETERS, GATED_PARAMETERS = 61_058, 61_122
TELEMETRY_STEPS = (0, 46, 236, 473)
CLIP_NORM, LR, WEIGHT_DECAY, BETAS = 1.0, 0.001, 0.01, (0.9, 0.999)
CONTRASTS_PER_SEED = (
    "recurrent_mix - static_mix",
    "recurrent_mix - projection_mix",
    "recurrent_mix - self_mix",
    "static_mix - fixed_mix",
    "recurrent_mix - fixed_mix",
    "projection_mix - fixed_mix",
    "self_mix - fixed_mix",
    "gelu - fixed_mix",
)
GATE_EPSILON = 1e-6
DESIGN_V2_SHA256 = "e2657b85e31c10d0bab3edab066ba71faa6f2eee79c5cddfd9aff86c6f89b398"
CRITIC_SHA256 = "7f1325b9f9517199bb0a2af08a4a86a7625c917814458bf17a254fe9b79303d4"
CRITIC_RESPONSE_SHA256 = "9325be932eb0427db6631bc616b35dfbe285da78c876df99c966b19b8817315e"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
E477_SHA256 = "2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"


def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources() -> dict[str, str]:
    """Validate design, frozen parent code, dataset and cohort identities."""
    sources = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SHA256),
        "e477": (E477_DIR / "fit_activation_head_mixture.py", E477_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
    }
    actual = {name: file_sha256(path) for name, (path, _expected) in sources.items()}
    for name, (_path, expected) in sources.items():
        if actual[name] != expected:
            raise ValueError(f"frozen parent source mismatch: {name}={actual[name]}")
    documents = {
        "design_v2": (HERE / "E484_DESIGN_V2.md", DESIGN_V2_SHA256),
        "blind_critic": (HERE / "SPEC_CRITIC.md", CRITIC_SHA256),
        "critic_response": (HERE / "SPEC_CRITIC_V2_RESPONSE.md", CRITIC_RESPONSE_SHA256),
    }
    for name, (path, expected) in documents.items():
        if file_sha256(path) != expected:
            raise ValueError(f"registered E484 document changed: {name}")
    if file_sha256(e470.DATA_PATH) != DATASET_SHA256:
        raise ValueError("frozen E484 dataset bytes changed")
    if file_sha256(e470.COHORT_PATH) != COHORT_SHA256:
        raise ValueError("frozen E484 cohort manifest changed")
    return actual


def make_initial_states() -> dict[int, dict[str, Tensor]]:
    """Create one E474 initialization per new seed, shared bytewise across arms."""
    states: dict[int, dict[str, Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = e474.PlacementRNN("gelu", "gelu")
        if sum(parameter.numel() for parameter in baseline.parameters()) != BASE_PARAMETERS:
            raise ValueError("frozen E474 model parameter count changed")
        states[seed] = {
            name: value.detach().clone() for name, value in baseline.state_dict().items()
        }
    if len({tensor_state_sha256(state) for state in states.values()}) != len(SEEDS):
        raise ValueError("fresh E484 seed initializations are not distinct")
    return states


def tensor_state_sha256(state: dict[str, Tensor]) -> str:
    """Hash stable tensor keys, dtypes, shapes and contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ContextMixtureActivation(nn.Module):
    """GELU, fixed mixture or registered zero-start per-channel mixture gate."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unregistered E484 arm: {arm}")
        self.arm = arm
        if arm in ("static_mix", *CONTEXT_MODES):
            self.gate_coeff = nn.Parameter(torch.zeros(HIDDEN))

    def forward_details(
        self, hidden: Tensor, recurrent: Tensor, projected: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Return activated output and optional gate/context for fixed diagnostics."""
        gelu = F.gelu(hidden, approximate="none")
        if self.arm == "gelu":
            return gelu, None, None
        leaky = F.leaky_relu(hidden, negative_slope=0.01)
        if self.arm == "fixed_mix":
            gate = torch.full_like(hidden, 0.5)
            context = None
        elif self.arm == "static_mix":
            gate = torch.sigmoid(self.gate_coeff).view(1, 1, HIDDEN).expand_as(hidden)
            context = None
        else:
            source = CONTEXT_MODES[self.arm]
            context = {"recurrent": recurrent, "projection": projected, "head_self": hidden}[source]
            normalized = normalize_context(context)
            gate = torch.sigmoid(self.gate_coeff.view(1, 1, HIDDEN) * normalized)
        return gate * gelu + (1.0 - gate) * leaky, gate, context

    def forward(self, hidden: Tensor, recurrent: Tensor, projected: Tensor) -> Tensor:
        """Apply the selected pointwise readout activation."""
        output, _gate, _context = self.forward_details(hidden, recurrent, projected)
        return output


class ContextMixtureRNN(e474.PlacementRNN):
    """E474 topology with one registered activation intervention after head.0."""

    def __init__(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unregistered E484 arm: {arm}")
        super().__init__(embedding_activation="gelu", head_activation="gelu")
        self.arm = arm
        self.head[1] = ContextMixtureActivation(arm)

    def forward_details(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Return predictions/state plus gate and dynamic context for audits."""
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("expected inputs shaped [batch,time,42]")
        projected = self.encode((x - self.mean) / self.std)
        recurrent, final_state = self.gru(projected, state)
        features = torch.cat((recurrent, projected), dim=-1)
        hidden = self.head[0](features)
        activated, gate, context = self.head[1].forward_details(hidden, recurrent, projected)
        prediction = self.head[2](activated)
        return prediction, final_state, gate, context

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return predictions and carried two-layer GRU state."""
        prediction, final_state, _gate, _context = self.forward_details(x, state)
        return prediction, final_state


def initialize_arm(arm: str, initial_state: dict[str, Tensor]) -> ContextMixtureRNN:
    """Load a common exact E474 baseline state into an E484 arm."""
    model = ContextMixtureRNN(arm)
    missing, unexpected = model.load_state_dict(initial_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected common-state keys for {arm}: {unexpected}")
    expected_missing = ["head.1.gate_coeff"] if arm in ("static_mix", *CONTEXT_MODES) else []
    if sorted(missing) != sorted(expected_missing):
        raise ValueError(f"unexpected missing common-state keys for {arm}: {missing}")
    expected = GATED_PARAMETERS if arm in ("static_mix", *CONTEXT_MODES) else BASE_PARAMETERS
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise ValueError(f"E484 {arm} must have {expected} parameters, got {count}")
    return model


def normalize_context(context: Tensor) -> Tensor:
    """Apply the registered non-centered per-sample/time channel RMS formula."""
    if context.ndim != 3 or context.shape[-1] != HIDDEN:
        raise ValueError("context must have shape [batch,time,64]")
    rms = torch.sqrt(context.square().mean(dim=-1, keepdim=True) + GATE_EPSILON)
    return context / rms


def check_common_parameter_bytes(
    model: ContextMixtureRNN, initial_state: dict[str, Tensor]
) -> None:
    """Check all inherited parameters and buffers still match the shared state."""
    current = model.state_dict()
    for name, value in initial_state.items():
        if name not in current or not torch.equal(current[name], value):
            raise ValueError(f"base initialization changed in {model.arm}: {name}")


def make_optimizer(model: ContextMixtureRNN) -> torch.optim.AdamW:
    """Return the unchanged E474 AdamW over baseline plus optional head gate."""
    return e474.make_optimizer(model)


def parameter_families(model: ContextMixtureRNN) -> dict[str, list[nn.Parameter]]:
    """Reuse E474's telemetry grouping; gate is nested under the existing head."""
    return e474.parameter_families(model)


def check_reference_parity(device: torch.device, state: dict[str, Tensor]) -> dict:
    """Compare E484 GELU/fixed-mixture arms to frozen E477 outputs and gradients."""
    torch.manual_seed(48401)
    inputs = torch.randn(2, 23, FEATURES, device=device)
    checks = []
    for arm, reference_arm in (("gelu", "gelu"), ("fixed_mix", "same_channel_mix")):
        candidate = initialize_arm(arm, state).to(device=device, dtype=torch.float32).train()
        reference = (
            e477.initialize_arm(reference_arm, 0.0, state)
            .to(device=device, dtype=torch.float32)
            .train()
        )
        candidate_input = inputs.detach().clone().requires_grad_(True)
        reference_input = inputs.detach().clone().requires_grad_(True)
        candidate_prediction, candidate_state = candidate(candidate_input)
        reference_prediction, reference_state = reference(reference_input)
        if not torch.equal(candidate_prediction, reference_prediction):
            raise ValueError(f"initial {arm} prediction differs from E477 reference")
        if not torch.equal(candidate_state, reference_state):
            raise ValueError(f"initial {arm} state differs from E477 reference")
        candidate_objective = 1000.0 * (
            candidate_prediction.square().mean() + candidate_state.square().mean()
        )
        reference_objective = 1000.0 * (
            reference_prediction.square().mean() + reference_state.square().mean()
        )
        candidate_objective.backward()
        reference_objective.backward()
        if not torch.equal(candidate_input.grad, reference_input.grad):
            raise ValueError(f"initial {arm} input gradient differs from E477 reference")
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            if name not in reference_parameters or not torch.equal(
                parameter.grad, reference_parameters[name].grad
            ):
                raise ValueError(f"initial {arm} parameter gradient differs: {name}")
        candidate_norm = torch.linalg.vector_norm(
            torch.stack([parameter.grad.detach().norm() for parameter in candidate.parameters()])
        )
        reference_norm = torch.linalg.vector_norm(
            torch.stack([parameter.grad.detach().norm() for parameter in reference.parameters()])
        )
        if not torch.equal(candidate_norm, reference_norm) or candidate_norm <= CLIP_NORM:
            raise ValueError(f"{arm} parity fixture did not reach matched clipping")
        candidate_optimizer = make_optimizer(candidate)
        reference_optimizer = e474.make_optimizer(reference)
        e474.apply_optimizer_update(
            candidate, candidate_optimizer, parameter_families(candidate), 0, f"parity/{arm}"
        )
        e474.apply_optimizer_update(
            reference,
            reference_optimizer,
            e474.parameter_families(reference),
            0,
            f"parity/{reference_arm}",
        )
        updated_reference = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            if not torch.equal(parameter, updated_reference[name]):
                raise ValueError(f"clipped AdamW update differs for {arm}: {name}")
        checks.append(
            {
                "arm": arm,
                "reference": reference_arm,
                "exact_forward_state_input_and_parameter_gradients": True,
                "preclip_gradient_norm": float(candidate_norm.detach().cpu()),
                "clipping_exercised": True,
                "exact_clipped_adamw_step": True,
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"device": device.type, "checks": checks}


def load_fit_cohort() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load only the E470 24-component fit cohort and validate its shape/role."""
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if (
        x.shape != (24, 20_000, FEATURES)
        or y.shape != (24, 20_000, 2)
        or need.shape != (24, 20_000)
        or len(nodes) != 24
        or any(node["role"] != "fit" for node in nodes)
    ):
        raise ValueError("E484 loader did not return exactly 24 fit-only components")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("E484 fit cohort contains nonfinite arrays")
    return manifest, x, y, need, nodes


def one_update(
    model: ContextMixtureRNN,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
    owner: str,
) -> tuple[float, dict]:
    """Apply one actual fit-only B4×256 update and report gate diagnostics."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device=device)
    if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
        raise ValueError(f"nonfinite fit inputs: {owner}")
    optimizer.zero_grad(set_to_none=True)
    prediction, state, gate, context = model.forward_details(xx)
    if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
        raise ValueError(f"nonfinite fit prediction/state: {owner}")
    if gate is not None and not torch.isfinite(gate).all():
        raise ValueError(f"nonfinite gate: {owner}")
    loss = e470.loss_for_batch(prediction, yy, mask)
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite weighted fit loss: {owner}")
    loss.backward()
    telemetry = e474.apply_optimizer_update(model, optimizer, parameter_families(model), 0, owner)
    if telemetry is None:
        raise AssertionError("step-zero telemetry missing")
    telemetry["gate"] = gate_summary(gate)
    if context is not None and not torch.isfinite(context).all():
        raise ValueError(f"nonfinite gate context: {owner}")
    return float(loss.detach().cpu()), telemetry


def gate_summary(gate: Tensor | None) -> dict[str, float] | None:
    """Summarize gate values without changing the training objective."""
    if gate is None:
        return None
    detached = gate.detach()
    return {
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "mean": float(detached.mean().cpu()),
        "saturated_fraction_at_0_01": float(
            ((detached < 0.01) | (detached > 0.99)).float().mean().cpu()
        ),
    }


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Run one fit-only real-data CPU B4×256 update in all twelve arms."""
    receipts: list[dict] = []
    for seed in SEEDS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        row_groups = [nodes[int(index)]["row_group"] for index in ids]
        for arm in ARMS:
            torch.manual_seed(seed)
            model = initialize_arm(arm, states[seed])
            check_common_parameter_bytes(model, states[seed])
            optimizer = make_optimizer(model)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss, telemetry = one_update(
                model,
                optimizer,
                x,
                y,
                need,
                ids,
                torch.device("cpu"),
                f"{seed}/{arm}",
            )
            changed = [
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            ]
            if not any(changed) or not all(
                torch.isfinite(parameter).all().item() for parameter in model.parameters()
            ):
                raise ValueError(f"CPU update failed finite/change check: {seed}/{arm}")
            receipts.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "fit_row_groups": row_groups,
                    "batch": 4,
                    "chunk_rows": CHUNK_ROWS,
                    "updates": 1,
                    "loss": loss,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "changed_parameter_tensors": sum(changed),
                    "parameters_finite": True,
                    "optimizer": optimizer_description(model, optimizer),
                    "telemetry_step0": telemetry,
                }
            )
    if len(receipts) != 12:
        raise ValueError("E484 preflight omitted a registered seed/arm")
    return receipts


def optimizer_description(model: ContextMixtureRNN, optimizer: torch.optim.Optimizer) -> dict:
    """Return the effective, unmodified E474 AdamW group settings."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("E484 must use one unchanged E474 AdamW group")
    group = optimizer.param_groups[0]
    expected = (LR, WEIGHT_DECAY, BETAS)
    actual = (group["lr"], group["weight_decay"], tuple(group["betas"]))
    if actual != expected:
        raise ValueError(f"E474 optimizer settings changed: {actual}")
    model_parameters = {id(parameter): name for name, parameter in model.named_parameters()}
    group_ids = [id(parameter) for parameter in group["params"]]
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != set(model_parameters):
        raise ValueError("E484 AdamW group does not contain every model parameter exactly once")
    return {
        "name": "AdamW",
        "lr": group["lr"],
        "weight_decay": group["weight_decay"],
        "betas": list(group["betas"]),
        "groups": len(optimizer.param_groups),
        "parameter_names": [model_parameters[identifier] for identifier in group_ids],
        "parameter_count": sum(parameter.numel() for parameter in group["params"]),
    }


def run_fit(
    seed: int,
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    state_dict: dict[str, Tensor],
    device: torch.device,
    output: Path,
) -> dict:
    """Train one exact 474-update arm with unchanged batch/order/state semantics."""
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(arm, state_dict).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model)
    check_common_parameter_bytes(model, state_dict)
    optimizer_receipt = optimizer_description(model, optimizer)
    families = parameter_families(model)
    telemetry: list[dict] = []
    updates = 0
    started = time.monotonic()
    for batch_start in range(0, 24, 4):
        ids = order[batch_start : batch_start + 4].tolist()
        recurrent_state = None
        for row_start in range(0, 20_000, CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mask = torch.from_numpy(need[ids, row_start : row_start + CHUNK_ROWS]).to(device=device)
            if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
                raise ValueError(f"nonfinite raw training batch: {seed}/{arm}/{row_start}")
            optimizer.zero_grad(set_to_none=True)
            prediction, recurrent_state, gate, context = model.forward_details(xx, recurrent_state)
            if not torch.isfinite(prediction).all() or not torch.isfinite(recurrent_state).all():
                raise ValueError(f"nonfinite prediction/state: {seed}/{arm}/{row_start}")
            if gate is not None and not torch.isfinite(gate).all():
                raise ValueError(f"nonfinite gate: {seed}/{arm}/{row_start}")
            if context is not None and not torch.isfinite(context).all():
                raise ValueError(f"nonfinite context: {seed}/{arm}/{row_start}")
            loss = e470.loss_for_batch(prediction, yy, mask)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss: {seed}/{arm}/{row_start}")
            loss.backward()
            stats = e474.apply_optimizer_update(
                model, optimizer, families, updates, f"{seed}/{arm}"
            )
            if updates in TELEMETRY_STEPS:
                if stats is None:
                    raise ValueError("required E474 telemetry sample missing")
                stats.update(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                        "chunk_start": row_start,
                        "loss": float(loss.detach().cpu()),
                        "state_abs_max": float(recurrent_state.detach().abs().max().cpu()),
                        "gate": gate_summary(gate),
                    }
                )
                telemetry.append(stats)
            recurrent_state = recurrent_state.detach()
            updates += 1
    if (
        updates != UPDATES
        or tuple(item["update_index_zero_based"] for item in telemetry) != TELEMETRY_STEPS
    ):
        raise ValueError(f"wrong update/telemetry cadence: {seed}/{arm}/{updates}")
    checkpoint = output / f"seed{seed}_{arm}.pt"
    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
    torch.save(model.cpu().state_dict(), temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint)
    return {
        "seed": seed,
        "arm": arm,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[int(index)]["row_group"] for index in order],
        "optimizer": optimizer_receipt,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
    }


def atomic_json(path: Path, value: dict) -> None:
    """Write a JSON receipt atomically so interrupted fits retain prior arms."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fit_only_manifest(manifest: dict) -> dict:
    """Retain only the fixed fit-role component descriptors in a receipt."""
    return {
        **manifest,
        "cohort": [item for item in manifest["cohort"] if item["role"] == "fit"],
    }


def common_manifest(
    sources: dict[str, str], manifest: dict, states: dict[int, dict[str, Tensor]]
) -> dict:
    """Build shared E484 recipe and provenance metadata for every execution phase."""
    return {
        "design_sha256": DESIGN_V2_SHA256,
        "blind_critic_sha256": CRITIC_SHA256,
        "critic_response_sha256": CRITIC_RESPONSE_SHA256,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "registered_contrasts_per_seed": list(CONTRASTS_PER_SEED),
        "registered_contrast_count": len(CONTRASTS_PER_SEED) * len(SEEDS),
        "fit_count": 12,
        "parameters_per_arm": {
            "gelu": BASE_PARAMETERS,
            "fixed_mix": BASE_PARAMETERS,
            "static_mix": GATED_PARAMETERS,
            "recurrent_mix": GATED_PARAMETERS,
            "projection_mix": GATED_PARAMETERS,
            "self_mix": GATED_PARAMETERS,
        },
        "updates_per_fit": UPDATES,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "schedule": "constant",
            "clip_norm": CLIP_NORM,
        },
        "gate": {
            "mixture_formula": "g*GELU(h)+(1-g)*LeakyReLU_0.01(h)",
            "sources": {
                **CONTEXT_MODES,
                "static_mix": "learned constant per-channel",
                "fixed_mix": "0.5",
            },
            "context_rms_epsilon": GATE_EPSILON,
            "context_centered": False,
            "initial_coefficients": 0.0,
            "extra_parameters_per_learned_gate": HIDDEN,
        },
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": file_sha256(e470.COHORT_PATH),
        "parent_source_sha256": sources,
        "actor_source_sha256": {
            path.name: file_sha256(path)
            for path in (HERE / "fit_context_mixture.py", HERE / "tests.py", HERE / "gpu_probe.py")
        },
        "initial_state_sha256": {
            str(seed): tensor_state_sha256(state) for seed, state in states.items()
        },
        "fit_only_cohort": fit_only_manifest(manifest),
        "fit_only": True,
        "test_rows_loaded": False,
    }


def main() -> None:
    """Run only the requested preflight or separately admitted full matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight-real", action="store_true")
    operation.add_argument("--run-fits", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.run_fits and args.device != "cuda":
        raise ValueError("full E484 fits require separate CUDA admission")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("real-data preflight must use CPU")
    torch.set_num_threads(1)
    sources = check_sources()
    states = make_initial_states()
    manifest, x, y, need, nodes = load_fit_cohort()
    result = {
        "kind": "E484 causal context-conditioned activation mixture",
        "status": "cpu_fit_only_preflight_running" if args.preflight_real else "fitting",
        **common_manifest(sources, manifest, states),
        "fit_only_cohort": fit_only_manifest(manifest),
        "fit_manifest_arms": [],
    }
    if args.preflight_real:
        result["fit_manifest_arms"] = run_cpu_preflight(x, y, need, nodes, states)
        result["status"] = "cpu_fit_only_one_update_per_arm_passed_test_unread"
        args.output.mkdir(parents=True)
        atomic_json(args.output / "fit_manifest.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "arms": len(result["fit_manifest_arms"]),
                    "test_rows_loaded": False,
                    "output": str(args.output),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.device == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    completed: list[dict] = []
    args.output.mkdir(parents=True)
    try:
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            result["device"] = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
            for seed in SEEDS:
                for arm in ARMS:
                    receipt = run_fit(
                        seed,
                        arm,
                        x,
                        y,
                        need,
                        nodes,
                        states[seed],
                        torch.device(args.device),
                        args.output,
                    )
                    completed.append(receipt)
                    arm_path = args.output / f"arm_seed{seed}_{arm}.json"
                    atomic_json(arm_path, receipt)
                    with (args.output / "fit_arms.jsonl").open("a") as stream:
                        stream.write(json.dumps(receipt, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
        if len(completed) != 12:
            raise ValueError("full E484 manifest requires exactly twelve completed arms")
    except Exception as exc:
        atomic_json(
            args.output / "FIT_FAILURE.json",
            {
                "status": "partial_failure",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
                "completed_arms": completed,
                "fit_count_complete": False,
                "test_rows_loaded": False,
            },
        )
        raise
    result["fit_manifest_arms"] = completed
    result["status"] = "all_12_fit_arms_complete_test_unread"
    atomic_json(args.output / "fit_manifest.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": len(completed),
                "test_rows_loaded": False,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
