"""Source-bound actual trainer chunk capture; trusted local torch artifacts only.

Dictionary payloads contain heterogeneous PyTorch state_dict values. Any is used
at this serialization boundary, not as an alternative execution implementation.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import random
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE_FILES = (
    "scripts/train_rnn.py",
    "src/wnn/nn.py",
    "src/wnn/optim.py",
    "src/wnn/spread_aux.py",
    "src/wnn/run_identity.py",
    "src/wnn/outcomes.py",
    "src/wnn/resources.py",
    "src/wnn/features.py",
    "src/wnn/data.py",
)
BATCH_KEYS = (
    "inputs",
    "y",
    "main_valid",
    "teach",
    "nxt_label",
    "sp_lab",
    "br_lab",
    "cl_lab",
    "sw",
    "nw",
    "nx_rows",
    "nx_up",
    "spread_encoded",
    "kth",
    "aux",
)
REQUIRED = {
    "version",
    "phase",
    "source",
    "execution",
    "args",
    "model_config",
    "student",
    "ema",
    "h",
    "h_ema",
    "optimizer",
    "optimizer_names",
    "scheduler",
    "scaler",
    "clock",
    "rng",
    "batch",
    "metadata",
    "force_skip",
    "ema_n",
    "ema_decay",
}


def sha(path: Path) -> str:
    """Stream a SHA256 identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def identities() -> dict[str, str]:
    current = {name: sha(ROOT / name) for name in SOURCE_FILES}
    binding = HERE / "source_binding.json"
    if binding.exists() and current != json.loads(binding.read_text()):
        raise ValueError("frozen source binding drift; independent review/rebuild required")
    return current


def cloned(value: Any, device: str = "cpu") -> Any:
    """Detach state recursively without sharing storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: cloned(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cloned(item, device) for item in value)
    return copy.deepcopy(value)


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def trainer_module() -> Any:
    spec = importlib.util.spec_from_file_location("capture_actual_trainer", ROOT / SOURCE_FILES[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("trainer import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend_state(device: str) -> dict[str, Any]:
    return {
        "device": device,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "autocast_cache": torch.is_autocast_cache_enabled(),
    }


def get_rng(device: str) -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - capture actual global trainer RNG
        "cuda_all": [v.cpu().clone() for v in torch.cuda.get_rng_state_all()]
        if device == "cuda"
        else [],
    }


def set_rng(value: dict[str, Any], device: str) -> None:
    torch.set_rng_state(value["torch_cpu"])
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])  # noqa: NPY002 - restore actual global trainer RNG
    if device == "cuda":
        torch.cuda.set_rng_state_all(value["cuda_all"])


def validate(payload: dict[str, Any]) -> None:
    """Fail closed before reconstructing a live recipe state."""
    if (
        set(payload) != REQUIRED
        or payload["version"] != 1
        or payload["phase"] != "pre_forward_pre_zero_grad"
    ):
        raise ValueError("capture schema/phase mismatch")
    if payload["source"] != identities():
        raise ValueError("source identity drift")
    device = payload["execution"]["device"]
    if device not in {"cpu", "cuda"} or payload["execution"] != backend_state(device):
        raise ValueError("execution backend differs; no cross-backend parity")
    args = payload["args"]
    expected = {
        "optim": "muon",
        "muon_scope": "rnn",
        "sched": "onecycle",
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "dropout": 0.0,
        "loss": "wmse",
        "distill_alpha": 0.5,
        "distill_boot_h": 8,
        "ema_teacher": 1.0,
        "ema_decay": 0.999,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "sync_free": True,
        "ema_metric_weight": False,
        "distill_alpha_far": None,
        "swa_frac": 0.0,
        "td_alpha": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux": "",
        "aux_next_k": "",
        "spread_aux_mode": "original",
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported recipe option")
    if device == "cuda" and (args["precision"] != "fp16" or not args["fast"]):
        raise ValueError("CUDA adapter requires actual fast fp16 recipe")
    if device == "cpu" and args["precision"] != "cpu-protocol":
        raise ValueError("CPU is only explicitly labelled fp32 protocol evidence")
    cfg = payload["model_config"]
    expected_cfg = {
        "n_in": 3 if device == "cpu" else 211,
        "hidden": 4 if device == "cpu" else 256,
        "layers": 2,
        "n_out": 23,
    }
    if cfg != expected_cfg:
        raise ValueError("unsupported exact adapter architecture")
    if cfg["n_out"] != 23 or cfg["layers"] != 2 or cfg["hidden"] > 256 or cfg["n_in"] > 211:
        raise ValueError("unsupported model architecture")
    if payload["h"] is None or payload["h_ema"] is None or payload["ema_n"] < 1:
        raise ValueError("genuine warm EMA and both incoming hidden states required")
    batch = payload["batch"]
    if set(batch) != set(BATCH_KEYS):
        raise ValueError("missing teacher/boot/aux/batch state")
    b, t, width = batch["inputs"].shape
    if b > 32 or t > 20000 or width != cfg["n_in"] or payload["metadata"]["s"] <= 0:
        raise ValueError("batch bound/history mismatch")
    if payload["h"].shape != (2, b, cfg["hidden"]) or payload["h_ema"].shape != payload["h"].shape:
        raise ValueError("hidden shape mismatch")
    for key in ("y", "teach"):
        if batch[key].shape != (b, t, 2):
            raise ValueError("teacher/label shape mismatch")
    for key in (
        "main_valid",
        "nxt_label",
        "sp_lab",
        "br_lab",
        "cl_lab",
        "sw",
        "nw",
        "nx_rows",
        "nx_up",
    ):
        if batch[key].shape != (b, t):
            raise ValueError("weight/aux shape mismatch")
    if batch["spread_encoded"] is not None or batch["kth"] or batch["aux"]:
        raise ValueError("unsupported optional auxiliary state")
    if set(payload["rng"]) != {"torch_cpu", "python", "numpy", "cuda_all"}:
        raise ValueError("incomplete RNG state")
    if device == "cuda" and len(payload["rng"]["cuda_all"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device topology mismatch")
    if not {"last_epoch", "total_steps", "_step_count"} <= set(payload["scheduler"]):
        raise ValueError("scheduler clocks missing")
    if device == "cuda" and not {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    } <= set(payload["scaler"]):
        raise ValueError("CUDA scaler state missing")
    for state in (payload["student"], payload["ema"]):
        if any(
            value.dtype != torch.float32
            or value.requires_grad
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError("model state must be finite detached CPU fp32")
    for key in ("inputs", "y", "teach", "main_valid", "sw", "nw", "nx_rows", "nx_up"):
        value = batch[key]
        if (
            value.dtype != torch.float32
            or value.requires_grad
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("batch values must be finite detached CPU fp32")
    for key in ("nxt_label", "sp_lab", "br_lab", "cl_lab"):
        if batch[key].dtype != torch.int64:
            raise ValueError("auxiliary indices must be int64")
    if (batch["main_valid"] < 0).any() or (batch["nw"] < 0).any():
        raise ValueError("negative scientific weights unsupported")
    if tensor_bytes(payload) > 900_000_000:
        raise ValueError("capture exceeds bounded storage budget")
    if payload["metadata"].get("evidence_kind") not in {
        "synthetic_cpu_actual_code",
        "synthetic_cuda_actual_architecture",
        "real_fit_capture",
    }:
        raise ValueError("unclassified evidence")
    if payload["metadata"]["evidence_kind"] == "real_fit_capture":
        from fit_panel import PANEL

        if not PANEL.exists() or payload["metadata"].get("panel_sha256") != sha(PANEL):
            raise ValueError("real fit panel not independently bound")
        panel = json.loads(PANEL.read_text())
        if (
            payload["metadata"].get("row_ids") != panel["rows"]
            or cfg != {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}
            or device != "cuda"
            or t != 20000
        ):
            raise ValueError("real panel/architecture mismatch")
    names = list(payload["student"])
    groups = payload["optimizer"]["param_groups"]
    order = payload["optimizer_names"]
    if len(groups) != len(order) or any(
        len(group["params"]) != len(group_names)
        for group, group_names in zip(groups, order, strict=True)
    ):
        raise ValueError("optimizer group mapping mismatch")
    flat = [name for group in order for name in group]
    if len(flat) != len(set(flat)) or set(flat) != set(names) - {"mean", "std"}:
        raise ValueError("optimizer parameter membership mismatch")
    indexed = {
        index: name
        for group, group_names in zip(groups, order, strict=True)
        for index, name in zip(group["params"], group_names, strict=True)
    }
    if set(payload["optimizer"]["state"]) != set(indexed):
        raise ValueError("warm optimizer state missing")
    for group in groups:
        for index in group["params"]:
            entry = payload["optimizer"]["state"][index]
            required = (
                {"momentum_buffer"}
                if group.get("muon", False)
                else {"step", "exp_avg", "exp_avg_sq"}
            )
            if set(entry) != required:
                raise ValueError("optimizer moment fields missing")
            for key in required - {"step"}:
                if entry[key].shape != payload["student"][indexed[index]].shape:
                    raise ValueError("optimizer moment shape mismatch")


class StopAtLoss(Exception):
    """Separate gradient decomposition ends before online backward/update."""


class Capture:
    """Hooks preserve original arithmetic; enabled capture synchronizes for I/O."""

    def __init__(
        self,
        enabled: bool = False,
        directory: Path | None = None,
        force_skip: bool = False,
        loss_only: bool = False,
    ) -> None:
        self.enabled, self.directory, self.force_skip, self.loss_only = (
            enabled,
            directory,
            force_skip,
            loss_only,
        )
        self.pre: dict[str, Any] = {}
        self.post: dict[str, Any] = {}
        self.terms: dict[str, torch.Tensor] = {}
        self.preclip: dict[str, Any] = {}
        self.clipped: dict[str, Any] = {}

    def component(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.terms[name] = value
        return value

    def before(self, env: dict[str, Any]) -> None:
        model, ema, opt = env["model"], env["ema_model"], env["opt"]
        device = env["device"].type
        if not model.training or ema.module.training:
            raise ValueError("student train/EMA eval modes required")
        if env["scaler"].is_enabled():
            stages = env["scaler"]._per_optimizer_states.values()
            if any(str(state["stage"]) != "OptState.READY" for state in stages):
                raise ValueError("scaler must be between update attempts")
        by_id = {id(p): name for name, p in model.named_parameters()}
        self.terms = {}
        self.pre = {
            "version": 1,
            "phase": "pre_forward_pre_zero_grad",
            "source": identities(),
            "execution": backend_state(device),
            "args": vars(env["args"]).copy(),
            "model_config": env["capture_model_config"],
            "student": cloned(model.state_dict()),
            "ema": cloned(ema.module.state_dict()),
            "h": cloned(env["h"]),
            "h_ema": cloned(env["h_ema"]),
            "ema_n": ema.n,
            "ema_decay": ema.decay,
            "optimizer": cloned(opt.state_dict()),
            "optimizer_names": [
                [by_id[id(p)] for p in group["params"]] for group in opt.param_groups
            ],
            "scheduler": cloned(env["sched"].state_dict()),
            "scaler": cloned(env["scaler"].state_dict()),
            "clock": {
                "step_count": env["step_count"],
                "attempts": env["epoch_counter"].attempts,
                "skips": cloned(env["epoch_counter"].skips),
                "ema_updates": env["epoch_counter"].ema_updates,
                "completed_batches": env["epoch_counter"].completed_batches,
            },
            "rng": get_rng(device),
            "batch": {key: cloned(env[key]) for key in BATCH_KEYS},
            "metadata": {
                **env["capture_metadata"],
                "s": env["s"],
                "struct_at": env["struct_at"],
                "next_at": env["next_at"],
            },
            "force_skip": self.force_skip,
        }
        # Keep partial pre-state even if the online backward fails.
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
            torch.save(self.pre, self.directory / "pre.pt")

    def loss(self, env: dict[str, Any]) -> None:
        if self.loss_only:
            raise StopAtLoss()

    def inject(self, env: dict[str, Any]) -> None:
        if self.force_skip:
            if not env["scaler"].is_enabled():
                raise ValueError("deliberate skip requires enabled scaler")
            next(p for p in env["model"].parameters() if p.grad is not None).grad.flatten()[0] = (
                float("inf")
            )

    def gradients(self, env: dict[str, Any], clipped: bool) -> None:
        values = {name: cloned(p.grad) for name, p in env["model"].named_parameters()}
        if clipped:
            self.clipped = values
        else:
            self.preclip = values

    def after(self, env: dict[str, Any]) -> None:
        self.post = {
            "student": cloned(env["model"].state_dict()),
            "ema": cloned(env["ema_model"].module.state_dict()),
            "ema_n": env["ema_model"].n,
            "optimizer": cloned(env["opt"].state_dict()),
            "scheduler": cloned(env["sched"].state_dict()),
            "scaler": cloned(env["scaler"].state_dict()),
            "h": cloned(env["h"]),
            "h_ema": cloned(env["h_ema"]),
            "rng": get_rng(env["device"].type),
            "clock": {
                "step_count": env["step_count"],
                "attempts": env["epoch_counter"].attempts,
                "skips": cloned(env["epoch_counter"].skips),
                "ema_updates": env["epoch_counter"].ema_updates,
                "completed_batches": env["epoch_counter"].completed_batches,
            },
            "losses": {name: cloned(value) for name, value in self.terms.items()},
            "total_loss": cloned(env["loss"]),
            "preclip": self.preclip,
            "clipped": self.clipped,
        }
        self.post["parameter_delta"] = {
            name: self.post["student"][name] - self.pre["student"][name]
            for name, _ in env["model"].named_parameters()
        }
        if self.directory is not None:
            torch.save(self.post, self.directory / "post.pt")
        self.terms = {}  # release forward graph, retaining detached observations only


def hook(statement: str) -> list[ast.stmt]:
    return ast.parse("if _capture.enabled:\n    " + statement).body


def instrument_body(body: list[ast.stmt], cpu: bool = False) -> list[ast.stmt]:
    """Add ordered guarded hooks to the original chunk, without rewriting math."""
    result: list[ast.stmt] = []
    for original in body:
        node = copy.deepcopy(original)
        text = ast.unparse(node)
        if text.startswith("sl ="):
            result.append(node)
            result.extend(hook("_capture.before(locals())"))
            continue
        if isinstance(node, ast.If) and text.startswith("if moments is not None:"):
            result.append(node)
            result.extend(hook("_capture.component('main', loss)"))
            continue
        if isinstance(node, ast.If):
            arm = {
                "args.aux_struct > 0": "struct",
                "args.aux_next > 0": "next",
                "args.ema_teacher > 0": "ema",
            }.get(ast.unparse(node.test))
            if arm:
                for assignment in node.body:
                    if (
                        isinstance(assignment, ast.Assign)
                        and ast.unparse(assignment.targets[0]) == "loss"
                        and isinstance(assignment.value, ast.BinOp)
                    ):
                        rhs = assignment.value.right
                        assignment.value.right = ast.IfExp(
                            test=ast.Attribute(
                                value=ast.Name(id="_capture", ctx=ast.Load()),
                                attr="enabled",
                                ctx=ast.Load(),
                            ),
                            body=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="_capture", ctx=ast.Load()),
                                    attr="component",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Constant(arm), rhs],
                                keywords=[],
                            ),
                            orelse=copy.deepcopy(rhs),
                        )
        if text == "timer.mark('loss')":
            result.extend(hook("_capture.loss(locals())"))
        if text == "scaler.unscale_(opt)":
            result.extend(hook("_capture.inject(locals())"))
        result.append(node)
        if text == "epoch_counter.observe_unscaled(scaler, opt)":
            result.extend(hook("_capture.gradients(locals(), False)"))
        if text.startswith("torch.nn.utils.clip_grad_norm_"):
            result.extend(hook("_capture.gradients(locals(), True)"))
        if isinstance(node, ast.If) and text.startswith("if ema_model is not None:"):
            result.extend(hook("_capture.after(locals())"))
        if cpu:
            # Explicit protocol variant, not a claim that CUDA autocast is CPU parity.
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and ast.unparse(child.func) == "torch.autocast":
                    child.args[0] = ast.Constant("cpu")
                    child.keywords.append(ast.keyword(arg="enabled", value=ast.Constant(False)))
    return result


def chunk_nodes() -> list[ast.stmt]:
    tree = ast.parse((ROOT / SOURCE_FILES[0]).read_text())
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "s"
        and ast.unparse(node.iter) == "range(0, inputs.shape[1], args.chunk)"
    ]
    if len(loops) != 1:
        raise ValueError("actual update chunk structure changed")
    body = loops[0].body
    end = next(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "duty is not None"
    )
    return body[:end]


def compiled_chunk(cpu: bool = False, instrumented: bool = True) -> Any:
    body = instrument_body(chunk_nodes(), cpu) if instrumented else copy.deepcopy(chunk_nodes())
    if cpu and not instrumented:
        for node in body:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and ast.unparse(child.func) == "torch.autocast":
                    child.args[0] = ast.Constant("cpu")
                    child.keywords.append(ast.keyword(arg="enabled", value=ast.Constant(False)))
    return compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        str(ROOT / SOURCE_FILES[0]),
        "exec",
    )


class NullTimer:
    def mark(self, name: str) -> None:
        pass


def make_environment(
    module: Any,
    args: dict[str, Any],
    cfg: dict[str, int],
    device: str,
    total: int,
    scaler_enabled: bool,
) -> dict[str, Any]:
    model = module.RNNModel(**cfg).to(device)
    opt = module.MuonAdamW(
        module.muon_param_groups(model, "rnn"),
        lr=args["lr"],
        betas=(0.9, args["beta2"]),
        weight_decay=args["wd"],
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args["lr"], total_steps=total, pct_start=0.05, final_div_factor=100
    )
    scaler = torch.amp.GradScaler(device, enabled=scaler_enabled)
    ema = module.FastEMA(model, 0.999)
    return {
        **vars(module),
        "args": SimpleNamespace(**args),
        "capture_model_config": cfg,
        "model": model,
        "ema_model": ema,
        "opt": opt,
        "sched": sched,
        "scaler": scaler,
        "device": torch.device(device),
        "amp_dtype": torch.float16,
        "epoch_counter": module.EpochUpdateCounter(torch.device(device)),
        "step_count": 0,
        "grad_stats": None,
        "swa_model": None,
        "swa_start": 0,
        "moments": None,
        "timer": NullTimer(),
        "_capture": Capture(),
        "n_out": 23,
        "struct_at": 2,
        "next_at": 21,
        "aux": [],
        "kth": [],
        "spread_encoded": None,
        "h": None,
        "h_ema": None,
    }


def restore(payload: dict[str, Any], module: Any) -> dict[str, Any]:
    validate(payload)
    device = payload["execution"]["device"]
    env = make_environment(
        module,
        payload["args"],
        payload["model_config"],
        device,
        payload["scheduler"]["total_steps"],
        bool(payload["scaler"]),
    )
    env["model"].load_state_dict(cloned(payload["student"], device))
    env["ema_model"].module.load_state_dict(cloned(payload["ema"], device))
    env["ema_model"].n = payload["ema_n"]
    mapping = {id(p): name for name, p in env["model"].named_parameters()}
    order = [[mapping[id(p)] for p in group["params"]] for group in env["opt"].param_groups]
    if order != payload["optimizer_names"]:
        raise ValueError("optimizer name order differs")
    # load_state_dict preserves Muon's integer Adam clock; tensor state goes to param device.
    env["opt"].load_state_dict(cloned(payload["optimizer"]))
    env["sched"].load_state_dict(payload["scheduler"])
    env["scaler"].load_state_dict(payload["scaler"])
    env.update(cloned(payload["batch"], device))
    env.update(
        {
            "h": cloned(payload["h"], device),
            "h_ema": cloned(payload["h_ema"], device),
            "s": payload["metadata"]["s"],
            "capture_metadata": payload["metadata"],
        }
    )
    clock = payload["clock"]
    env["step_count"] = clock["step_count"]
    for key in ("attempts", "ema_updates", "completed_batches"):
        setattr(env["epoch_counter"], key, clock[key])
    env["epoch_counter"].skips.copy_(clock["skips"].to(device))
    env["model"].train()
    set_rng(payload["rng"], device)
    return env


def replay(payload: dict[str, Any], module: Any, components: bool = False) -> dict[str, Any]:
    env = restore(payload, module)
    observer = Capture(True, force_skip=payload["force_skip"], loss_only=components)
    env["_capture"] = observer
    try:
        exec(compiled_chunk(payload["execution"]["device"] == "cpu"), env)
    except StopAtLoss:
        gradients = {}
        params = list(env["model"].parameters())
        names = [name for name, _ in env["model"].named_parameters()]
        for name, term in observer.terms.items():
            scale = env["scaler"].get_scale()
            values = torch.autograd.grad(
                env["scaler"].scale(term), params, retain_graph=True, allow_unused=True
            )
            gradients[name] = {
                key: cloned(value / scale) if value is not None else None
                for key, value in zip(names, values, strict=True)
            }
        return {
            "losses": {key: cloned(value) for key, value in observer.terms.items()},
            "component_gradients": gradients,
        }
    return observer.post


def compare(
    left: Any, right: Any, atol: float = 1e-7, rtol: float = 1e-6, path: str = "root"
) -> dict[str, float]:
    """Assert complete state equality and return every floating tensor max delta."""
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            raise AssertionError(path + ": tensor metadata differs")
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol, equal_nan=True, msg=path)
        if left.is_floating_point():
            finite = torch.isfinite(left) & torch.isfinite(right)
            return {
                path: float((left[finite] - right[finite]).abs().max()) if finite.any() else 0.0
            }
        return {}
    if isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(path)
        return {}
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(path + ": keys differ")
        out = {}
        for key in left:
            out.update(compare(left[key], right[key], atol, rtol, path + "." + str(key)))
        return out
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(path)
        out = {}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.update(compare(a, b, atol, rtol, path + "." + str(index)))
        return out
    if left != right:
        raise AssertionError(path + ": value differs")
    return {}


def generate_copy() -> None:
    """Produce a reviewable trainer copy; opt-in directory/step env variables."""
    import difflib

    original = (ROOT / SOURCE_FILES[0]).read_text()
    # This copy is deliberately not a real-fit launcher: capture metadata must be
    # injected through the explicit adapter; otherwise capture remains disabled.
    preamble = "\nfrom live_capture import Capture\n_capture = Capture(False)\n"
    nodes = chunk_nodes()
    lines = original.splitlines(True)
    chunk_text = (
        ast.unparse(
            ast.fix_missing_locations(ast.Module(body=instrument_body(nodes), type_ignores=[]))
        )
        + "\n"
    )
    generated = (
        "".join(lines[: nodes[0].lineno - 1])
        + textwrap.indent(chunk_text, " " * nodes[0].col_offset)
        + "".join(lines[nodes[-1].end_lineno :])
    )
    generated = generated.replace(
        "ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})"
    )
    position = generated.index("\nROOT =")
    generated = generated[:position] + preamble + generated[position:]
    (HERE / "instrumented_train_rnn.py").write_text(generated)
    (HERE / "trainer.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                generated.splitlines(True),
                fromfile="scripts/train_rnn.py",
                tofile="isolated/instrumented_train_rnn.py",
            )
        )
    )
    (HERE / "chunk_source.py.txt").write_text(
        ast.unparse(
            ast.fix_missing_locations(
                ast.Module(body=instrument_body(chunk_nodes()), type_ignores=[])
            )
        )
        + "\n"
    )
    binding = identities()
    if not (HERE / "source_binding.json").exists():
        (HERE / "source_binding.json").write_text(json.dumps(binding, indent=2) + "\n")


if __name__ == "__main__":
    generate_copy()
