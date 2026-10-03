from __future__ import annotations

import gc
import json
import resource
import time
from pathlib import Path
from typing import Any  # Authenticated heterogeneous Torch capture metadata.

import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
ROUND4 = ROOT / "research/campaign_2026-10-03/continuation/round4"
OUTPUT = ROUND4 / "gpu_capture_v4_default_2120r"


def describe(value: torch.Tensor) -> dict[str, Any]:
    """Inspect Tensor metadata without reading its full storage."""
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": value.requires_grad,
    }


def clock(value: dict[str, Any]) -> dict[str, int]:
    """Require canonical clock scalar types; never coerce an alias into validity."""
    for name in ("step_count", "attempts", "ema_updates", "completed_batches"):
        assert type(value[name]) is int
    skipped = value["skips"]
    assert type(skipped) is torch.Tensor and skipped.dtype == torch.int64 and skipped.shape == ()
    return {
        name: int(item.item()) if isinstance(item, torch.Tensor) else item
        for name, item in value.items()
    }


def main() -> None:
    """Inspect authenticated captures without actor imports, replay or CUDA calls."""
    torch.set_num_threads(1)
    started = time.monotonic()
    byte_audit = json.loads((OWN / "output_byte_verification.json").read_text())
    assert byte_audit["all_match"]
    receipt = json.loads((ROUND4 / "controller_receipts/capture_v4_2120.json").read_text())
    panel = json.loads((ROUND4 / "live_capture_v4/fit_panel.json").read_text())
    captures = {}
    for branch in ("synthetic_actual_architecture", "real_fit_panel"):
        for control in ("normal", "forced_skip"):
            directory = OUTPUT / branch / control
            for name in ("pre.pt", "post.pt"):
                key = str((directory / name).relative_to(OUTPUT))
                record = byte_audit["files"][key]
                stat = (directory / name).stat()
                assert (stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"])
            pre = torch.load(
                directory / "pre.pt", mmap=True, map_location="cpu", weights_only=False
            )
            post = torch.load(
                directory / "post.pt", mmap=True, map_location="cpu", weights_only=False
            )
            assert pre["version"] == 4 and pre["phase"] == "pre_forward_pre_zero_grad"
            assert pre["source"] == receipt["source"]
            args, batch, cfg = pre["args"], pre["batch"], pre["model_config"]
            for name, expected in {
                "arch": "gru",
                "cell": "gru",
                "optim": "muon",
                "muon_scope": "rnn",
                "precision": "fp16",
                "fast": True,
                "sched": "onecycle",
                "chunk": 1000,
                "dropout": 0.0,
                "loss": "wmse",
                "distill_alpha": 0.5,
                "distill_boot_h": 8,
                "distill_alpha_far": None,
                "ema_teacher": 1.0,
                "ema_decay": 0.999,
                "aux_struct": 0.3,
                "aux_next": 0.3,
                "sync_free": True,
            }.items():
                assert args[name] == expected, name
            assert cfg == {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}
            assert pre["execution"]["device"] == "cuda"
            assert batch["inputs"].shape == (32, 20000 if branch == "real_fit_panel" else 6000, 211)
            assert batch["inputs"].dtype == torch.float32
            for name in ("h", "h_ema"):
                assert pre[name].shape == (2, 32, 256) and pre[name].dtype == torch.float16
                assert not pre[name].requires_grad and pre[name].device.type == "cpu"
                assert post[name].shape == pre[name].shape and post[name].dtype == pre[name].dtype
            assert batch["y"].shape[-1] == batch["teach"].shape[-1] == 2
            assert batch["y"].dtype == batch["teach"].dtype == torch.float32
            for name in ("nxt_label", "sp_lab", "br_lab", "cl_lab"):
                assert batch[name].dtype == torch.int64
            assert all(
                value.dtype == torch.float32 and not value.requires_grad
                for value in pre["student"].values()
            )
            assert all(
                value.dtype == torch.float32 and not value.requires_grad
                for value in pre["ema"].values()
            )
            incoming, outgoing = clock(pre["clock"]), clock(post["clock"])
            assert incoming == {
                "step_count": 4,
                "attempts": 4,
                "skips": 0,
                "ema_updates": 4,
                "completed_batches": 0,
            }
            expected_out = {
                **incoming,
                "step_count": 5,
                "attempts": 5,
                "ema_updates": 5,
                "skips": 1 if control == "forced_skip" else 0,
            }
            assert outgoing == expected_out
            assert type(pre["ema_n"]) is int and pre["ema_n"] == 4 and post["ema_n"] == 5
            assert post["scheduler"]["last_epoch"] == pre["scheduler"]["last_epoch"] + 1
            assert post["scheduler"]["_step_count"] == pre["scheduler"]["_step_count"] + 1
            assert pre["scaler"]["scale"] > 0
            if control == "forced_skip":
                assert pre["force_skip"] is True
                assert (
                    post["scaler"]["scale"]
                    == pre["scaler"]["scale"] * pre["scaler"]["backoff_factor"]
                )
                assert post["scaler"]["_growth_tracker"] == 0
                assert torch.isinf(post["clip_norm"]), "deliberate nonfinite norm absent"
            else:
                assert pre["force_skip"] is False
                assert post["scaler"]["scale"] == pre["scaler"]["scale"]
                assert torch.isfinite(post["clip_norm"])
            adam_clocks = {}
            for identifier, incoming_state in pre["optimizer"]["state"].items():
                outgoing_state = post["optimizer"]["state"][identifier]
                if "step" in incoming_state:
                    assert type(incoming_state["step"]) is int and incoming_state["step"] == 4
                    assert type(outgoing_state["step"]) is int
                    assert outgoing_state["step"] == (4 if control == "forced_skip" else 5)
                    adam_clocks[str(identifier)] = [incoming_state["step"], outgoing_state["step"]]
                for name, value in incoming_state.items():
                    if isinstance(value, torch.Tensor):
                        assert (
                            value.dtype == torch.float32
                            and value.device.type == "cpu"
                            and not value.requires_grad
                        )
            # Read only the stopped target's 32x1000x2 slice and matching inputs.
            s = pre["metadata"]["s"]
            assert s == 4000
            yc = batch["y"][:, s : s + 1000].clamp(-2, 2)
            here_teacher = batch["teach"][:, s : s + 1000].clamp(-2, 2)
            ahead = torch.arange(s, s + 1000)[None] + args["distill_boot_h"]
            alive = (batch["nxt_label"][:, s : s + 1000] > ahead) & (
                ahead < batch["teach"].shape[1]
            )
            # Advanced indexing touches only the small required horizon slice.
            ahead_teacher = batch["teach"][:, s + 8 : s + 1008].clamp(-2, 2)
            label = torch.where(alive[..., None], ahead_teacher, yc)
            target = (1 - args["distill_alpha"]) * label + args["distill_alpha"] * here_teacher
            assert torch.equal(target, post["target"])
            assert post["target"].shape == (32, 1000, 2) and post["target"].dtype == torch.float32
            assert not post["target"].requires_grad and torch.isfinite(post["target"]).all()
            assert post["out"].shape == post["out_ema"].shape == (32, 1000, 23)
            assert post["out"].dtype == post["out_ema"].dtype == torch.float16
            assert set(post["losses"]) == {"main", "struct", "next", "ema"}
            assert all(torch.isfinite(value) for value in post["losses"].values())
            if branch == "real_fit_panel":
                assert pre["metadata"]["row_ids"] == panel["rows"]
                assert pre["metadata"]["teacher_binding"] == panel["bindings"]
                assert (
                    pre["metadata"]["panel_sha256"]
                    == json.loads((OWN / "lineage_and_numeric_results.json").read_text())[
                        "panel_sha256"
                    ]
                )
            captures[branch + ":" + control] = {
                "incoming_clock": incoming,
                "outgoing_clock": outgoing,
                "ema_n": [pre["ema_n"], post["ema_n"]],
                "scheduler_last_epoch": [
                    pre["scheduler"]["last_epoch"],
                    post["scheduler"]["last_epoch"],
                ],
                "scale": [pre["scaler"]["scale"], post["scaler"]["scale"]],
                "Adam_native_steps": adam_clocks,
                "h": describe(pre["h"]),
                "h_ema": describe(pre["h_ema"]),
                "target": describe(post["target"]),
                "out": describe(post["out"]),
                "target_exact_supported_formula": True,
                "execution": pre["execution"],
                "departure": pre["metadata"].get("departure"),
                "nonfinite_clip_norm_expected_skip": control == "forced_skip",
            }
            del (
                pre,
                post,
                batch,
                target,
                yc,
                here_teacher,
                ahead_teacher,
                label,
                alive,
                incoming_state,
                outgoing_state,
            )
            gc.collect()
    (OWN / "capture_metadata_results.json").write_text(
        json.dumps(
            {
                "all_pass": True,
                "captures": captures,
                "mmap_CPU_only": True,
                "CUDA_calls": False,
                "replay_performed": False,
                "wall_seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "capture_metadata_pass": True,
                "captures": len(captures),
                "wall_seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
    )


if __name__ == "__main__":
    main()
