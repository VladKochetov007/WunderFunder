from __future__ import annotations

import json
import math
import platform
import resource
import time
from pathlib import Path

import torch
from attrs import define, field

from wnn.research_reporting import RunMetadata, RunRecorder
from wnn.splits import file_sha256
from wnn.training.checkpoints import serialized_args
from wnn.training.state import TrainingRun


@define
class SmoothingObserver:
    """Record the single-objective FP32 experiment; fail before nonfinite updates.

    These explicit scalar observations synchronize CUDA. Both arms pay the same
    logging cost; measured throughput includes it. This is not enabled by the
    general trainer CLI or by historical recipes.
    """

    recorder: RunRecorder | None = None
    started: float = field(factory=time.perf_counter)
    gradient_norm: float = 0.0
    loss: float = 0.0
    rates: list[float] = field(factory=list)
    attempts: int = 0
    completed: int = 0
    clipped: int = 0
    nonfinite: int = 0
    rows: int = 0

    def __call__(self, run: TrainingRun, phase: str) -> None:
        """Observe explicit lifecycle boundaries without altering the optimizer."""
        if phase == "start":
            self._start(run)
        elif phase == "before_update":
            self._before(run)
        elif phase == "after_update":
            self._after(run)
        else:
            raise ValueError(f"Unsupported observer phase: {phase}")

    def _start(self, run: TrainingRun) -> None:
        args = run.options.args
        packet = Path(__file__).resolve().parent
        recipe = json.loads((packet / "recipe.json").read_text())
        for name in (
            "epochs",
            "batch",
            "batch_order",
            "chunk",
            "update_rows",
            "detach_rows",
            "hidden",
            "layers",
            "dropout",
            "precision",
            "optim",
            "lr",
            "wd",
            "beta2",
            "clip",
            "sched",
            "warmup_frac",
            "wsd_decay",
            "wsd_final",
            "wsd_shape",
            "loss",
            "checkpoint_policy",
            "checkpoint_epoch",
            "no_eval_during_fit",
            "save_training_state",
        ):
            if getattr(args, name) != recipe["training"][name]:
                raise ValueError(f"R1 training setting differs from recipe: {name}")
        if args.max_batches and not args.profile_steps:
            raise ValueError("Partial scientific fits are not R1; use an engineering profile")
        if (
            run.data.campaign_split is None
            or args.train_source != "train+valid"
            or args.fit_subset_manifest
            or args.arch != "gru"
        ):
            raise ValueError("R1 needs a full source-qualified split and the declared GRU")
        if args.precision != "fp32" or args.device != "cuda":
            raise ValueError("Smoothing receipts require CUDA FP32")
        if run.optimization.use_autocast or run.optimization.scaler.is_enabled():
            raise ValueError("Autocast/scaler must be disabled")
        if args.loss != "wmse" or run.network.n_out != 2 or args.resume_state:
            raise ValueError("R1 records fresh, two-output WMSE fits; use a new attempt to retry")
        if any(
            (
                args.init,
                args.distill,
                args.distill_cache_root,
                args.propensity,
                args.prop_input,
                args.ema_teacher,
                args.td_alpha,
                args.eval_ema,
                args.swa_frac,
            )
        ):
            raise ValueError("R1 requires scratch labels-only fits without fitted ancestors")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("TF32 must remain disabled")
        tensors = [*run.network.model.parameters(), *run.network.model.buffers()]
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
            raise ValueError("Floating model tensors must be FP32")
        metadata = RunMetadata(
            source={
                "training": run.data.provenance["trainer_implementation"],
                "packet": {
                    name: file_sha256(packet / name)
                    for name in ("train.py", "training_report.py", "recipe.json")
                },
            },
            config=serialized_args(args),
            provenance={
                "trainer": run.data.provenance,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "python": platform.python_version(),
                "precision": "fp32",
                "tf32": False,
                "loss_components": ["main_wmse"],
                "normalization": "fit-loader statistics; no evaluation data",
                "timing_scope": "training loop, including scalar reporting synchronization",
            },
        )
        self.recorder = RunRecorder(run.options.out_dir, metadata)
        self.started = time.perf_counter()

    def _before(self, run: TrainingRun) -> None:
        self.attempts += 1
        if run.batch.inputs.dtype != torch.float32 or run.batch.out.dtype != torch.float32:
            raise ValueError("Inputs and outputs must remain FP32")
        if not torch.is_tensor(run.batch.h) or run.batch.h.dtype != torch.float32:
            raise ValueError("The GRU state must remain FP32")
        gradients = [p.grad for p in run.network.model.parameters() if p.grad is not None]
        if not gradients or any(g.dtype != torch.float32 for g in gradients):
            raise ValueError("Expected nonempty FP32 gradients")
        self.gradient_norm = float(
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach()) for g in gradients])
            ).item()
        )
        self.loss = float(run.batch.loss.detach().item())
        self.rates = [float(group["lr"]) for group in run.optimization.opt.param_groups]
        if not math.isfinite(self.gradient_norm) or not math.isfinite(self.loss):
            self.nonfinite += 1
            self._event(run, "nonfinite_before_update")
            raise FloatingPointError("Nonfinite loss/gradient; stopped before optimizer update")
        self.clipped += int(self.gradient_norm > run.options.args.clip)

    def _after(self, run: TrainingRun) -> None:
        self.completed += 1
        if self.completed != run.progress.step_count or self.attempts != self.completed:
            raise RuntimeError("Observer and trainer update clocks disagree")
        if self.completed == 1:
            for state in run.optimization.opt.state.values():
                for value in state.values():
                    if (
                        torch.is_tensor(value)
                        and value.is_floating_point()
                        and value.dtype != torch.float32
                    ):
                        raise ValueError("Optimizer state must be FP32")
        width = run.batch.sl.stop - run.batch.sl.start
        self.rows += int(run.batch.inputs.shape[0]) * width
        elapsed = time.perf_counter() - self.started
        if self.recorder is None:
            raise RuntimeError("Observer start event is missing")
        self.recorder.record_metrics(
            step=int(run.progress.step_count),
            epoch=int(run.progress.epoch),
            loss=self.loss,
            learning_rates=self.rates,
            gradient_norm=self.gradient_norm,
            throughput={"processed_rows_per_second": self.rows / max(elapsed, 1e-9)},
            timing={"training_loop_elapsed_seconds": elapsed},
        )
        self._event(run, "optimizer_completed")

    def _event(self, run: TrainingRun, status: str) -> None:
        record = {
            "status": status,
            "step": int(run.progress.step_count),
            "epoch": int(run.progress.epoch),
            "batch": int(run.batch.b),
            "optimizer_attempts": self.attempts,
            "optimizer_completed": self.completed,
            "clipped_updates": self.clipped,
            "nonfinite_updates": self.nonfinite,
            "processed_rows": self.rows,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if platform.system() == "Darwin" else 1024),
            "ram_scope": "training process only; excludes loader children",
        }
        with (Path(run.options.out_dir) / "update_events.jsonl").open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
