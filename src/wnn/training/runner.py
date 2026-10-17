from __future__ import annotations

from wnn.training.batches import prepare_batch, prepare_labels, prepare_windows
from wnn.training.cli import build_parser
from wnn.training.data_caches import bind_caches
from wnn.training.data_loaders import build_loaders
from wnn.training.data_membership import select_membership
from wnn.training.environment import TrainingEnvironment
from wnn.training.epoch import epoch_diagnostics, finish_batch, prepare_epoch, record_fit_window
from wnn.training.forward_backward import backward_standard, forward_standard
from wnn.training.losses import auxiliary_loss, main_loss
from wnn.training.macro_updates import backward_controlled, prepare_window
from wnn.training.optimization import build_optimization
from wnn.training.outputs import evaluate_epoch, finalize, save_epoch
from wnn.training.profile_results import report_profile, write_parity
from wnn.training.provenance import record_provenance
from wnn.training.setup_lineage import configure_runtime, load_lineage, validate_membership
from wnn.training.setup_model import build_model, load_initializer
from wnn.training.setup_model_runtime import configure_activation_backend, record_model_identity
from wnn.training.setup_preprocessing import configure_initializer_teacher, fit_preprocessing
from wnn.training.setup_progress import initialize_progress, restore_progress
from wnn.training.state import TrainingRun
from wnn.training.updates import optimizer_step
from wnn.training.validation_inputs import validate_distillation, validate_inputs
from wnn.training.validation_training import validate_clocks, validate_models


def run_training(
    argv: list[str] | None = None, *, environment: TrainingEnvironment | None = None
) -> None:
    """Execute the existing recipe through focused setup and training components."""
    environment = environment if environment is not None else TrainingEnvironment()
    if argv is not None:
        environment.argv = list(argv)
    run = TrainingRun(environment)
    run.options.ap = build_parser()
    run.options.args = run.options.ap.parse_args(argv)
    validate_inputs(run)
    validate_distillation(run)
    validate_models(run)
    load_lineage(run)
    validate_membership(run)
    configure_runtime(run)
    validate_clocks(run)
    select_membership(run)
    bind_caches(run)
    build_loaders(run)
    record_provenance(run)
    build_model(run)
    load_initializer(run)
    fit_preprocessing(run)
    configure_initializer_teacher(run)
    configure_activation_backend(run)
    record_model_identity(run)
    if not build_optimization(run):
        return
    initialize_progress(run)
    restore_progress(run)
    args = run.options.args
    for epoch in range(run.progress.start_epoch, args.epochs):
        if args.stop_after_epoch is not None and epoch > args.stop_after_epoch:
            break
        run.progress.epoch = epoch
        prepare_epoch(run)
        for b, batch in enumerate(run.progress.profile_batches):
            if args.max_batches and b >= args.max_batches:
                break
            run.batch.b, run.batch.batch = (b, batch)
            prepare_batch(run)
            prepare_labels(run)
            prepare_windows(run)
            for window_index, sl in enumerate(run.batch.windows):
                run.batch.window_index, run.batch.sl = (window_index, sl)
                prepare_window(run)
                if run.options.controlled_loss:
                    backward_controlled(run)
                else:
                    forward_standard(run)
                    main_loss(run)
                    auxiliary_loss(run)
                    backward_standard(run)
                record_fit_window(run)
                optimizer_step(run)
            if not finish_batch(run):
                break
        epoch_diagnostics(run)
        if args.profile_steps:
            report_profile(run)
            return
        if args.parity_dump:
            write_parity(run)
            return
        evaluate_epoch(run)
        save_epoch(run)
    finalize(run)
