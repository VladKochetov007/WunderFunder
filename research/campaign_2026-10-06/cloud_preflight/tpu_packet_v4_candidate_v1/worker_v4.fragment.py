def _combo_sufficient_statistics(
    jax: Any,
    jnp: Any,
    prediction: Any,
    batch: dict[str, Any],
    ema_prediction: Any,
) -> tuple[Any, Any]:
    """Return the frozen combo-loss component numerators and denominators.

    JAX arrays and parameter pytrees are backend-dependent runtime values, so
    ``Any`` is unavoidable at this device boundary.
    """
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    primary_num = jnp.sum(
        primary_weights
        * (
            prediction[..., :2].astype(jnp.float32)
            - jnp.clip(batch["fixed_target"].astype(jnp.float32), -2.0, 2.0)
        )
        ** 2
    )
    consistency_num = jnp.sum(
        row_weights[..., None]
        * (
            prediction[..., :2].astype(jnp.float32)
            - ema_prediction[..., :2].astype(jnp.float32)
        )
        ** 2
    )
    structural_labels = batch["structural_labels"].astype(jnp.int32)
    structural_num = jnp.asarray(0.0, dtype=jnp.float32)
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        log_probs = jax.nn.log_softmax(prediction[..., start : start + width], axis=-1)
        selected = jnp.take_along_axis(
            log_probs,
            structural_labels[..., target_index, None],
            axis=-1,
        )[..., 0]
        structural_num -= jnp.sum(selected * row_weights)

    next_weights = batch["next_weights"].astype(jnp.float32)
    next_distance = (batch["next_rows"].astype(jnp.float32) - 3.0) / 2.0
    next_mse = (prediction[..., 21].astype(jnp.float32) - next_distance) ** 2
    next_logit = prediction[..., 22].astype(jnp.float32)
    next_bce = jax.nn.softplus(next_logit) - batch["next_up"].astype(jnp.float32) * next_logit
    next_num = jnp.sum((next_mse + next_bce) * next_weights)
    numerators = jnp.stack((primary_num, consistency_num, structural_num, next_num))
    denominators = jnp.stack(
        (
            jnp.sum(primary_weights),
            jnp.sum(row_weights),
            jnp.sum(next_weights),
        )
    )
    return numerators, denominators


def _combo_denominators(jnp: Any, batch: dict[str, Any]) -> Any:
    """Compute only the three global normalization masses for one shard."""
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    return jnp.stack(
        (
            jnp.sum(primary_weights),
            jnp.sum(row_weights),
            jnp.sum(batch["next_weights"].astype(jnp.float32)),
        )
    )


def _combo_loss_from_global_sums(jnp: Any, numerators: Any, denominators: Any) -> Any:
    """Combine raw sums only after their masses have been reduced globally."""
    primary = numerators[0] / jnp.maximum(denominators[0], 1e-8)
    consistency = numerators[1] / jnp.maximum(denominators[1], 1e-8)
    structural = numerators[2] / jnp.maximum(denominators[1], 1e-8)
    next_aux = numerators[3] / jnp.maximum(denominators[2], 1e-8)
    return primary + consistency + 0.3 * structural + 0.3 * next_aux


def _synthetic_sharded_batch(np: Any, rows: int) -> tuple[Any, dict[str, Any], Any]:
    """Make distinct replica inputs with strongly unequal objective masses."""
    rng = np.random.default_rng(20261007)
    prefix = (8, LOCAL_BATCH, rows)
    shard_scales = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 8.0], dtype=np.float32)
    labels = rng.normal(0.0, 0.2, (*prefix, 2)).astype(np.float32)
    fixed_target = np.clip(
        labels * 0.7 + rng.normal(0.0, 0.05, (*prefix, 2)), -2.0, 2.0
    ).astype(np.float32)
    row_weights = rng.uniform(0.1, 1.0, prefix).astype(np.float32)
    row_weights *= shard_scales[:, None, None]
    row_weights[:, :, ::31] = 0.0
    next_weights = rng.uniform(0.1, 1.0, prefix).astype(np.float32)
    next_weights *= shard_scales[:, None, None]
    next_weights[0] = 0.0
    inputs = rng.normal(0.0, 0.1, (*prefix, DIMENSIONS[0])).astype(np.float32)
    ema_prediction = rng.normal(0.0, 0.15, (*prefix, 2)).astype(np.float32)
    structural_labels = np.stack(
        (
            rng.integers(0, 12, size=prefix),
            rng.integers(0, 3, size=prefix),
            rng.integers(0, 4, size=prefix),
        ),
        axis=-1,
    ).astype(np.int32)
    batch = {
        "labels": labels,
        "fixed_target": fixed_target,
        "main_weights": row_weights,
        "structural_labels": structural_labels,
        "next_weights": next_weights,
        "next_rows": rng.integers(0, 8, size=prefix).astype(np.float32),
        "next_up": rng.integers(0, 2, size=prefix).astype(np.float32),
    }
    return inputs, batch, ema_prediction


def _flatten_replica_axis(np: Any, value: Any) -> Any:
    """Concatenate eight local batches into the matching global batch."""
    return value.reshape((GLOBAL_BATCH, *value.shape[2:]))


def _gradient_error(jax: Any, np: Any, reference: Any, observed: Any) -> dict[str, float]:
    """Summarize a pytree difference after synchronizing values to the host."""
    squared_error = 0.0
    squared_reference = 0.0
    max_absolute = 0.0
    for reference_leaf, observed_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(observed),
        strict=True,
    ):
        reference_array = np.asarray(jax.device_get(reference_leaf), dtype=np.float64)
        observed_array = np.asarray(jax.device_get(observed_leaf), dtype=np.float64)
        difference = observed_array - reference_array
        squared_error += float(np.sum(difference**2))
        squared_reference += float(np.sum(reference_array**2))
        max_absolute = max(max_absolute, float(np.max(np.abs(difference))))
    return {
        "relative_l2": float(np.sqrt(squared_error) / max(np.sqrt(squared_reference), 1e-12)),
        "max_absolute": max_absolute,
    }


def run_device_worker() -> dict[str, Any]:
    """Check full-objective sharded gradients and profile synthetic TPU updates."""
    global PHASE
    record_phase("framework_import", "device_worker")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from combo_loss_reference import component_sums, loss_from_sums
    from jax import lax

    source_root = Path(os.environ["WNN_E463_SOURCE_ROOT"])
    run = source_root / "artifacts/runs/c20261003_combo_s3_f0"
    npz = run / "model_ep3.jax.npz"
    sys.path.insert(0, str(source_root / "src"))
    from wnn_jax import bridge, rnn

    record_phase("actual_tpu_inventory", "device_worker")
    local_devices = list(jax.local_devices())
    all_devices = list(jax.devices())
    fallback_hbm_capacity, tpu_info = tpu_hbm_capacity(jax)
    records = device_records(all_devices, fallback_hbm_capacity)
    write_json(
        OUTPUT / "tpu_inventory.json",
        {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "local_device_count": len(local_devices),
            "global_device_count": len(all_devices),
            "tpu_hardware_info": tpu_info,
            "devices": records,
        },
    )
    if jax.default_backend() != "tpu" or len(local_devices) != 8 or len(all_devices) != 8:
        raise RuntimeError(
            "requested TpuV5E8 must expose exactly eight local/global TPU devices; "
            f"backend={jax.default_backend()}, local={len(local_devices)}, global={len(all_devices)}"
        )
    if any(device.platform != "tpu" for device in all_devices):
        raise RuntimeError("JAX exposed a non-TPU device; CPU fallback is not accepted")
    for device_record in records:
        current = device_record["hbm_current_bytes"]
        limit = device_record["hbm_capacity_bytes"]
        if current is None or limit is None:
            raise RuntimeError(
                f"TPU {device_record['id']} did not report measurable HBM capacity/current use"
            )
        if limit - current < int(limit * 0.20):
            raise MemoryError(f"TPU {device_record['id']} lacks 20% free HBM before the test")
    write_json(OUTPUT / "tpu_hbm_before.json", records)

    record_phase("checkpoint_hash_and_load", "device_worker")
    input_identity = verify_inputs(source_root)
    params = rnn.load_npz_state(npz, *DIMENSIONS)
    optimizer = rnn.adamw_init(params)

    def make_fullbatch_shard(value: Any) -> Any:
        return jnp.asarray(_flatten_replica_axis(np, value))

    def make_replicated(value: Any) -> Any:
        host_value = np.asarray(jax.device_get(value))
        return jnp.asarray(np.repeat(host_value[None, ...], len(local_devices), axis=0))

    def mapped_update_one_replica(
        model_params: Any,
        opt_state: Any,
        sequence: Any,
        batch: dict[str, Any],
        ema_targets: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        local_denominators = _combo_denominators(jnp, batch)
        global_denominators = lax.psum(local_denominators, "tpu_replicas")

        def objective(current: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            prediction, next_state = rnn.forward_chunk(current, sequence, precision="fp32")
            numerators, _ = _combo_sufficient_statistics(jax, jnp, prediction, batch, ema_targets)
            local_loss = _combo_loss_from_global_sums(
                jnp, numerators, global_denominators
            )
            finite = jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(next_state))
            return local_loss, (finite, next_state, numerators)

        (local_loss, (finite_forward, next_state, local_numerators)), local_gradients = (
            jax.value_and_grad(objective, has_aux=True)(model_params)
        )
        global_gradients = jax.tree_util.tree_map(
            lambda gradient: lax.psum(gradient, "tpu_replicas"), local_gradients
        )
        global_loss = lax.psum(local_loss, "tpu_replicas")
        global_numerators = lax.psum(local_numerators, "tpu_replicas")
        updated, next_optimizer = rnn.adamw_update(
            model_params,
            global_gradients,
            opt_state,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        finite_gradients = jnp.asarray(True)
        for gradient in jax.tree_util.tree_leaves(global_gradients):
            finite_gradients = finite_gradients & jnp.all(jnp.isfinite(gradient))
        finite_updated = jnp.asarray(True)
        for parameter in jax.tree_util.tree_leaves(updated):
            finite_updated = finite_updated & jnp.all(jnp.isfinite(parameter))
        return (
            updated,
            next_optimizer,
            global_loss,
            global_gradients,
            global_numerators,
            global_denominators,
            finite_forward & finite_updated,
            finite_gradients & jnp.all(jnp.isfinite(next_state)),
        )

    record_phase("synthetic_unequal_mass_fixture", "device_worker")
    full_inputs, full_batch_np, full_ema_np = _synthetic_sharded_batch(np, CHUNK_ROWS)
    if float(np.sum(full_batch_np["next_weights"][0])) != 0.0:
        raise AssertionError("synthetic negative-control replica must have zero next mass")
    if float(np.sum(full_batch_np["next_weights"][1:])) <= 0.0:
        raise AssertionError("synthetic fixture must have positive global next mass")
    if np.array_equal(full_inputs[0], full_inputs[1]):
        raise AssertionError("synthetic replica inputs must be distinct")
    per_replica_denominators = np.stack(
        [
            _combo_denominators(
                np,
                {name: value[replica_index] for name, value in full_batch_np.items()},
            )
            for replica_index in range(8)
        ]
    )
    row_masses = per_replica_denominators[:, 1]
    if np.any(row_masses <= 0.0) or float(np.max(row_masses) / np.min(row_masses)) < 10.0:
        raise AssertionError(
            f"synthetic replica main-weight masses are not sufficiently unequal: {row_masses}"
        )
    fixture_denominators = _combo_denominators(
        np,
        {
            name: value.reshape((GLOBAL_BATCH, *value.shape[2:]))
            for name, value in full_batch_np.items()
        },
    )
    if not np.isfinite(fixture_denominators).all() or np.any(fixture_denominators <= 1e-8):
        raise AssertionError(f"synthetic fixture has invalid global normalization masses: {fixture_denominators}")
    shard_params = jax.tree_util.tree_map(make_replicated, params)
    shard_optimizer = jax.tree_util.tree_map(make_replicated, optimizer)
    mapped_update = jax.pmap(
        mapped_update_one_replica,
        axis_name="tpu_replicas",
        in_axes=(0, 0, 0, 0, 0),
        devices=local_devices,
    )

    record_phase("global_batch_reference_gradient", "device_worker")
    parity_rows = 128
    parity_inputs = full_inputs[:, :, :parity_rows]
    parity_batch_np = {name: value[:, :, :parity_rows] for name, value in full_batch_np.items()}
    parity_ema_np = full_ema_np[:, :, :parity_rows]
    full_inputs_ref = make_fullbatch_shard(parity_inputs)
    full_batch_ref = {
        name: make_fullbatch_shard(value) for name, value in parity_batch_np.items()
    }
    full_ema_ref = make_fullbatch_shard(parity_ema_np)

    def full_batch_objective(current: Any) -> tuple[Any, tuple[Any, Any]]:
        prediction, next_state = rnn.forward_chunk(current, full_inputs_ref, precision="fp32")
        loss = bridge.combo_sbd_loss(prediction, full_batch_ref, full_ema_ref)
        return loss, (prediction, next_state)

    reference_value_and_grad = jax.jit(
        jax.value_and_grad(full_batch_objective, has_aux=True)
    )
    (reference_loss, (reference_prediction, reference_state)), reference_gradients = (
        reference_value_and_grad(params)
    )
    jax.block_until_ready((reference_loss, reference_prediction, reference_state, reference_gradients))
    check_finite(jnp, reference_prediction, "full-batch reference prediction")
    check_finite(jnp, reference_state, "full-batch reference state")
    for leaf in jax.tree_util.tree_leaves(reference_gradients):
        check_finite(jnp, leaf, "full-batch reference gradient")

    flat_batch_np = {name: _flatten_replica_axis(np, value) for name, value in parity_batch_np.items()}
    flat_ema_np = _flatten_replica_axis(np, parity_ema_np)
    independent_sums = component_sums(
        np.asarray(jax.device_get(reference_prediction)),
        flat_batch_np,
        flat_ema_np,
    )
    independent_loss = loss_from_sums(independent_sums)
    source_loss = float(np.asarray(jax.device_get(reference_loss)))
    if not np.isclose(source_loss, independent_loss, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"frozen JAX combo loss differs from independent NumPy equation: "
            f"source={source_loss}, numpy={independent_loss}"
        )

    record_phase("unequal_mass_distributed_gradient_parity", "device_worker")
    parity_batch = {
        name: jnp.asarray(value) for name, value in parity_batch_np.items()
    }
    parity_ema = jnp.asarray(parity_ema_np)
    parity_result = mapped_update(
        shard_params,
        shard_optimizer,
        jnp.asarray(parity_inputs),
        parity_batch,
        parity_ema,
    )
    jax.block_until_ready(parity_result)
    (
        _,
        _,
        distributed_losses,
        distributed_gradients,
        distributed_numerators,
        distributed_denominators,
        finite_forward,
        finite_gradients,
    ) = parity_result
    distributed_loss = float(np.asarray(jax.device_get(distributed_losses[0])))
    if not np.isclose(distributed_loss, source_loss, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"global distributed loss differs from full batch: "
            f"distributed={distributed_loss}, full={source_loss}"
        )
    if not bool(jnp.all(finite_forward)) or not bool(jnp.all(finite_gradients)):
        raise FloatingPointError("unequal-mass distributed update produced nonfinite tensors")
    observed_gradients = jax.tree_util.tree_map(lambda value: value[0], distributed_gradients)
    gradient_metrics = _gradient_error(jax, np, reference_gradients, observed_gradients)
    for reference_leaf, observed_leaf in zip(
        jax.tree_util.tree_leaves(reference_gradients),
        jax.tree_util.tree_leaves(observed_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(observed_leaf)),
            np.asarray(jax.device_get(reference_leaf)),
            rtol=3e-4,
            atol=3e-5,
        )
    observed_sums = np.asarray(jax.device_get(distributed_numerators[0]), dtype=np.float64)
    expected_sums = np.asarray(
        [
            independent_sums["primary_num"],
            independent_sums["consistency_num"],
            independent_sums["structural_num"],
            independent_sums["next_num"],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed_sums, expected_sums, rtol=1e-4, atol=1e-5)
    observed_denominators = np.asarray(
        jax.device_get(distributed_denominators[0]), dtype=np.float64
    )
    expected_denominators = np.asarray(
        [independent_sums["primary_den"], independent_sums["row_den"], independent_sums["next_den"]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed_denominators, expected_denominators, rtol=1e-5, atol=1e-5)

    record_phase("locally_normalized_pmean_negative_control", "device_worker")

    def negative_control_one_replica(
        model_params: Any,
        sequence: Any,
        batch: dict[str, Any],
        ema_targets: Any,
    ) -> tuple[Any, Any]:
        def local_objective(current: Any) -> Any:
            prediction, _ = rnn.forward_chunk(current, sequence, precision="fp32")
            return bridge.combo_sbd_loss(prediction, batch, ema_targets)

        local_loss, local_gradients = jax.value_and_grad(local_objective)(model_params)
        mean_gradients = jax.tree_util.tree_map(
            lambda gradient: lax.pmean(gradient, "tpu_replicas"), local_gradients
        )
        return lax.pmean(local_loss, "tpu_replicas"), mean_gradients

    negative_control = jax.pmap(
        negative_control_one_replica,
        axis_name="tpu_replicas",
        in_axes=(0, 0, 0, 0),
        devices=local_devices,
    )
    bad_losses, bad_gradients = negative_control(
        shard_params,
        jnp.asarray(parity_inputs),
        parity_batch,
        parity_ema,
    )
    jax.block_until_ready((bad_losses, bad_gradients))
    bad_observed = jax.tree_util.tree_map(lambda value: value[0], bad_gradients)
    bad_gradient_metrics = _gradient_error(jax, np, reference_gradients, bad_observed)
    if bad_gradient_metrics["relative_l2"] < 1e-3:
        raise AssertionError(
            "unequal-mass negative control failed to distinguish local pmean "
            f"from the global-batch gradient: {bad_gradient_metrics}"
        )
    write_json(
        OUTPUT / "tpu_objective_reduction_parity.json",
        {
            "status": "passed",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parity_global_batch": GLOBAL_BATCH,
            "parity_local_batch_per_tpu": LOCAL_BATCH,
            "parity_chunk_rows": parity_rows,
            "synthetic_seed": 20261007,
            "fixed_ema_predictions_synthetic": True,
            "raw_training_data_consumed": False,
            "source_jax_loss": source_loss,
            "independent_numpy_loss": independent_loss,
            "distributed_global_loss": distributed_loss,
            "independent_component_sums": independent_sums,
            "per_replica_component_denominators": per_replica_denominators.tolist(),
            "distributed_component_numerators": observed_sums.tolist(),
            "distributed_component_denominators": observed_denominators.tolist(),
            "distributed_gradient_vs_full_batch": gradient_metrics,
            "invalid_pmean_gradient_vs_full_batch": bad_gradient_metrics,
            "negative_control_losses": np.asarray(jax.device_get(bad_losses)).tolist(),
            "pass_tolerances": {"loss_rtol": 1e-4, "loss_atol": 1e-5, "gradient_rtol": 3e-4, "gradient_atol": 3e-5},
        },
    )

    record_phase("fullshape_synthetic_update_profile", "device_worker")
    fullshape_inputs = jnp.asarray(full_inputs)
    fullshape_batch = {name: jnp.asarray(value) for name, value in full_batch_np.items()}
    fullshape_ema = jnp.asarray(full_ema_np)
    profile_params = jax.tree_util.tree_map(make_replicated, params)
    profile_optimizer = jax.tree_util.tree_map(make_replicated, optimizer)
    warmup_started = time.monotonic()
    profile_state = mapped_update(
        profile_params,
        profile_optimizer,
        fullshape_inputs,
        fullshape_batch,
        fullshape_ema,
    )
    jax.block_until_ready(profile_state)
    warmup_seconds = time.monotonic() - warmup_started
    profile_params, profile_optimizer = profile_state[0], profile_state[1]
    fullshape_times = []
    for _ in range(3):
        update_started = time.monotonic()
        profile_state = mapped_update(
            profile_params,
            profile_optimizer,
            fullshape_inputs,
            fullshape_batch,
            fullshape_ema,
        )
        jax.block_until_ready(profile_state)
        fullshape_times.append(time.monotonic() - update_started)
        profile_params, profile_optimizer = profile_state[0], profile_state[1]
    step_counts = np.asarray(jax.device_get(profile_optimizer.count)).tolist()
    if step_counts != [4] * len(local_devices):
        raise AssertionError(f"synthetic profile should perform warmup+3 updates: {step_counts}")
    final_profile = profile_state
    if not bool(jnp.all(final_profile[6])) or not bool(jnp.all(final_profile[7])):
        raise FloatingPointError("full-shape synthetic profile produced nonfinite tensors")
    check_finite(jnp, final_profile[2], "full-shape loss")
    check_finite(jnp, final_profile[4], "full-shape component numerators")
    check_finite(jnp, final_profile[5], "full-shape component denominators")
    for leaf in jax.tree_util.tree_leaves(final_profile[0]):
        check_finite(jnp, leaf, "full-shape updated parameter")
    for leaf in jax.tree_util.tree_leaves(final_profile[3]):
        check_finite(jnp, leaf, "full-shape global gradient")
    for leaf in jax.tree_util.tree_leaves(profile_optimizer):
        check_finite(jnp, leaf, "full-shape optimizer state")
    profile_records = {
        "status": "passed",
        "global_batch": GLOBAL_BATCH,
        "local_batch_per_tpu": LOCAL_BATCH,
        "chunk_rows": CHUNK_ROWS,
        "input_width": DIMENSIONS[0],
        "hidden_width": DIMENSIONS[1],
        "layers": DIMENSIONS[2],
        "output_width": DIMENSIONS[3],
        "synthetic_inputs_and_targets": True,
        "same_seeded_fixture_reused_for_profile_steps": True,
        "per_replica_component_denominators": per_replica_denominators.tolist(),
        "compile_warmup_update_seconds": warmup_seconds,
        "steady_state_update_seconds": fullshape_times,
        "optimizer_steps_per_replica_including_warmup": step_counts,
        "finite_loss": np.asarray(jax.device_get(final_profile[2])).tolist(),
        "finite_outputs_states_gradients_parameters": True,
        "checkpoint_write_or_export": False,
        "scientific_score_claim": None,
    }
    write_json(OUTPUT / "tpu_fullshape_profile.json", profile_records)

    record_phase("hbm_headroom_validation", "device_worker")
    final_records = device_records(all_devices, fallback_hbm_capacity)
    hbm_failures = []
    for device_record in final_records:
        peak = device_record["hbm_peak_bytes"]
        capacity = device_record["hbm_capacity_bytes"]
        if peak is None or capacity is None or peak > int(capacity * 0.80):
            hbm_failures.append(device_record["id"])
    write_json(OUTPUT / "tpu_hbm_after.json", final_records)
    if hbm_failures:
        raise MemoryError(f"TPU devices exceeded the 80% peak HBM cap or lacked stats: {hbm_failures}")
    result = {
        "status": "passed",
        "classification": "actual TPUv5e eight-device unequal-mass combo-objective gradient diagnostic and full-shape synthetic profile",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": len(local_devices),
        "input_identity": input_identity,
        "synthetic_seed": 20261007,
        "no_training_data_used": True,
        "no_checkpoint_saved_or_exported": True,
        "objective_reduction_parity": "tpu_objective_reduction_parity.json",
        "fullshape_profile": profile_records,
        "hbm_before": records,
        "hbm_after": final_records,
        "scientific_fit_or_score": None,
    }
    write_json(OUTPUT / "tpu_diagnostic_complete.json", result)
    return result
