"""Independent complete-trace receipt audit; never runs a model or reads targets.

Run only after root authorizes completed V1 handoff. A timeout/partial trace needs
a separate failure audit. Complete integrity and runtime gates remain separate.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, math

P = Path(__file__).resolve().parent.parent
ROOT = P.parents[2]
D = P / 'v1-runtime-audit'
E = D / 'execution'
PLAN_SHA = '881a8476ece923c36fb2a1405b9a1525438426aa6e1ed02cef0d1cff15af77e7'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (a, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allocation', type=Path, required=True)
    ap.add_argument('--allocation-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    assert sha(D / 'plan.json') == PLAN_SHA
    assert sha(args.allocation) == args.allocation_sha256
    plan, alloc = read(D / 'plan.json'), read(args.allocation)
    assert alloc['v1_runtime_audit_authorized'] is True and alloc['attempts'] == 1
    assert alloc['plan_sha256'] == PLAN_SHA
    for rel, expected in plan['source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    launch, start = read(E / 'launch-result.json'), read(E / 'launch.json')
    worker, process = read(E / 'worker-result.json'), read(E / 'worker-process.json')
    assert launch['worker_result_sha256'] == sha(E / 'worker-result.json')
    assert launch['log_sha256'] == sha(E / 'worker.log')
    for receipt in [launch, start, worker]:
        assert receipt['plan_sha256'] == PLAN_SHA
        assert receipt['allocation_sha256'] == args.allocation_sha256
    assert launch['status'] == worker['status'] == 'COMPLETE'
    assert launch['exit_code'] == 0 and not launch['hard_timeout']
    assert launch['attempts'] == 1 and launch['retry_allowed'] is False
    assert process['pid'] == process['process_group'] == worker['pid']
    assert process['start_ticks'] > 0
    expected_argv = [str(Path(plan['scorer_environment']) / 'run.sh'), str(D / 'audit_v1.py'),
                     '--plan', str(D / 'plan.json'), '--allocation', str(args.allocation.resolve()),
                     '--allocation-sha256', args.allocation_sha256]
    assert process['argv'] == launch['argv'] == start['argv'] == expected_argv
    caps = {'cpu.max': '100000 100000', 'memory.max': str(16 << 30), 'memory.swap.max': '0'}
    assert launch['cgroup_limits'] == start['cgroup_limits'] == caps
    assert all(worker['cgroup_limits'][k] == v for k, v in caps.items())
    assert launch['affinity'] == start['affinity'] == [18]
    assert launch['RLIMIT_AS'] == start['RLIMIT_AS'] == 16 << 30
    assert launch['environment'] == start['environment'] == {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'CUDA_VISIBLE_DEVICES': '', 'ARROW_NUM_THREADS': '1', 'MALLOC_CONF': 'background_thread:false'}
    level = worker['natural_native_level']
    assert level in ['v2', 'avx2', 'v3']
    native = f'{level}/wnn_cpp.cpython-311-x86_64-linux-gnu.so'
    assert worker['native_sha256'] == plan['artifact']['files'][native]['sha256']
    assert worker['governor_defaults'] == {'EXPECTED_ROWS': 39400000, 'BUDGET_S': 3120,
                                           'override_variables_absent': True}
    assert worker['initial_members'] == plan['artifact']['members']
    assert worker['initial_coefficients'] == [.25] * 4
    assert worker['library_versions'] == plan['library_versions']
    identity = {k: plan['input'][k] for k in ['bytes', 'mtime_ns', 'resolved_path', 'sha256']}
    assert worker['input_identity_before'] == worker['input_identity_after'] == identity
    artifact = ROOT / plan['artifact']['path']
    assert sha(artifact) == plan['artifact']['sha256'] == 'ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d'
    rows = [json.loads(line) for line in (E / 'sequences.jsonl').read_text().splitlines()]
    schedule = list(range(1873)) + list(range(97))
    assert len(rows) == len(worker['raw_read_calls']) == 1970
    seen, original, next_fresh = set(), [], 0
    expected_k, transitions, retained = 4, [], {str(k): 0 for k in [1, 2, 3, 4]}
    for index, (row, raw, group) in enumerate(zip(rows, worker['raw_read_calls'], schedule)):
        assert row['schedule_index'] == index and row['row_group'] == group
        assert row['callbacks'] == 20000 and row['emitted'] == 19901
        assert row['repeated_raw_sequence'] is (index >= 1873)
        raw_id = row['raw_seq_ix']
        if index < 1873:
            assert raw_id not in seen
            expected_id = raw_id
            original.append(raw_id)
        else:
            assert raw_id == original[group]
            while next_fresh in seen:
                next_fresh += 1
            expected_id, next_fresh = next_fresh, next_fresh + 1
        assert row['seq_ix'] == row['wrapper_seq_ix_after'] == expected_id
        assert expected_id not in seen
        seen.add(expected_id)
        assert raw == {'row_group': group, 'seq_ix': raw_id, 'rows': 20000}
        g = row['governor']
        assert g['k_before_boundary'] == expected_k
        assert 1 <= g['k_after_sequence'] == g['k_after_first_callback'] <= expected_k
        assert g['rows_before'] == index * 20000 and g['rows_after'] == (index + 1) * 20000
        assert g['govern_calls_before'] == max(0, index - 1) and g['govern_calls_after'] == index
        expected_k = g['k_after_sequence']
        assert g['active_members'] == plan['artifact']['members'][:expected_k]
        assert g['active_coefficients'] == [1 / expected_k] * expected_k
        retained[str(expected_k)] += 20000
        if g['k_before_boundary'] != expected_k:
            transitions.append({'schedule_index': index, 'row_group': group, **g})
        assert len(row['emitted_f32_sha256']) == 64
        for key in ['callback_loop_wall_seconds', 'callback_loop_cpu_seconds', 'sequence_total_seconds']:
            assert math.isfinite(row[key]) and row[key] > 0
    assert worker['member_count_transitions'] == transitions
    assert worker['final_member_count'] == expected_k
    assert worker['callback_count'] == worker['final_model_rows'] == sum(r['callbacks'] for r in rows) == 39400000
    assert worker['emitted_count'] == sum(r['emitted'] for r in rows) == 39204970
    assert worker['warmup_none_count'] == 195030 and worker['final_govern_calls'] == 1969
    assert worker['completed_sequences'] == worker['unique_sequence_ids'] == worker['sequence_resets_observed'] == len(seen) == 1970
    assert worker['unique_raw_sequences'] == len(set(original)) == 1873
    assert worker['repeated_raw_rows'] == 1940000
    assert worker['targets_decoded'] is False and worker['scoring_masks_read'] is False and worker['WP_queries'] == 0
    assert worker['opaque_native_hidden_state_serialized'] is False
    for key in ['callback_loop_wall_seconds', 'callback_loop_cpu_seconds', 'io_decode_conversion_seconds', 'validation_checksum_seconds']:
        close(sum(row[key] for row in rows), worker[key])
    for cpu in ['cpu18', 'cpu58']:
        ticks = sum(r['callback_activity'][cpu]['total_ticks'] for r in rows)
        busy = sum(r['callback_activity'][cpu]['busy_ticks'] for r in rows)
        assert worker['callback_activity'][cpu]['total_ticks'] == ticks > 0
        assert worker['callback_activity'][cpu]['busy_ticks'] == busy
        close(busy / ticks, worker['callback_activity'][cpu]['busy_fraction'])
        whole = worker['whole_trace_activity'][cpu]
        close(whole['busy_ticks'] / whole['total_ticks'], whole['busy_fraction'])
    statuses = [worker[k] for k in ['process_status_after_dependencies', 'final_process_status', 'postguard_process_status']]
    statuses += [r[k] for r in rows for k in ['status_before', 'status_after']]
    for status in statuses:
        assert status['threads'] == len(status['tasks'])
        assert all(t['affinity'] == [18] for t in status['tasks'].values())
    maxrss = max([worker['max_rss_KiB']] + [s[k] for s in statuses for k in ['vm_rss_KiB', 'vm_hwm_KiB']])
    assert maxrss == worker['max_observed_rss_KiB'] < 16 * 1024 * 1024
    assert max(s['threads'] for s in statuses) == worker['max_observed_os_threads']
    close(worker['callback_average_us'], worker['callback_loop_wall_seconds'] / 39400000 * 1e6)
    close(worker['actual_callback_minutes_39_4M'], worker['callback_loop_wall_seconds'] / 60)
    close(launch['actual_full_process_minutes_39_4M'], launch['process_wall_seconds'] / 60)
    assert worker['worker_total_seconds'] <= launch['process_wall_seconds']
    assert worker['first_callback_worker_elapsed_seconds'] < worker['last_callback_worker_elapsed_seconds'] <= worker['worker_total_seconds']
    quiet_pre = worker['quiet_precheck']['cpu58']['busy_fraction'] < .05
    quiet_callback = worker['callback_activity']['cpu58']['busy_fraction'] < .05
    quiet_whole = worker['whole_trace_activity']['cpu58']['busy_fraction'] < .05
    assert worker['callback_smt_clean'] == quiet_callback and worker['whole_trace_smt_clean'] == quiet_whole
    assert worker['quiet_runtime_qualified'] == launch['quiet_runtime_qualified'] == (quiet_callback and quiet_whole)
    gates = {'complete_finite_trace': True, 'quiet_precheck': quiet_pre, 'quiet_callbacks': quiet_callback,
             'quiet_whole_trace': quiet_whole, 'actual_full_process_le60min': launch['process_wall_seconds'] <= 3600,
             'observed_external_cap_le65min': launch['process_wall_seconds'] <= 3900, 'not_timed_out': not launch['hard_timeout']}
    assert launch['functional_complete'] is True
    assert launch['actual_full_process_at_most_60min'] == gates['actual_full_process_le60min']
    assert launch['process_deadline_observed'] == gates['observed_external_cap_le65min']
    assert launch['delivery_runtime_qualified'] == all(gates.values())
    release = read(E / 'release-check.json')
    assert release['worker_PID'] == worker['pid'] and release['worker_proc_exists'] is False
    assert 'ActiveState=inactive' in release['scope_show_stdout'] and 'SubState=dead' in release['scope_show_stdout']
    files = [D / 'plan.json', args.allocation, E / 'launch.json', E / 'worker-process.json',
             E / 'worker-result.json', E / 'launch-result.json', E / 'sequences.jsonl', E / 'worker.log', E / 'release-check.json']
    result = {'status': 'PASS_INDEPENDENT_COMPLETE_V1_RECEIPT_INTEGRITY',
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checker_sha256': sha(__file__),
              'delivery_runtime_qualified': all(gates.values()), 'gates': gates,
              'callbacks': 39400000, 'emitted_pairs': 39204970, 'sequences': 1970,
              'native_level': level, 'callback_minutes': worker['callback_loop_wall_seconds'] / 60,
              'full_process_minutes': launch['process_wall_seconds'] / 60,
              'member_count_transitions': transitions, 'callbacks_by_member_count': retained,
              'max_observed_rss_KiB': maxrss, 'max_observed_OS_threads': worker['max_observed_os_threads'],
              'SMT58_callback_busy_fraction': worker['callback_activity']['cpu58']['busy_fraction'],
              'SMT58_whole_busy_fraction': worker['whole_trace_activity']['cpu58']['busy_fraction'],
              'plan_sha256': PLAN_SHA, 'allocation_sha256': args.allocation_sha256,
              'receipts': {str(path): sha(path) for path in files}, 'worker_and_scope_released': True,
              'targets_read': False, 'model_or_empirical_inputs_executed': False, 'WP_queries': 0,
              'limits': ['Receipt audit only; no independent inference replay or new output oracle.',
                         'Observed governor member counts do not establish public-run counts or quality.',
                         'One natural-ISA warm-cache shared-host trace; no other platform qualification.',
                         'Complete integrity and runtime feasibility are reported separately.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['receipts', 'member_count_transitions']}))

if __name__ == '__main__':
    main()
