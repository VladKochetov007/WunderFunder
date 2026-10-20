from pathlib import Path
import base64
import datetime
import hashlib
import json
import os
import subprocess

R = Path('/Users/nb/Documents/wunder/WunderFunder-maintenance')
C = R / 'research/campaign_2026-10-19/full_gru_tweaks_r1'
RECORDS = C / 'records/child_dictionary_fit_v1'
D = RECORDS / 'status_20261020T2150Z'
reader = (RECORDS / 'health_20261020T2114Z/REMOTE.py').read_bytes()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert len(reader) == 13027 and sha(reader) == 'f25aed539a796f96f5863fa9cb85a8e7a0b6c08964e1d44cb2a32612ad697653'
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
assert datetime.datetime.now(datetime.timezone.utc) >= datetime.datetime(2026, 10, 20, 21, 50, tzinfo=datetime.timezone.utc)
D.mkdir(exist_ok=False)

def write_at(directory, name, raw):
    with (directory / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

write_at(D, 'REMOTE.py', reader)
write_at(D, 'ROOT_CAPTURE.py', Path(__file__).read_bytes())
start = now()
timed = False
try:
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ConnectionAttempts=1', 'beleriand', 'python3 -E -B -'], input=reader, capture_output=True, timeout=55)
    stdout, stderr, code = result.stdout, result.stderr, result.returncode
except subprocess.TimeoutExpired as error:
    stdout, stderr, code = error.stdout or b'', error.stderr or b'', None
    timed = True
end = now()
write_at(D, 'STDOUT.raw.json', stdout)
write_at(D, 'STDERR.raw.txt', stderr)
transport = {'body_sha256': sha(reader), 'start_utc': start, 'end_utc': end, 'returncode': code, 'timed_out': timed, 'stdout_bytes': len(stdout), 'stdout_sha256': sha(stdout), 'stderr_bytes': len(stderr), 'stderr_sha256': sha(stderr), 'persisted_before_parse': True}
write_at(D, 'TRANSPORT.json', (json.dumps(transport, indent=2) + '\n').encode())
assert code == 0 and not timed, (code, timed)
capture = json.loads(stdout)
for name, item in capture['receipts'].items():
    raw = base64.b64decode(item['base64'], validate=True)
    assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    write_at(D, name, raw)
for name in ('ADMISSION.json', 'LAUNCH.json'):
    assert (D / name).read_bytes() == (RECORDS / name).read_bytes()
u = capture['unit_properties']
assert u['InvocationID'] == '73b10b699e034a1b9952b2377b845095'
state = tuple(u.get(key) for key in ('ActiveState', 'SubState', 'MainPID', 'ExecMainCode', 'ExecMainStatus', 'Result'))
if state != ('active', 'exited', '0', '1', '0', 'success'):
    print(json.dumps({'captured_utc': capture['captured_utc'], 'transport': transport, 'unit_properties': u, 'status': 'NO_NATIVE_EXIT0_PROOF_NO_PAYLOAD_OR_SUPPORT_ACTION'}))
    raise SystemExit(0)
native_exit = datetime.datetime.strptime(u['ExecMainExitTimestamp'], '%a %Y-%m-%d %H:%M:%S %Z').replace(tzinfo=datetime.timezone.utc).isoformat()
assert datetime.datetime.fromisoformat(native_exit) <= datetime.datetime.fromisoformat(capture['captured_utc'])
terminal_dir = RECORDS / 'terminal_native_capture_v1'
terminal_dir.mkdir(exist_ok=False)
for path in D.iterdir():
    if path.is_file():
        write_at(terminal_dir, path.name, path.read_bytes())
terminal = {
    'schema': 'fg19.child-dictionary-fit-native-terminal-evidence.v1',
    'run_name': 'fg19_child_dictionary_f0_s1001_fit_v1',
    'unit': 'computer-vision-homework-fg19-child-dictionary-fit-v1.service',
    'invocation_id': u['InvocationID'],
    'active_state': 'active', 'sub_state': 'exited',
    'main_pid': 0, 'exec_main_code': 1, 'exec_main_status': 0,
    'raw_bytes': len(stdout), 'raw_sha256': sha(stdout),
    'captured_utc': capture['captured_utc'], 'native_exit_utc': native_exit,
    'raw_path': str((terminal_dir / 'STDOUT.raw.json').relative_to(R)),
}
write_at(RECORDS, 'TERMINAL_EVIDENCE.json', (json.dumps(terminal, indent=2) + '\n').encode())
lines = (D / 'unit.log').read_text().splitlines()
fits = [json.loads(line[4:]) for line in lines if line.startswith('FIT ')]
grads = [json.loads(line[11:]) for line in lines if line.startswith('GRAD_STATS ')]
counts = [fit['update_diagnostics']['epoch'] for fit in fits]
summary = {'schema': 'fg19.child-dictionary-native-record-summary.v1', 'typed_native_evidence': terminal, 'completed_epochs': [fit['epoch'] for fit in fits], 'successful_updates': sum(count['optimizer_completed_updates'] for count in counts), 'completed_batches': sum(count['completed_batches'] for count in counts), 'amp_skips': sum(count['amp_skipped_updates'] for count in counts), 'gradient_nonfinite_steps': sum(grad['nonfinite_steps'] for grad in grads), 'fit_records': fits, 'gradient_records': grads, 'run_directory_stat_only': capture['run_directory_stat_only'], 'own_cgroup_state': capture['own_cgroup_state'], 'parent_state': capture['parent_state'], 'no_payloads_read_or_hashed': True, 'tensor_integrity_or_quality_claim': False, 'raw_origin_copy': str((D / 'STDOUT.raw.json').relative_to(R)), 'copy_identical_no_additional_SSH': True}
assert summary['completed_epochs'] == list(range(6)) and len(grads) == 6
assert (summary['successful_updates'], summary['completed_batches'], summary['amp_skips'], summary['gradient_nonfinite_steps']) == (44760, 4476, 0, 0)
write_at(terminal_dir, 'NATIVE_RECORD_SUMMARY.json', (json.dumps(summary, indent=2) + '\n').encode())
print(json.dumps({'captured_utc': capture['captured_utc'], 'native_exit_utc': native_exit, 'transport': transport, 'terminal_evidence_bytes': (RECORDS / 'TERMINAL_EVIDENCE.json').stat().st_size, 'terminal_evidence_sha256': sha((RECORDS / 'TERMINAL_EVIDENCE.json').read_bytes()), 'completed_epochs': summary['completed_epochs'], 'successful_updates': summary['successful_updates'], 'completed_batches': summary['completed_batches'], 'unit_properties': u, 'all_populated': capture['parent_state']['populated_descendants']}))
