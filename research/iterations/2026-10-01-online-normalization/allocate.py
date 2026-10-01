"""Root's prospective allocation; writes a new plan, does not start jobs."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    commands = json.loads((HERE / 'commands.json').read_text())
    sources = dict(commands['source_sha256'])
    for p in sorted((ROOT / 'src/wnn').glob('*.py')):
        sources[str(p.relative_to(ROOT))] = sha(p)
    for p in sorted(HERE.iterdir()):
        if p.suffix in ('.py', '.md') or p.name in ('source-check.json', 'evaluation-generated-checks.json'):
            sources[str(p.relative_to(ROOT))] = sha(p)
    now = datetime.now(timezone.utc)
    value = dict(execution_authorized=True, created_utc=now.isoformat(),
        deadline_utc=(now + timedelta(seconds=7800)).isoformat(),
        commands_sha256=sha(HERE / 'commands.json'), runner_sha256=sha(HERE / 'run.py'),
        source_sha256=sources, input_bindings_path=str((HERE / 'input-bindings.json').relative_to(ROOT)),
        input_bindings_sha256=sha(HERE / 'input-bindings.json'),
        execution_directory=str((HERE / 'execution-01').relative_to(ROOT)),
        resources=dict(gpu_id=6, gpu_uuid='GPU-53eaee12-1803-f8da-7399-ccdbfda91a18',
            allocator_fraction=.375, allocator_max_MiB=6144, owned_GPU_MiB_max=7168,
            minimum_initial_free_MiB=8192, minimum_running_free_MiB=1024,
            cpu_ids=[22, 23, 24, 25, 28, 32], CPUQuota=600,
            host_memory_bytes=20 * 2**30, swap_bytes=0, stage_seconds=7800,
            step_seconds=dict(parent=3600, H2=1200, OOF=1200)),
        selection_budget=dict(candidate_recipes=1, training_seeds=[3], fits=4, OOF=2,
            primary_comparisons_after_complete_pipeline=1, retries=0, within_run_metrics=0),
        source_only_commands_execution_flag='Superseded only by this explicit root allocation; prepared commands themselves remain immutable.',
        score_execution='Only after complete six-step pipeline, using frozen evaluate.py and actual output bindings.',
        GitHub_authorized=False, public_submission_authorized=False)
    with (HERE / 'allocation-01.json').open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps(dict(path=str(HERE / 'allocation-01.json'), sha256=sha(HERE / 'allocation-01.json'),
                         deadline_utc=value['deadline_utc'], sources=len(sources))))


if __name__ == '__main__':
    main()
