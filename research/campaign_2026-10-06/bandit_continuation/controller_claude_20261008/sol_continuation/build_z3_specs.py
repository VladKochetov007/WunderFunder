from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
OUT = Path(__file__).resolve().parent
QUEUE = ROOT / 'research/campaign_2026-10-06/bandit_continuation/runtime/warm_queue_local_v1'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(name: str, spec: dict) -> None:
    (OUT / name).write_text(json.dumps(spec, indent=2) + '\n')


def replace_value(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = value


def main() -> None:
    z3 = read(QUEUE / 'running/z3-combo-s4-parent-student-v1.json')
    spec = copy.deepcopy(z3)
    spec.update(run_id='f0_z3_combo_s4_student_v1', priority=8, arm='Z3-F0-EXACT', hypothesis='Z3 matched seed4 fold0 analogue: incumbent Muon parent -> raw-family distilled student; fixed ep1', notes='Fold0 exposed development evidence. Only membership, fold-specific parent/cache change from all-data Z3; same source and student science. Not a full faithful heterogeneous recipe or public forecast.')
    fit = spec['steps'][1]
    argv = fit['argv']
    replace_value(argv, '--name', spec['run_id'])
    replace_value(argv, '--valid-fold', '0')
    replace_value(argv, '--init', 'c20261003_combo_s4_f0@ep3')
    replace_value(argv, '--distill-cache-root', str(ROOT / 'cache/bd26_h230_w384_family_v1'))
    argv.remove('--include-heldout-valid')
    argv.remove('--distill-cache-all-data')
    argv.extend(['--split-manifest', str(ROOT / 'research/campaign_2026-09-29/artifacts/splits/fold0.json')])
    spec['steps'] = [fit]
    spec['expected_outputs'] = [str(ROOT / 'artifacts/runs' / spec['run_id'] / filename) for filename in ('model_ep1.pt', 'provenance.json', 'history.json')]
    save('f0_z3_combo_s4_student_v1.json', spec)
    for treatment in ('parent', 'student'):
        control = read(QUEUE / 'pending/f0_combo_s3_parent_hft_v1.json')
        name = f'f0_z3_combo_s4_{treatment}_hft_v1'
        init = 'c20261003_combo_s4_f0@ep3' if treatment == 'parent' else 'f0_z3_combo_s4_student_v1_core@ep1'
        control.update(run_id=name, priority=9, arm='Z3-F0-CONTROL' if treatment == 'parent' else 'Z3-F0-ENDPOINT', notes='Matched seed4, fixed epoch0 GCorr finish. Independent grader owns point/delta/bootstrap; fixedhalf endpoint. No queue-owned grading.')
        control['steps'] = control['steps'][:2]
        fit = control['steps'][0]
        replace_value(fit['argv'], '--name', name)
        replace_value(fit['argv'], '--init', init)
        replace_value(fit['argv'], '--seed', '4')
        fit['argv'].extend(['--valid-fold', '0'])
        control['steps'][1]['argv'][2] = name
        if treatment == 'student':
            core = copy.deepcopy(read(QUEUE / 'pending/f0_h230_hetero_s3_hft_v1.json')['steps'][0])
            core['argv'][-3:] = ['f0_z3_combo_s4_student_v1', '1', 'f0_z3_combo_s4_student_v1_core']
            control['steps'].insert(0, core)
            predict = copy.deepcopy(control['steps'][-1])
            predict['name'] = 'predict_core'
            predict['argv'][2] = 'f0_z3_combo_s4_student_v1_core'
            predict['argv'][-1] = '1'
            control['steps'].append(predict)
        control['expected_outputs'] = [str(ROOT / 'artifacts/runs' / name / 'model_ep0.pt'), str(ROOT / 'cache' / f'pred_{name}_valid.npy'), str(ROOT / 'cache' / f'pred_{name}_valid.json')]
        save(name + '.json', control)


if __name__ == '__main__':
    main()
