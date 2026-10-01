"""Independent source/parsed-recipe/generated-time checks; no actual outcomes."""
from pathlib import Path
import copy, datetime, hashlib, importlib.util, json

P = Path(__file__).resolve().parent.parent
ROOT = P.parents[2]
S = P / 'scalar'
D = S / 'replication-preparation'
UTC = datetime.timezone.utc

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def main():
    old = json.loads((S / 'training-plan.seed3.v1.json').read_text())
    new = json.loads((D / 'training-plan.seed4.draft.json').read_text())
    assert sha(S / 'training-plan.seed3.v1.json') == '8b3b0f25bb22ecab73857aed7f75d8afae7c92c39e51c9a76b7189a9985a17a1'
    recipe = module('independent_recipe', S / 'training_recipe.py')
    f = module('independent_feasibility', D / 'feasibility.py')
    assert len(new['runs']) == 8
    for name, job in new['runs'].items():
        prior = old['runs'][name.replace('_s4_', '_s3_')]
        recipe.verify_job(job, name)
        args = job['parsed_args']
        changed = {k for k in args if args[k] != prior['parsed_args'][k]}
        assert changed == ({'name', 'seed'} if job['stage'] == 'parent' else {'name', 'seed', 'init'})
        for key in ['name', 'init']:
            if key in changed:
                assert args[key] == prior['parsed_args'][key].replace('_s3_', '_s4_')
        assert args['seed'] == job['seed'] == 4
        assert job['allocation_gate'] == 'only_after_both_seed3_WP_gates_PASS'
    for path, digest in new['sources'].items():
        assert sha(ROOT / path) == digest, path
    for key in ['limits', 'cutoffs', 'approved_scalar_fit', 'historical_input_identity',
                'incumbent_checkpoints', 'quality_contract', 'native_quality_contract',
                'input_identity_protocol', 'derived_allocation_protocol']:
        assert old[key] == new[key], key
    assert len(new['sources']) == len(old['sources']) + 2
    assert all(new['sources'][k] == v for k, v in old['sources'].items())
    for a, b in zip(old['lanes'], new['lanes']):
        assert {k: v for k, v in a.items() if k != 'runs'} == {k: v for k, v in b.items() if k != 'runs'}
        assert b['runs'] == [n.replace('_s3_', '_s4_') for n in a['runs']]

    # Crossed slow folds ensure no favorable-fold timing substitution.
    records = []
    def add(label, seconds):
        records.append(dict(label=label, seconds=seconds, exit_code=0, failure=None))
    add('inputs-before', 11)
    add('inputs-after', 13)
    values = [[101, 19, 211, 23, 3, 5, 7, 11, 17, 29],
              [97, 31, 223, 13, 7, 3, 5, 2, 23, 19]]
    lane_sums = []
    for fold, v in enumerate(values):
        names = [f'on_plectl_s3_f{fold}_parent', f'on_plectl_s3_f{fold}',
                 f'on_ple_s3_f{fold}_parent', f'on_ple_s3_f{fold}']
        for i, name in enumerate(names):
            add(name, v[i]); add(name + '-check', v[i + 4])
        add(names[1] + '-OOF', v[8]); add(names[3] + '-OOF', v[9])
        lane_sums.append(sum(v))
    c = old['cost']['control_median_seconds']; p = old['cost']['ple_median_seconds']
    launch = datetime.datetime(2026, 10, 1, 5, tzinfo=UTC)
    result = f.calculation(records, launch, c, p)
    assert result['PASS']
    expected = {'fixed_remaining_seconds': 1.25 * 15000 * (c + 2 * p) + 5400,
                'actual_remaining_seconds': 1.25 * (101 + 2 * 223) + 5400,
                'projected_PLE_parent_delay_seconds': 1.25 * (11 + 101 + 31 + 7 + 5),
                'projected_screen_seconds': 1.25 * (11 + max(lane_sums) + 13)}
    assert all(result[k] == v for k, v in expected.items())
    mapping = {'fixed_cost': 'fixed_remaining_cost', 'actual_cost': 'actual_remaining_parent_cost',
               'all_parents_start': 'all_parents_projected_start_before06',
               'screen_complete': 'screen_projected_complete_before_reserve'}
    cases = ['crossed-fold maxima and full-lane sums']
    for constraint, gate in mapping.items():
        boundary = datetime.datetime.fromisoformat(result['latest_start_by_constraint'][constraint])
        earlier = f.calculation(records, boundary - datetime.timedelta(microseconds=2), c, p)
        later = f.calculation(records, boundary + datetime.timedelta(microseconds=2), c, p)
        assert earlier['gates'][gate] and not later['gates'][gate], constraint
        cases.append(constraint + ' ±2µs boundary')
    for mutation in ['failed', 'duplicate', 'missing', 'nonpositive']:
        bad = copy.deepcopy(records)
        if mutation == 'failed': bad[2]['exit_code'] = 1
        if mutation == 'duplicate': bad.append(copy.deepcopy(bad[2]))
        if mutation == 'missing': bad.pop(2)
        if mutation == 'nonpositive': bad[2]['seconds'] = 0
        try:
            f.calculation(bad, launch, c, p)
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError('accepted ' + mutation)
        cases.append('reject ' + mutation)
    output = {'status': 'PASS_SOURCE_RECIPES_AND_GENERATED_FEASIBILITY_ONLY',
              'UTC': datetime.datetime.now(UTC).isoformat(), 'checker_sha256': sha(__file__),
              'draft_plan_sha256': sha(D / 'training-plan.seed4.draft.json'),
              'helper_sha256': sha(D / 'feasibility.py'), 'driver_sha256': sha(D / 'run_training.py'),
              'source_hashes_verified': len(new['sources']), 'exact_parsed_jobs': 8,
              'recipe_changes': ['seed 3 to 4', 'names and own-parent paths only'],
              'generated_checks': cases, 'independent_generated_arithmetic': expected,
              'empirical_receipts_or_data_read': False, 'model_forward': False,
              'GPU': False, 'WP_queries': 0,
              'limitation': 'Draft preparation only. No future allocation or completed seed3 evidence checked.'}
    path = P / 'critic/scalar-replication-generated-review.json'
    assert not path.exists()
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))

if __name__ == '__main__':
    main()
