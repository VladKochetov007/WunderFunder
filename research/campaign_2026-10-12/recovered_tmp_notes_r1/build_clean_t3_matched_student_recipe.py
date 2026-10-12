import hashlib, json
from pathlib import Path

root = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
wt = Path('/home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile')
base_rel = Path('research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011')
out_rel = base_rel / 'clean_t3_f0_matched_student_r5_cache_r1'
out_dir = wt / out_rel
out_dir.mkdir(parents=True, exist_ok=True)

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def jread(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

recipe_path = root / base_rel / 'clean_t3_fold0_h390_r1/CLEANFOLD_T3_F0_RECIPE_R1.json'
parents_path = root / base_rel / 'clean_t3_fold0_h390_r1/STUDENT_PARENT_ASSET_R1.json'
stages_path = root / base_rel / 'clean_t3_fold0_h390_r1/STUDENT_STAGE_TEMPLATES_R1.json'
preflight_path = root / base_rel / 'clean_t3_local_s3_fit_train_preflight_r1/LOCAL_S3_FIT_TRAIN_PREFLIGHT_R1.json'
grade_path = root / base_rel / 'clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json'
f1_prov_path = root / 'artifacts/runs/vt_f1_dsq_dm_t3_a9_e6/provenance.json'
f1_cache_meta_path = root / base_rel / 'h391_f1_h384_t3_student_r1/actual_producer_receipts_r1/dc_98411230_metadata.json'
split_path = root / 'research/campaign_2026-09-29/artifacts/splits/fold0.json'
r3_root = root / base_rel / 'clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot'
r3_manifest = r3_root / 'SOURCE_MANIFEST_H390.json'
r3_delta_path = root / base_rel / 'clean_t3_fold0_h390_r3_teacher_cohort/SOURCE_DELTA_R3.json'
builder = r3_root / 'scripts/build_cleanfold_t3_cache.py'
trainer = r3_root / 'scripts/train_rnn.py'
wrapper = r3_root / 'scripts/cleanfold_exec.py'
prop = root / 'artifacts/runs/c20260929_prop_f0/model.pt'

recipe, parent_asset, stage_templates = map(jread, (recipe_path, parents_path, stages_path))
preflight, grade, f1_prov = map(jread, (preflight_path, grade_path, f1_prov_path))
inputs = {
    'f0_recipe': recipe_path,
    'parent_asset': parents_path,
    'stage_templates': stages_path,
    'fit_train_preflight': preflight_path,
    'equal_trio_grade': grade_path,
    'f1_student_provenance': f1_prov_path,
    'f1_cache_metadata': f1_cache_meta_path,
    'fold0_split': split_path,
    'r3_source_manifest': r3_manifest,
    'r3_source_delta': r3_delta_path,
    'r3_cache_builder': builder,
    'r3_trainer': trainer,
    'r3_cleanfold_exec': wrapper,
    'f0_propensity_checkpoint': prop,
}

f1_args = f1_prov['args']
fit_id = f1_prov['fit_membership_identity']
full_valid_names = {}
for seed in (3, 4, 5):
    teacher = grade['endpoints']['teachers'][f's{seed}']
    run = teacher['run']
    stem = f'pred_{run}_local_full_valid_r1_valid'
    array_path = root / 'cache' / f'{stem}.npy'
    sidecar_path = root / 'cache' / f'{stem}.json'
    sidecar = jread(sidecar_path)
    full_valid_names[str(seed)] = {
        'run': run,
        'epoch': teacher['checkpoint_epoch'],
        'checkpoint_sha256': teacher['checkpoint_sha256'],
        'provenance_sha256': teacher['provenance_sha256'],
        'array_path': str(array_path),
        'array_bytes': array_path.stat().st_size,
        'sidecar_path': str(sidecar_path),
        'sidecar_sha256': sha(sidecar_path),
        'sidecar_prediction_sha256': sidecar['prediction_sha256'],
        'sidecar_campaign_fit_scope': sidecar.get('campaign_fit_scope'),
        'full_valid_row_count': len(sidecar['row_groups']),
        'expected_full_valid_shape': [1873, 20000, 2],
        'expected_full_valid_dtype': 'float32',
        'use_for_fit_valid': 'Only after an independently validated row-group map projects to the 1,405 F0 FIT_VALID groups; exclude all 468 evaluation groups. Do not rerun this full-valid export.',
    }

full_cache_root = 'cache/clean_t3_f0_h390_t3r5_a9_r1'
assert not (root / full_cache_root).exists(), 'refusing a cache-root collision'

run_base = {
    'control': 'clean_t3_f0_control_s3_a9_t3r5_r1',
    'treatment': 'clean_t3_f0_treatment_s3_a9_t3r5_r1',
}
asset_parent_by_arm = {row['arm']: row for row in parent_asset['parents']}
recipe_parent_by_arm = {row['arm']: row for row in recipe['student_parent_arms']}
parent_by_arm = {arm: {**asset_parent_by_arm[arm], **recipe_parent_by_arm[arm]} for arm in ('control', 'treatment')}

# Preserve the existing reviewed continuation template, changing only source-root,
# cache-root, distinct run IDs, and local continuation-state paths. Explicitly set
# behaviorally relevant parser defaults to make the pending recipe self-contained.
stages = []
for old in stage_templates['stages']:
    arm = old['arm']
    part = old['part']
    argv = list(old['argv'])
    argv[1] = str(wrapper)
    argv[argv.index('--name') + 1] = f"{run_base[arm]}_p{part}"
    argv[argv.index('--distill-cache-root') + 1] = full_cache_root
    argv[argv.index('--init') + 1] = f"{parent_by_arm[arm]['run']}@ep{parent_by_arm[arm]['epoch']}"
    if '--resume-state' in argv:
        previous_part = part - 1
        argv[argv.index('--resume-state') + 1] = f"artifacts/runs/{run_base[arm]}_p{previous_part}/training_state.pt"
    # Pin defaults from the completed strong-student provenance and reviewed R3 parser.
    explicit = [
        '--distill-loss', 'mse', '--distill-channel-weights', '1,1',
        '--distill-denominator', 'chunk', '--distill-alpha-tau', '32',
        '--distill-boot-h', '0', '--distill-reanchor', '0',
        '--distill-family-weight', '0', '--label-weight-power', '1',
        '--label-weight-floor', '0',
    ]
    argv += explicit
    stages.append({
        'arm': arm,
        'part': part,
        'name': f"{run_base[arm]}_p{part}",
        'stop_after_epoch_inclusive': old['stop_after_epoch_inclusive'],
        'schedule_continuity': old['schedule_continuity'],
        'command_argv': argv,
        'expected_outputs': [
            f"artifacts/runs/{run_base[arm]}_p{part}/training_state.pt",
            f"artifacts/runs/{run_base[arm]}_p{part}/provenance.json",
            f"artifacts/runs/{run_base[arm]}_p{part}/history.json",
        ],
        'requires_cache_binding_and_root_release': True,
    })

final_exports = []
for arm, base in run_base.items():
    final_run = f'{base}_p3'
    final_exports.append({
        'arm': arm,
        'run': final_run,
        'epoch': 5,
        'command_argv': [
            str(wrapper), 'scripts/predict_run.py', final_run, 'valid',
            '--epoch', '5', '--campaign-split-manifest',
            'research/campaign_2026-09-29/artifacts/splits/fold0.json',
            '--device', 'cuda', '--campaign-full-valid-feature-only',
            '--inference-precision', 'fp32',
        ],
        'expected_npy_shape': [1873, 20000, 2],
        'expected_dtype': 'float32',
        'valid_targets_row_order': True,
        'evaluation_scope': 'The fixed F0 source-qualified evaluation mask selects 468 groups; this is a repeatedly exposed development panel, not a fresh holdout.',
    })

teacher_train = {}
split_doc = jread(split_path)
expected_train_set = set(map(int, split_doc['fit']['train']))
expected_valid_set = set(map(int, split_doc['fit']['valid']))
for seed in (3,4,5):
    run = grade['endpoints']['teachers'][f's{seed}']['run']
    teacher_prov_path = root / 'artifacts' / 'runs' / run / 'provenance.json'
    teacher_prov = jread(teacher_prov_path)
    if sha(teacher_prov_path) != grade['endpoints']['teachers'][f's{seed}']['provenance_sha256']:
        raise ValueError(f'{run} provenance differs from the bound trio grade')
    fit_rows = teacher_prov['fit_row_groups']
    train_set_equal = set(map(int, fit_rows['train'])) == expected_train_set
    valid_set_equal = set(map(int, fit_rows['valid'])) == expected_valid_set
    if not train_set_equal or not valid_set_equal:
        raise ValueError(f'{run} fitting group set differs from fold 0')
    if seed == 3:
        train_preflight = preflight['planned_export']
        train_path = train_preflight['output_npy']
        fit_train = {
            'status': 'PENDING_ROOT_ADMISSION; the 16-sequence profile is not a cache input',
            'proposed_full_export_output_npy': train_path,
            'expected_shape': [10535, 20000, 2],
            'expected_dtype': 'float32',
            'expected_file_bytes': train_preflight['expected_npy_bytes_including_header'],
            'preflight_sha256': sha(preflight_path),
        }
    else:
        fit_train = {
            'status': 'PENDING_ACTUAL_PRODUCER_EXPORT_BINDING',
            'proposed_full_export_output_npy': None,
            'expected_shape': [10535, 20000, 2],
            'expected_dtype': 'float32',
            'expected_file_bytes': 10535*20000*2*4+128,
        }
    teacher_train[str(seed)] = {
        'run': run,
        'epoch': 2,
        'checkpoint_sha256': grade['endpoints']['teachers'][f's{seed}']['checkpoint_sha256'],
        'provenance_sha256': grade['endpoints']['teachers'][f's{seed}']['provenance_sha256'],
        'fit_membership_identity': teacher_prov['fit_membership_identity'],
        'fit_membership_set_equal_to_f0_split': {'train': train_set_equal, 'valid': valid_set_equal},
        'fit_train_prediction': fit_train,
        'fit_valid_source': full_valid_names[str(seed)],
    }

reference_sources = recipe['f1_homogeneous_reference']['source_coefficients']
reference_strong = [s for s in reference_sources if s['family'] == 'strong']
assert len(reference_strong) == 3
assert all(float(s['coefficient']) == 0.1 and s['epoch'] == 2 for s in reference_strong)
assert recipe['cache_target']['homogeneous_scale'] == 0.8125
assert recipe['student_recipe']['distill_alpha'] == f1_args['distill_alpha'] == 0.9

contract = {
    'schema': 'wnn.clean-t3-f0-matched-student-recipe-pending-cache.r1',
    'status': 'CACHE_BINDING_PENDING_ROOT_ADMISSION; NO_PROFILE_OR_FIT_LAUNCHED',
    'authorization': {'fit_launched': False, 'teacher_export_launched': False, 'cache_builder_launched': False,
                      'full_valid_teacher_exports_repeated': False, 'root_release_required': True},
    'scientific_question': 'Does the completed strong-T3 homogeneous distillation recipe improve a clean F0 H256 student under the same predeclared alpha, F0 initializer-parent pair, seed, and exact fold-0 fitting membership?',
    'reference_completed_fullfold_student': {
        'run': 'vt_f1_dsq_dm_t3_a9_e6',
        'provenance_path': str(f1_prov_path),
        'provenance_sha256': sha(f1_prov_path),
        'checkpoint_epoch': 5,
        'checkpoint_sha256': f1_prov['checkpoint_sha256']['model_ep5.pt'],
        'reference_cache_metadata_path': str(f1_cache_meta_path),
        'reference_cache_metadata_sha256': sha(f1_cache_meta_path),
        'recipe_reference': 'Completed F1 full-fit provenance; this is a recipe/target-semantics reference only. No F1 weights or fitting groups enter the clean F0 candidate.',
        'key_effective_args': {key: f1_args.get(key) for key in [
            'seed','hidden','layers','arch','cell','dropout','batch','chunk','epochs','checkpoint_epoch',
            'checkpoint_policy','no_eval_during_fit','train_source','train_seqs','valid_seqs','init',
            'inherit_init_preprocessing','loss','loss_norm','distill_alpha','distill_alpha_tau','distill_boot_h',
            'distill_denominator','distill_family_weight','distill_family_head','distill_reference_mass',
            'label_weight_power','label_weight_floor','lr','wd','optim','sched','precision','prop_power','prop_eps',
            'valid_true_mask','aux_struct','aux_next','since_weight','since_floor','fast','sync_free','fused_adam',
            'pin_memory','prop_cache']},
        'fit_membership_identity': f1_prov['fit_membership_identity'],
        'not_reused': ['F1 parent checkpoint weights', 'F1 fold membership', 'F1 propensity checkpoint', 'F1 validation scores as a clean F0 estimate'],
    },
    'clean_f0_lineage': {
        'fold': 0,
        'split_manifest_path': str(split_path.relative_to(root)),
        'split_manifest_sha256': sha(split_path),
        'actual_selector_preflight_path': str(preflight_path),
        'actual_selector_preflight_sha256': sha(preflight_path),
        'fit_train_groups': 10535,
        'fit_valid_groups': 1405,
        'total_fit_groups': 11940,
        'evaluation_valid_groups_excluded': 468,
        'evaluation_exposure': 'F0 is repeatedly exposed and used for local development; not a pristine holdout.',
        'train_membership_hashes': {
            'source_qualified_split_ordered_int64_le_sha256': preflight['actual_consumer_boundary']['selected_fit_train_groups_ordered_int64_le_sha256'],
            'seed3_fit_order_sha256': teacher_train['3']['fit_membership_identity']['train']['ordered_int64_le_sha256'],
            'verified_semantics': 'All three seed-specific fit groups are the exact F0 set. Preflight canonical split order and trainer fit-loader order differ; the training order also varies by seed. Cache building and training must map by row-group ID, not equate their order hashes.',
            'per_seed_fit_loader_order_sha256': {seed: row['fit_membership_identity']['train']['ordered_int64_le_sha256'] for seed,row in teacher_train.items()},
        },
        'fit_valid_membership': {
            'source_qualified_split_ordered_int64_le_sha256': recipe['ordered_fit_membership_sha256']['valid'],
            'per_seed_trainer_ordered_int64_le_sha256': {seed: row['fit_membership_identity']['valid']['ordered_int64_le_sha256'] for seed,row in teacher_train.items()},
        },
        'student_fit_args_keep_reference_protocol': {'train_source':'train+valid','train_seqs':f1_args['train_seqs'],'valid_seqs':f1_args['valid_seqs'],'valid_fold':0,'no_eval_during_fit':True,'epochs':6,'checkpoint_epoch':5},
        'f0_propensity_checkpoint_sha256': recipe['student_recipe']['propensity_checkpoint_sha256'],
        'source_qualified_fit_evaluation_overlap': 0,
    },
    'student_recipe': {
        'arms': [
            {k: v for k,v in parent_by_arm['control'].items()},
            {k: v for k,v in parent_by_arm['treatment'].items()},
        ],
        'student_seed': 3,
        'architecture': 'GRU(256, layers=2)',
        'epochs': 6,
        'checkpoint_policy': 'fixed',
        'checkpoint_epoch': 5,
        'optimizer': 'AdamW',
        'schedule': 'one OneCycle schedule across six epochs; staged resumes continue saved optimizer/scheduler/scaler/model/loader/RNG state',
        'batch': 16,
        'chunk': 2000,
        'precision': 'fp16',
        'loss': 'wMSE/chunk',
        'distillation': {
            'cache_variant': 'homogeneous',
            'alpha': 0.9,
            'alpha_far': None,
            'alpha_tau': 32.0,
            'bootstrap_horizon': 0,
            'reanchor': 0.0,
            'loss': 'MSE',
            'channel_weights': [1.0,1.0],
            'denominator': 'chunk',
            'label_weight_power': 1.0,
            'label_weight_floor': 0.0,
            'family_head': False,
            'family_weight': 0.0,
            'homogeneous_target_formula': 'T = float32(0.8125 * float32_mean([teacher_s3_epoch2, teacher_s4_epoch2, teacher_s5_epoch2])); student target = 0.1*y + 0.9*T.',
            'teacher_target_clipping': 'none during cache construction',
        },
        'aux_next': 0.3,
        'aux_struct': 0.3,
        'lr': 0.0001,
        'weight_decay': 0.01,
        'propensity': 'c20260929_prop_f0',
        'propensity_power': 2.0,
        'propensity_epsilon': 0.03,
        'valid_true_mask': 0.03,
        'since_weight': 0.0,
        'since_floor': 0.25,
        'inherit_init_preprocessing': False,
        'fixed_fit_membership': 'same split and identical membership for both parent arms; 10,535 train + 1,405 fit-valid; exclude 468 evaluation groups',
        'prior_f0_student_control_not_repeated': 'oldz4_h230_student_f0_s3_r3 is retained as an existing comparator; this recipe does not rerun it.',
    },
    'teacher_sources': teacher_train,
    'target_cache': {
        'status': 'PENDING_ACTUAL_FIT_TRAIN_EXPORT_AND_ROW_GROUP_BOUND_CACHE',
        'unique_cache_root': full_cache_root,
        'outputs': {
            'metadata': f'{full_cache_root}/metadata.json',
            'main_homogeneous_train': {'path':f'{full_cache_root}/main_homogeneous_train.npy','shape':[10535,20000,2],'dtype':'float32','expected_bytes':1685600128},
            'main_homogeneous_valid': {'path':f'{full_cache_root}/main_homogeneous_valid.npy','shape':[1405,20000,2],'dtype':'float32','expected_bytes':224800128},
        },
        'family_source_coefficients': [{'run':x['run'],'epoch':2,'coefficient':0.1} for x in reference_strong],
        'family_coefficient_subtotal':0.3,
        'homogeneous_scale':0.8125,
        'operation_order':'float32 mean across current own F0 teacher seed 3/4/5 epoch-2 predictions; then float32 multiply by 0.8125; no cache clipping.',
        'fit_train_input_status':'All three complete [10535,20000,2] FP32 prediction artifacts remain unbound. The existing 16-group s3 profile is prohibited as a student-cache input. The s3 full export successor is pending root admission.',
        'fit_valid_input_status':'The three existing full-valid feature-only arrays are available as projection sources. Bind exact producer checkpoint/sidecar hashes, map their 1,873 row groups, select only exact 1,405 FIT_VALID groups, and omit all 468 evaluation groups. Do not rerun a full-valid export.',
        'builder': {
            'frozen_path': str(builder),
            'frozen_sha256': sha(builder),
            'must_not_run_unchanged': True,
            'known_mismatches': [
                'TEACHER_RUNS hardcodes old _r1 teacher IDs; current clean F0 producers are _r5.',
                'Input filenames are hardcoded as pred_{run}_{source}.npy; current run-scoped exports use _local_fit_train_r1_train.npy and _local_full_valid_r1_valid.npy.',
                'Current full-valid sidecars say full-valid-feature-only and contain 1,873 rows, while the target cache must contain only the 1,405 fit-valid rows.',
                'The builder requires the seed-3 randomized training-order hash for all teachers; current s4/s5 have different randomized order hashes despite exact equal membership sets.',
            ],
            'smallest_required_correction':'Before cache build, use a separately identified manifest-driven adapter to bind the actual producer arrays/sidecars and map/filter by exact source-qualified F0 row-group membership; preserve this frozen builder/source snapshot.',
        },
        'cache_acceptance_gates':['hash-bound current teacher checkpoint and provenance','exact model epoch 2 / split F0 / propensity checkpoint','all arrays finite FP32 with expected shapes','each seed-specific teacher train row group set equals exact 10,535 fit-train groups; bind its own ordered hash','valid projection emits exact 1,405 fit-valid groups only and zero evaluation groups','metadata binds coefficients, source order, float32 operation order, input/output hashes','actual producer fixture consumed by the validator'],
    },
    'source': {
        'source_snapshot_root':str(r3_root),
        'source_manifest_path':str(r3_manifest),
        'source_manifest_sha256':sha(r3_manifest),
        'source_delta_path':str(r3_delta_path),
        'source_delta_sha256':sha(r3_delta_path),
        'teacher_student_trainer_path':str(trainer),
        'teacher_student_trainer_sha256':sha(trainer),
        'cleanfold_exec_path':str(wrapper),
        'cleanfold_exec_sha256':sha(wrapper),
        'source_delta_semantics':'The R3 trainer change from the prior clean F0 student snapshot adds opt-in actual update-counter telemetry only; distillation/cache semantics are unchanged. Keep source identity separately pinned for the new fit.',
        'prior_distillation_implementation_hashes':{
            'distillation_py':sha(r3_root/'src/wnn/distillation.py'),
            'distillation_cache_py':sha(r3_root/'src/wnn/distillation_cache.py'),
            'final_distillation_cache_py':sha(r3_root/'src/wnn/final_distillation_cache.py'),
            'kd_loss_py':sha(r3_root/'src/wnn/kd_loss.py'),
        },
    },
    'stages': stages,
    'final_student_exports': final_exports,
    'no_launch_gates':[
        'Do not fit until the exact R8 or local teacher cache receipt is available and verified; do not duplicate six caches if R8 delivers them.',
        'Materialize the exact F0 student parent checkpoints/provenance under the frozen runner’s expected artifacts/runs/<run>/ paths and verify SHA256 immediately before launch; control source is local, treatment staged parent requires explicit path verification.',
        'Run/record actual H256 B16 C2000 device/resource profile and schedule-resume checks before any six-epoch fit.',
        'Root must recheck aggregate resource admission and explicitly release the student fit; this recipe is not launch authorization.',
        'Bind exact cache metadata and array hashes, split, source manifest, data/content identities, parent hashes, and new run names into fresh conditional fit contracts.',
    ],
    'reviewed_artifact_hashes': {k:sha(v) for k,v in inputs.items()},
    'distinct_from_existing_profile_commit':'The profile lineage records only a 16-group s3 resource profile. It is not a cache producer or a training permission.',
}

contract_path = out_dir/'STUDENT_RECIPE_PENDING_CACHE_R1.json'
contract_path.write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')

notes = f'''# Matched clean F0 strong-T3 student recipe (cache binding pending)\n\nStatus: metadata/command preparation only. No cache build, fit, or student prediction has been launched. The full-fold F1 strong student `vt_f1_dsq_dm_t3_a9_e6` is already completed and is used only as the recipe/target-semantics reference; its weights and fold membership are not reused. The existing F0 `oldz4_h230_student_f0_s3_r3` remains a comparator and is not rerun.\n\n## Findings\n\nThe intended target is the previously selected homogeneous strong-family target: `T = float32(0.8125 * mean_float32(F0 clean-T3 seeds 3,4,5 epoch-2 predictions))`, then `0.1*y + 0.9*T` during student fitting. The student is a seed-3 GRU(256, 2 layers), six epochs, fixed epoch-5 checkpoint, FP16, batch 16, chunk 2,000, wMSE/chunk, AdamW `1e-4`/`0.01`, OneCycle, F0 propensity power 2/epsilon .03, and the existing `.3` structure/next auxiliaries.\n\nThe paired F0 initializers are the exact own control `c20261003_combo_s3_f0@ep3` (checkpoint SHA `{parent_by_arm['control']['checkpoint_sha256']}`) and own treatment `kg_h382r3_f0_muonall_prop0_e4_s3@ep3` (checkpoint SHA `{parent_by_arm['treatment']['checkpoint_sha256']}`). Both arms use seed 3 and the same F0 split. The training pool is 10,535 FIT_TRAIN + 1,405 FIT_VALID = 11,940 groups; all 468 F0 evaluation groups are excluded. F0 is repeatedly exposed development evidence, not a fresh holdout.\n\nCache binding is the remaining blocker. The frozen R3 cache builder SHA `{sha(builder)}` binds old `_r1` teacher IDs and conventional prediction filenames, while actual current teachers are `_r5`; their full-valid arrays use `_local_full_valid_r1_valid.npy` and sidecars with 1,873 rows. The final student target cache must project only the exact 1,405 FIT_VALID rows. Complete FIT_TRAIN arrays for all three current teachers still need a root-released producer path. The s3 16-group profile output is not a cache input. Use a separately reviewed, manifest-driven adapter; leave the frozen builder untouched.\n\nThe train-selector canonical-order hash is `__TRAIN_SELECTOR_HASH__`; training provenance records per-seed randomized fit-loader order hashes s3=`__TRAIN_S3_HASH__`, s4=`__TRAIN_S4_HASH__`, s5=`__TRAIN_S5_HASH__`. All three group sets equal the same F0 membership. FIT_VALID order hashes all match at `__VALID_HASH__`. A cache adapter must join by row-group ID and preserve the training loader's alignment; never treat distinct ordered hashes as interchangeable.\n\n## Recipe checks\n\n| Dimension | Completed F1 reference | Clean F0 candidate | Assessment |\n|---|---|---|---|\n| Student | H256 GRU, seed 3, epochs 0–5, checkpoint 5 | Same | Required match |\n| Distillation | alpha .9, homogeneous strong target | Same alpha and target transform with own F0 clean-T3 seeds 3/4/5 | Required recipe match; regenerated fold-local weights |\n| Init parents | F1 own fold-1 combo parent | Same two predeclared F0 own parent arms in both paired fits | Intentional fold/parent adaptation; never load F1 weights |\n| Fit rows | 10,533 train + 1,416 fit-valid | 10,535 train + 1,405 fit-valid; 468 eval excluded | Correct fold-local membership, not byte-identical to F1 |\n| Propensity | fold 1, power 2, epsilon .03 | fold 0, power 2, epsilon .03 | Intentional own-fold adaptation |\n| Cache producer binding | prior F1 actual metadata | current `_r5` predictions not yet bound | Pending; the existing R3 `_r1` builder cannot consume current filenames/sidecars unchanged |\n| Evaluation | exposed F1 fold-1 panel | repeatedly exposed F0 468-group panel | No pristine-holdout or public claim |\n\n## Exact source and commands reviewed\n\nPrior student provenance SHA `{sha(f1_prov_path)}` and F1 cache metadata SHA `{sha(f1_cache_meta_path)}`; F0 recipe SHA `{sha(recipe_path)}`; parent asset SHA `{sha(parents_path)}`; stage template SHA `{sha(stages_path)}`; current selector preflight SHA `{sha(preflight_path)}`; current trio-grade SHA `{sha(grade_path)}`. The student command vectors in the JSON preserve the reviewed three-stage six-epoch OneCycle schedule, point to the R3 frozen source snapshot, and change only run/cache IDs plus local resume-state paths. Relevant R3 parser defaults were checked in `scripts/train_rnn.py`: MSE, channel weights `1,1`, chunk denominator, alpha tau 32, bootstrap 0, reanchor 0, family loss 0, label weight power 1/floor 0. The R3 source delta documents update-counter telemetry only.\n\n## Remaining limitations\n\nNo actual cache metadata/array hash is available yet; no current local fit-train prediction exists for seeds 4/5 in this recipe bundle; the local treatment parent path must be reverified at launch; the current frozen cache builder requires a new, independently reviewed input binding/projection adapter; H256/B16/C2000 profile and continuation check are still required; no fit, grade, or submission decision is authorized by this artifact.\n'''
# Fill the order-hash facts with explicit values after constructing the narrative.
notes = notes.replace('__TRAIN_SELECTOR_HASH__', contract['clean_f0_lineage']['train_membership_hashes']['source_qualified_split_ordered_int64_le_sha256'])
notes = notes.replace('__TRAIN_S3_HASH__', teacher_train['3']['fit_membership_identity']['train']['ordered_int64_le_sha256'])
notes = notes.replace('__TRAIN_S4_HASH__', teacher_train['4']['fit_membership_identity']['train']['ordered_int64_le_sha256'])
notes = notes.replace('__TRAIN_S5_HASH__', teacher_train['5']['fit_membership_identity']['train']['ordered_int64_le_sha256'])
notes = notes.replace('__VALID_HASH__', recipe['ordered_fit_membership_sha256']['valid'])
(out_dir/'RECIPE_AUDIT_R1.md').write_text(notes)
print('wrote', contract_path)
print('contract sha', sha(contract_path))
print('stage count',len(stages),'teacher fullvalid seeds',len(full_valid_names))
print('new cache root absent',not (root/full_cache_root).exists())
