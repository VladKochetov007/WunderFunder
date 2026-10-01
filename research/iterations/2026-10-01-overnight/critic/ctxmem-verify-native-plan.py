"""Independent runtime-preparation receipt; hashes source/binaries, never raw fixture."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent/'ctxmem'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
plan_path=ACTOR/'native-runtime-plan.json';expected='6a27499f415f978e453586784f6423c19cbd5f5b392e4de623eb723d013d5f71';assert sha(plan_path)==expected
p=json.loads(plan_path.read_text());assert p['revision']==2
for name,digest in p['files_sha256'].items():assert sha(ACTOR/name)==digest,name
assert sha(ACTOR/p['native_module'])==p['native_module_sha256']
assert sha(ACTOR/'native_ctx/build/v3/wnn_cpp.cpython-313-x86_64-linux-gnu.so')==p['native_author_module_sha256']
receipts={}
for name in ['ctxmem-projection-correction-review.json','ctxmem-independent-state-python.json','ctxmem-independent-state-native.json','ctxmem-independent-cuda-result.json','ctxmem-independent-full-baseline-cp311.json','ctxmem-independent-full-native.json']:
    result=json.loads((HERE/name).read_text());assert result['status']=='PASS';receipts[name]=sha(HERE/name)
r={'status':'NATIVE_PREPARATION_PASS_AWAIT_ROOT_RUNTIME_ALLOCATION','plan_sha256':expected,'verified_source_files':len(p['files_sha256']),'scorer_native_sha256':p['native_module_sha256'],'author_native_sha256':p['native_author_module_sha256'],'independent_receipts':receipts,'review':{'frozen_three_members_and_original_arrays':True,'only_first_two_gain_fixed_projection_and_nonzero_synthetic_extra':True,'ordinary_FP16_extra_weight_policy_FP32_projection':True,'same_wrapper_and_complete_callbacks':True,'first_three_clean_AB_BA_pairs_max_five_attempts':True,'no_clean_unfavorable_pair_dropped':True,'quiet_SMT_precheck_and_summed_timed_region_activity':True,'one_CPU_and_thread_source_package_native_hash_guards':True,'required_zero_extra_archived_baseline_parity':True,'primary_median_paired_ratio_at_most_1_03':True,'no_forecasting_or_metric_in_this_allocation':True},'qualification_limits':['Scorer CPython3.11 complete-callback parity and runtime remain unexecuted by critic.','Independent nonzero full-model comparison uses same-source CPython3.13 extension; zero-extra frozen baseline was separately run under actual scorer CPython3.11.','Overhead PASS alone does not authorize training or prove predictive gain.','Trainer commands/transform/checkpoints/provenance need a separate freeze after the runtime gate.','Final complete ZIP still requires unchanged 50-minute absolute runtime gate and all declared quality gates.'],'empirical_fixture_read':False,'checkpoint_reads':0,'targets_read':False,'forecast_fits':0,'WP_queries':0}
dest=HERE/'ctxmem-native-preparation-review.json';assert not dest.exists();dest.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
