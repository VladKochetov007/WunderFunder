"""Source-only seed4 derivative; original103-argument AST parser, no Torch/data reads."""
from pathlib import Path
import hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=HERE.parent/'student'
spec=importlib.util.spec_from_file_location('accepted_student_ast_parser',BASE/'build_plan.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
parser_from_source=base.parser_from_source
resolve=base.resolve
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    accepted=json.loads((BASE/'commands.json').read_text());parser=parser_from_source();jobs=[]
    for old in accepted['jobs']:
        name=old['name'].replace('_s3','_s4')
        argv=list(old['argv']);argv[argv.index('--name')+1]=name;argv[argv.index('--seed')+1]='4'
        if old['stage']=='H2':argv[argv.index('--init')+1]=old['resolved_args']['init'].replace('_s3','_s4')
        args=resolve(parser,argv[2:]);assert len(args)==103
        diff={k:[v,args[k]] for k,v in old['resolved_args'].items() if v!=args[k]}
        assert set(diff)==({'name','seed'} if old['stage']=='parent' else {'name','seed','init'})
        job={'name':name,'stage':old['stage'],'fold':old['fold'],'argv':argv,'resolved_args':args,'endpoint':old['endpoint'].replace('_s3','_s4'),'wall_limit_seconds':old['wall_limit_seconds'],'accepted_seed3_job':old['name'],'exact_argument_changes':diff}
        if old['stage']=='H2':
            job['OOF_argv']=[x.replace('/student/predict_fixed.py','/student-seed4/predict_fixed.py').replace('_s3','_s4') for x in old['OOF_argv']]
            job['OOF_wall_limit_seconds']=old['OOF_wall_limit_seconds']
        jobs.append(job)
    inv=BASE/'inventory/receipt.json';inventory=json.loads(inv.read_text())
    stats={}
    def collect(x):
        if isinstance(x,dict):
            if all(k in x for k in ('path','size','mtime_ns','inode')):stats[x['path']]={k:x[k] for k in ('size','mtime_ns','inode')}
            for v in x.values():collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    for k in ('teacher_caches','row_maps','chain_folds','propensity_models','propensity_caches','datasets'):collect(inventory[k])
    sources={k:v for k,v in accepted['source_sha256'].items()}
    for p in (Path(__file__),HERE/'run_plan.py',HERE/'predict_fixed.py',BASE/'commands.json',inv):sources[str(p.relative_to(ROOT))]=sha(p)
    result={'status':'SOURCE_PREPARATION_ONLY_NO_ALLOCATION','execution_authorized':False,'source_sha256':sources,'jobs':jobs,'environment':accepted['environment'],'input_inventory':str(inv.relative_to(ROOT)),'input_inventory_sha256':sha(inv),'input_stat_bindings':stats,'input_lineage_limitation':'Reuse accepted finite/fullhash inventory; prospective fresh stat checks only, no redundant fullcache scan and no new teacher-epoch claim.','budget':{'fits':4,'H2_OOF_jobs':2,'parent_OOF_jobs':0,'retries':0,'metrics_authorized':False},'proposed_resource_limits':{'GPU_id':7,'allocator_fraction':.25,'device_total_envelope_MiB':5632,'CPU_ids':[22,23,24,25,28,32],'CPUQuota_percent':600,'MemoryMax_bytes':20*2**30,'MemorySwapMax':0,'whole_stage_seconds_max':6000},'comparisons':['double_shallow-current_student4','double_shallow-strong4','single_seed4-current_student4'],'scope':'No execution allocation; exactly seed4 optimizer replication on existingfolds; no fresh data confirmation, no seed/epoch/weight search.'}
    with (HERE/'commands.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS_SOURCE_ONLY_4_JOBS_103_ARGS','commands_sha256':sha(HERE/'commands.json'),'changes':[j['exact_argument_changes'] for j in jobs]},indent=2))
if __name__=='__main__':main()
