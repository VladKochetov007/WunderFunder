from pathlib import Path
import json,hashlib,base64,sys,shutil,time,gzip,subprocess,os
from collections import Counter
H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');src=H/'reports/full-primary-refits-r1';out=H/'reports/full-primary-final-delivery-r1';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
r=json.loads((src/'full_primary_refit_report.json').read_text());assert r['status']=='complete'
independent=json.loads((H/'reports/independent-full-primary-r1.json').read_text());assert independent['status']=='verified' and independent['report_sha256']==sha(src/'full_primary_refit_report.json')
copy_specs=[]
for p in src.iterdir():
 if p.is_file() and p.suffix!='.npy':copy_specs.append((p,Path(p.name)))
for d in ['member_receipts','process_receipts','input_receipts']:
 for p in (src/d).glob('*.json'):copy_specs.append((p,Path(d)/p.name))
for s,n in [('reports/independent-full-primary-r1.json','independent_primary_arithmetic.json'),('receipts/full-primary-checkpoints-release-r1.json','full_primary_weight_release.json'),('receipts/checkpoint-release-r1.json','auxiliary_weight_release.json'),('receipts/full-primary-refits-r1.child.json','collector_process_receipt.json'),('campaign-r2/budget.json','original_fit_profile_ledger.json'),('requests/verify-full-primary-independent-r1.py','verify_full_primary_independent.py'),('requests/verify-primary-independent-r1.py','verify_primary_independent_helpers.py')]:copy_specs.append((H/s,Path(n)))
compression=[]
for p,dest in copy_specs:
 target=out/dest;target.parent.mkdir(parents=True,exist_ok=True)
 if p.stat().st_size>1000000:
  target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(p.read_bytes(),mtime=0));compression.append({'path':str(target.relative_to(out)),'compressed_sha256':sha(target),'original_name':str(dest),'original_sha256':sha(p),'original_bytes':p.stat().st_size})
 else:shutil.copyfile(p,target)
write(out/'COMPRESSION_RECEIPT.json',compression)
sys.path.insert(0,str(H/'cpu-deps-r1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,2,figsize=(12,14));curves=[]
for row,member in enumerate(r['fit_statuses']):
 fit=member['fit_id'];p=H/'runs'/fit/'monitor_evaluations.jsonl';events=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];assert len(events)==member['config']['epochs']
 curves.append({'fit_id':fit,'native_journal_sha256':sha(p),'epochs':[{'epoch':x['epoch'],'metrics':x['metrics']} for x in events]})
 for col,key in enumerate(['loss','wp']):
  ax=axes[row,col]
  for cohort in ['train_eval','valid']:ax.plot([x['epoch'] for x in events],[x['metrics'][cohort+'/'+key] for x in events],marker='o',label=cohort)
  ax.set_title(fit+' '+key);ax.set_xlabel('completed epoch');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Full-primary refit trajectories: train diagnostic cohort=16/all rows; valid=1873/is_scored. Masks differ.',fontsize=10);fig.tight_layout(rect=(0,0,1,.97))
for ext in ['png','svg']:fig.savefig(out/('native_train_valid_curves.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig);write(out/'native_curve_values.json',curves)
proc=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if str(H) in cmd and int(p.name)!=os.getpid() and fields[0] not in ['Z','X']:proc.append({'pid':int(p.name),'state':fields[0],'pgid':int(fields[2]),'birth':fields[19],'cmdline':cmd})
 except (OSError,ValueError,IndexError):pass
owned_training=[x for x in proc if any(y in x['cmdline'] for y in ['train_sequence.py','train_stage','multistage_worker.py','report_multistage_primary_refits_worker.py','report_multistage_primary_ensemble_worker.py'])]
assert not owned_training,owned_training
nv=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
nv_rows=[]
for line in nv.splitlines():
 cols=[x.strip() for x in line.split(',')];pid=int(cols[1]);path=Path('/proc')/str(pid)/'cmdline';cmd=path.read_bytes().replace(bytes([0]),b' ').decode(errors='replace') if path.exists() else ''
 assert str(H) not in cmd and '/tabdata/' not in cmd
 nv_rows.append({'gpu_uuid':cols[0],'pid':pid,'process_name':cols[2],'used_memory':cols[3],'cmdline':cmd,'preserved':True})
b=json.loads((H/'campaign-r2/budget.json').read_text())
cleanup={'schema':'wnn.multistage-final-gpu-release.v1','status':'verified_no_owned_gpu_processes','created_at_epoch':time.time(),'runtime_root':str(H),'scientific_attempts':len(b['fit_attempts']),'configured_final_model_artifacts':39,'partial_model_artifacts':1,'profiles':len(b['profile_attempts']),'original_ledger_status_counts':dict(Counter(x['status'] for x in b['fit_attempts'])),'original_ledger_unchanged':True,'unknown_original_outer_exit_codes_preserved':True,'observed_owned_live_training_processes':owned_training,'remaining_runtime_processes':proc,'gpu_clients_preserved':nv_rows,'no_signals_sent':True}
write(out/'GPU_RELEASE_RECEIPT.json',cleanup);write(H/'receipts/final-gpu-release-r1.json',cleanup)
rows=r['comparison_rows'];reference=r['fixed_TL3_reference']['wp'];best=next(x for x in rows if x['name']=='equal_weight')
summary={'schema':'wnn.multistage-final-summary.v1','status':'complete','configured_final_models':39,'partial_models':1,'scientific_attempts':40,'profile_attempts':56,'primary_fit_train_sequences':10325,'primary_fit_valid_sequences':0,'primary_valid_sequences':1873,'scored_rows':3528273,'comparison_rows':rows,'best_new_ensemble_delta_vs_fixed_TL3':best['wp']-reference,'weights_frozen_before_full_refits':True,'primary_valid_used_for_weights':False,'independent_arithmetic_verified_outputs':6,'public_score_measured':False,'gpu_release_receipt':'GPU_RELEASE_RECEIPT.json'};write(out/'FINAL_SUMMARY.json',summary)
release='https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-full-primary-checkpoints-20261031-r1'
oldrelease='https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-multistage-checkpoints-20261031-r1'
lines=['# Final four-stage B300 wave: full-primary refits','','| Candidate | Official valid WP |','|---|---:|']+['| '+x['name']+' | '+format(x['wp'],'.9f')+' |' for x in rows]+['','Best new mixture: equal four-member weights, WP '+str(best['wp'])+'. Fixed fresh TL3 reference '+str(reference)+'; delta '+str(best['wp']-reference)+'. Baseline not beaten. Both mixtures were frozen before the four refits; no validation weight search.','', 'All four fresh refits used the same 10,325 eligible earlier official-train sequences, zero fitting valid sequences. Full 1,873-sequence official valid / 3,528,273 scored rows after warmup99. TL3 is a fixed previously evaluated reference, not rescored here; training recipes/budgets differ. Validation is development-exposed, not independent confirmation.','', 'Forty scientific attempts produced 39 configured-final model artifacts and one partial GALAM. Main auxiliary study: 15 base, 6 fine-tune, 12 paired student/control endpoints; two earlier fine-tune pilots are separate. Four fresh full-primary direct-label refits complete the count. The original ledger retains 9 terminal_unverified exits and a stale GALAM running state; artifact completion and current process absence are separately evidenced. No invented exit codes. 56 bounded engineering profiles, 24 completed CPU Optuna mixture trials.','', 'Independent NumPy/PyArrow arithmetic reads raw official labels/mask: all four singletons and both mixtures reproduce within 1e-10; mixtures are FP32 bit-exact. This does not replicate training or establish a public score.','', 'Native train/valid curves are included. Train diagnostic cohort uses 16 fitting sequences/all rows after warmup; valid uses all official valid/is_scored. Their absolute gaps are not directly comparable. Configured-final endpoints remain fixed despite visible trajectories.','', '[Four full-primary configured-final weights and SHA-verified assets]('+release+'). [33 auxiliary configured-final models and pinned replay source capsules]('+oldrelease+'). Both are research prereleases, not a serving ABI/submission. Two historical pilots and partial GALAM are excluded from releases.','', 'GPU_RELEASE_RECEIPT.json verifies no live tabdata GPU clients after training/scoring. Foreign NIER/MPS clients are preserved; no signals were sent. Historical evidence is immutable. Large original JSONs are losslessly gzipped with compressed/original byte hashes.','']
(out/'README.md').write_text(chr(10).join(lines))
manifest={'schema':'wnn.multistage-final-delivery.v1','files':[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]};write(out/'DELIVERY_MANIFEST.json',manifest)
sys.path.insert(0,str(H/'reporting-r3/scripts'))
from multistage_github_node_api import GitHubAPI,APIError
from galam_l4_quality8.publisher import read_token
api=GitHubAPI(read_token(H/'private/github-report.token')[0]);assert api.request('GET','/user')['login']=='SemyonEpanov';repo='/repos/VladKochetov007/WunderFunder';branch='results/tabdata-multistage-b300-20261031';blobs=[]
for p in sorted(out.rglob('*')):
 if not p.is_file():continue
 assert p.stat().st_size<2000000,(p,p.stat().st_size)
 obj=api.request('POST',repo+'/git/blobs',{'content':base64.b64encode(p.read_bytes()).decode(),'encoding':'base64'})
 blobs.append({'path':'research/experiment_drafts/multistage-optuna-b300-20261031/results/full-primary-final-r1/'+str(p.relative_to(out)),'mode':'100644','type':'blob','sha':obj['sha']})
for attempt in range(3):
 parent=api.request('GET',repo+'/git/ref/heads/'+branch)['object']['sha'];tree0=api.request('GET',repo+'/git/commits/'+parent)['tree']['sha'];tree=api.request('POST',repo+'/git/trees',{'base_tree':tree0,'tree':blobs});commit=api.request('POST',repo+'/git/commits',{'message':'Record completed full-primary refits, frozen blends and released GPUs','tree':tree['sha'],'parents':[parent]})
 try:api.request('PATCH',repo+'/git/refs/heads/'+branch,{'sha':commit['sha'],'force':False});break
 except APIError as exc:
  if exc.status not in (409,422) or attempt==2:raise
url='https://github.com/VladKochetov007/WunderFunder/tree/'+branch+'/research/experiment_drafts/multistage-optuna-b300-20261031/results/full-primary-final-r1'
body='<!--tabdata-full-primary-final-r1--> **Final results:** 39 configured-final model artifacts + 1 partial GALAM, 40 scientific attempts / 56 engineering profiles. Four fresh full-primary refits complete: GRU 0.682425941, regularized GRU 0.681210508, LSTM 0.682520594, Transformer 0.676479699. Frozen equal4 **WP 0.684699354**, frozen Stage4-direct4 **0.684304429**, fixed TL3 **0.692377425**: baseline not beaten (best delta -0.007678071). Same 10325 eligible fitting members, zero fitting valid; all1873 valid / 3528273 scored rows. Different training recipes/budgets; development-exposed report only. Independent raw-label NumPy/PyArrow check verifies all6 outputs and bit-exact blends. **No tabdata GPU clients remain; NIER/MPS preserved.** [Full table, train/valid curves, raw receipts and cleanup proof]('+url+'). [Four full-primary weights]('+release+'); [33 auxiliary weights + replay sources]('+oldrelease+'), actual GitHub download hashes checked. Original unknown exits remain unobserved. No new public score.'
comment=api.request('POST',repo+'/issues/53/comments',{'body':body})
receipt={'status':'complete','commit':commit['sha'],'comment_url':comment['html_url'],'url':url,'files':len(blobs),'author':'SemyonEpanov','manifest_sha256':sha(out/'DELIVERY_MANIFEST.json')};write(H/'receipts/publish-full-primary-final-r1.json',receipt);print('FINAL_PUBLISHED',comment['html_url'],len(blobs),flush=True)
