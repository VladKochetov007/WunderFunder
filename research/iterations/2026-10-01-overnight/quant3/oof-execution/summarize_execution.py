"""Metadata/hash-only completion report, after all four frozen OOF jobs succeed."""
from pathlib import Path
import datetime,hashlib,json,os,resource
D=Path(__file__).resolve().parent;Q=D.parent;ROOT=Q.parents[3]
os.sched_setaffinity(0,{36});resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
summary=json.loads((D/'summary.json').read_text());assert summary['status']=='COMPLETE_PENDING_ROOT_PROVENANCE_AUDIT'
allocation=Q/'oof-allocation.json';assert sha(allocation)==summary['allocation_sha256']=='a98412b6903dd027acdcd92b4b1dbdfcb147347ed0d00a26b7c6532409d3cedf'
alloc=json.loads(allocation.read_text());freeze=json.loads((Q/'FREEZE.json').read_text());assert sha(Q/'FREEZE.json')==alloc['freeze_sha256']
for group in ['source_and_binaries','actor_receipts','independent_receipts']:
 for path,expected in freeze[group].items():assert sha(ROOT/path)==expected,path
assert sha(ROOT/freeze['zip'])==alloc['zip_sha256']
for phase in ['before','after']:
 r=json.loads((D/f'input-identity-{phase}.json').read_text());assert r['allocation_sha256']==sha(allocation)
 assert all(v['match'] and v['observed']==v['expected'] for v in r['inputs'].values())
jobs=[];children={}
for fold in [0,1]:
 for mode in ['quant','fp16']:
  path=Q.parent/'cache'/f'pred_on_q3{mode}_f{fold}_valid.npy';meta=path.with_suffix('.json');m=json.loads(meta.read_text())
  assert m['fold']==fold and m['epoch']==1 and len(m['rows'])==[467,469][fold]
  assert not m['targets_read'] and not m['masks_read'] and m['metrics_computed']==0
  assert sha(path)==m['prediction_sha256'] and path.stat().st_size==299680128
  assert sha(m['checkpoint'])==m['checkpoint_sha256'];manifest=json.loads(Path(m['checkpoint']).read_text())
  assert manifest['kind']=='ensemble_recipe_manifest_not_torch_checkpoint' and manifest['children']==m['child_checkpoints']
  assert sha(manifest['native'])==manifest['native_sha256']==m['native_sha256'] and manifest['zip_sha256']==m['zip_sha256']==alloc['zip_sha256']
  for c in m['child_checkpoints']:assert sha(c['checkpoint'])==c['checkpoint_sha256'] and sha(c['export'])==c['export_sha256']
  if fold in children:assert children[fold]==m['child_checkpoints']
  else:children[fold]=m['child_checkpoints']
  usage=(D/f'fold{fold}-{mode}.time.txt').read_text();assert 'Exit status: 0' in usage
  peak=int(next(x.split(':',1)[1].strip() for x in usage.splitlines() if 'Maximum resident set size' in x))
  jobs.append({'fold':fold,'mode':mode,'sequences':len(m['rows']),'prediction_path':str(path),'prediction_sha256':m['prediction_sha256'],'prediction_bytes':path.stat().st_size,'receipt_sha256':sha(meta),'composite_checkpoint_sha256':m['checkpoint_sha256'],'seconds':m['seconds'],'maximum_process_RSS_KiB':peak,'native_sha256':m['native_sha256']})
guard=[json.loads(x) for x in (D/'watchdog-v2.jsonl').read_text().splitlines()]
assert not any(x['event'] in ['guard_kill','IDENTITY_MISMATCH_NO_SIGNAL'] for x in guard)
rss={str(f):max(x['aggregate_rss_bytes'] for x in guard if x['event']=='sample' and x['fold']==f) for f in [0,1]}
out={'status':'COMPLETE_PENDING_INDEPENDENT_PROVENANCE_AUDIT_AND_ROOT_QUALITY_READS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation_sha256':sha(allocation),'freeze_sha256':sha(Q/'FREEZE.json'),'zip_sha256':alloc['zip_sha256'],'input_identity_before_sha256':sha(D/'input-identity-before.json'),'input_identity_after_sha256':sha(D/'input-identity-after.json'),'execution_summary_sha256':sha(D/'summary.json'),'jobs':jobs,'sampled_group_peak_RSS_bytes_since_watchdog_v2_started':rss,'guard_actions':0,'actor_targets_decoded':False,'actor_masks_decoded':False,'actor_quality_queries':0,'AVX2_audit':False,'prediction_arrays_relocated':False,'source_and_ZIP_after_checks':'PASS','reporter_source_sha256':sha(__file__)}
outpath=Q/'oof-completion.json';assert not outpath.exists();outpath.write_text(json.dumps(out,indent=2)+'\n')
lines=['# Q3 OOF completion receipt','','All four frozen producers completed successfully. This is a provenance and resource report; quality has not been evaluated by this actor. Independent audit and the root\'s two registered comparisons remain pending.','','| Fold | Engine | Sequences | Seconds | Peak process RSS KiB | Prediction SHA256 |','|---|---|---:|---:|---:|---|']
for j in jobs:lines.append(f"| {j['fold']} | {j['mode']} | {j['sequences']} | {j['seconds']:.2f} | {j['maximum_process_RSS_KiB']} | `{j['prediction_sha256']}` |")
lines+=['',f"Allocation: `{out['allocation_sha256']}`. ZIP: `{out['zip_sha256']}`. Exact epoch1 children for each fold are `on_ctld256`, `on_ctl128s3`, `on_ctlc500`, in that order, with float32 equal-thirds and original-order FMA reduction. The JSON checkpoint artifacts are explicitly composite recipe manifests, not Torch checkpoints.",'','Inputs were rehashed before any producer and after all four jobs: raw validation parquet, raw-reader source, chain-fold metadata and serialized row-order metadata. Target/mask arrays were not decoded. All expected sizes, paths, mtimes and SHA256 hashes match. Sources, binaries, ZIP, child checkpoints, exports and four output files passed final hash checks.','',"Predictions remain remotely in `P/cache/pred_on_q3{quant,fp16}_f{0,1}_valid.npy`; each is float32 `(1873,20000,2)`,299,680,128bytes, with unselected sequences NaN. Local sync contains manifests, exports, metadata and execution receipts, not duplicate full prediction caches.",'',f"Resource allocation: one compute CPU per lane (36/37), BLAS/OMP threads1, GPU0. Each process had8GiB RLIMIT_AS. Sampled aggregate lane RSS after watchdog_v2 started peaked at {rss['0']:,}/{rss['1']:,}bytes; these sampled group peaks are distinct from GNU time's per-process peaks. The scorer exposed2 OS threads with one idle library thread; affinity limited each lane to one CPU.",'','The original watchdog and its replacement are preserved. Version2 binds process start identities and uses a monotonic deadline; no deadline/memory kill or identity refusal occurred. Its orphan fallback intentionally refuses to signal never-observed descendants, so this is not an unconditional cgroup kill guarantee. Aggregate RSS samples can miss between-sample peaks. Actual successful durations and process peaks are preserved separately in the receipts.','',"These OOF execution durations do not replace the earlier complete-callback runtime gate. Only the measured v3/VNNI path has passed the50min runtime threshold. AVX2 failed its first runtime pilot, so no AVX2 OOF audit or quality query was performed. No fresh-holdout or competition-ranking claim is made.",'']
(Q/'oof-report.md').write_text('\n'.join(lines));print(json.dumps(out),flush=True)
