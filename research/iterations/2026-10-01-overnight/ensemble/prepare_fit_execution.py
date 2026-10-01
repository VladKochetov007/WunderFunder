"""Build root run_queue schema from frozen ENS scientific plan + completed immutable manifests."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
 plan=json.loads((HERE/'plan.json').read_text());audit=json.loads((HERE/'cache-completion-audit.json').read_text());assert audit['status']=='PASS' and audit['cache_jobs_completed']==8 and audit['manifest_jobs_completed']==2
 code={name:digest(HERE/name) for name in plan['code_sha256']};assert code==plan['code_sha256']
 for rel in ['../training/predict_fixed.py','../../../../scripts/train_rnn.py','../../../../src/wnn/nn.py','../../../../src/wnn/features.py','../../../../src/wnn/data.py','../../../../src/wnn/resources.py','plan.json','cache-completion-audit.json']:code[rel]=digest(HERE/rel)
 for record in audit['manifests']:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];code[str(path.relative_to(HERE))]=record['sha256']
 inventory=json.loads((HERE.parent/'training/checkpoint-inventory.json').read_text())['files']
 for fold in (0,1):
  for run in ('ref','refs4'):
   original=f'/home/grinchenkoeg/WunderFunder/artifacts/runs/{run}_f{fold}/model_ep3.pt'
   code[f'../../../../artifacts/runs/{run}_f{fold}/model_ep3.pt']=inventory[original]['sha256']
 execution={'id':'ENSHEAD-01-fit-execution','frozen_spec_sha256':digest(HERE/'plan.json'),'independent_review': '../critic/check-ensemble-result.json','source_cache_audit_sha256':digest(HERE/'cache-completion-audit.json'),'code_sha256':code,'new_forecasting_fits':8,'new_candidate_configurations':2,'no_per_epoch_WP':True,'no_fits_started_by_actor':True,'jobs':plan['fit_jobs']}
 with (HERE/'execution-plan.json').open('x') as out:json.dump(execution,out,indent=2);out.write('\n')
 print(json.dumps({'path':str(HERE/'execution-plan.json'),'sha256':digest(HERE/'execution-plan.json'),'jobs':len(execution['jobs']),'verified_manifest_count':len(audit['manifests'])}))
if __name__=='__main__':main()
