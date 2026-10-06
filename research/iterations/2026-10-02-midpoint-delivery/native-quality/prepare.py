"""Export exactly the two allocated fold recipes and bind inputs before prediction."""
import time
START=time.monotonic()
import json,hashlib
from pathlib import Path
from common import P,ROOT,Q4,NAMES,NATIVE_SHA,sha,write,identity,guards,rowmaps
import numpy as np
import torch

def main():
    allocation,limits=guards(30);torch.set_num_threads(1)
    groups,info,orderhash=rowmaps();native=ROOT/'research/iterations/2026-10-01-overnight/quant3/native/build/v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so';assert sha(native)==NATIVE_SHA
    known=json.loads((Q4/'prepared.json').read_text());assert sha(Q4/'prepared.json')==allocation['reference_prepared_sha256']
    inputs={}
    for rel,expected in known['inputs'].items():
        path=ROOT/rel;before=identity(path);digest=sha(path);assert digest==expected['sha256'] and before['bytes']==expected['bytes'] and identity(path)==before
        inputs[rel]={**before,'sha256':digest}
    for rel,digest in allocation['midpoint_proof_sha256'].items():assert sha(ROOT/rel)==digest,rel
    refs={}
    for fold in (0,1):
        path=Q4/'cache'/f'pred_student_quant_f{fold}_valid.npy';receipt=path.with_suffix('.json');meta=json.loads(receipt.read_text());digest=sha(path)
        assert digest==allocation['reference_prediction_sha256'][str(fold)]==meta['prediction_sha256']
        assert sha(receipt)==allocation['reference_receipt_sha256'][str(fold)]
        assert [c['run'] for c in meta['child_checkpoints']]==[f'{n}_f{fold}' for n in ('cont_sh256_s3h2',*NAMES[1:])]
        assert meta['source_row_group_order_sha256']==orderhash
        refs[str(fold)]={'path':str(path),'sha256':digest,'receipt':str(receipt),'receipt_sha256':sha(receipt)}
        d=P/f'fold{fold}';d.mkdir(exist_ok=False);children=[]
        for i,name in enumerate(NAMES):
            path=ROOT/'artifacts/runs'/f'{name}_f{fold}'/'model_ep1.pt';digest=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);cfg=ck['args']
            assert int(ck['epoch'])==1 and cfg['valid_fold']==fold and ck['n_in']==211
            if i==0:
                assert digest==allocation['midpoint_checkpoint_sha256'][str(fold)]
                pair=next(j for j in json.loads((ROOT/'research/iterations/2026-10-02-warm-midpoint/commands.json').read_text())['pairs'] if j['name']==f'{name}_f{fold}')
                assert cfg==pair['resolved_args'] and pair['seed']==3 and pair['fold']==fold
                construction_path=path.with_name('construction.json');cr=json.loads(construction_path.read_text())
                assert sha(construction_path)==allocation['midpoint_proof_sha256'][str(construction_path.relative_to(ROOT))]
                assert cr['checkpoint']['sha256']==digest and cr['alpha']==.5 and cr['seed']==3 and cr['fold']==fold
                assert cr['mean_std_byte_identical'] and cr['all10_parameters_exact_midpoint'] and cr['ordinary_strict_load']
                assert ck['construction']=={k:cr[k] for k in ck['construction']}
                for role in ('old','warm'):assert cr[role+'_checkpoint']==dict(path=pair[role]['path'],sha256=pair[role]['sha256'],epoch=1)
            else:
                assert digest==meta['child_checkpoints'][i]['checkpoint_sha256'], 'Anchor changed from current native comparator'
            assert cfg['hidden']==(256 if i==0 else 128) and cfg['layers']==(1 if i==0 else 2)
            assert cfg.get('arch','gru')=='gru' and cfg.get('cell','gru')=='gru' and cfg.get('enc','linear')=='linear'
            assert not any(cfg.get(k) for k in ['lookahead','class_heads','heuristics','quant_feats','prop_input','atom_head','online_norm','transition_features','state_reset'])
            arrays={k:v.detach().cpu().float().numpy() for k,v in ck['model'].items()};arrays['head.2.weight']=arrays['head.2.weight'][:2].copy();arrays['head.2.bias']=arrays['head.2.bias'][:2].copy()
            assert arrays['inp.weight'].shape==((256 if i==0 else 128),211) and all(np.isfinite(v).all() for v in arrays.values()) and (arrays['std']>0).all()
            pathout=d/f'weights_{i}.npz';np.savez(pathout,**arrays);assert sha(path)==digest
            children.append({'run':f'{name}_f{fold}','checkpoint':str(path),'checkpoint_sha256':digest,'epoch':1,'fold':fold,'args':cfg,'export':str(pathout),'export_sha256':sha(pathout),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()},'coef':.25,**({'midpoint_construction_receipt_sha256':sha(construction_path)} if i==0 else {})})
        manifest={'schema':'midpoint-native-quant-fold-v1','midpoint_proof_sha256':allocation['midpoint_proof_sha256'],'fold':fold,'epoch':1,'children':children,'coefficients':[.25]*4,'native':str(native),'native_sha256':NATIVE_SHA,'native_backend':'auto natural VNNI','source_row_group_order_sha256':orderhash,'chain_folds_sha256':inputs['cache/chain_folds.npz']['sha256'],'rows':[i for i,g in enumerate(groups) if info[int(g)][0]==fold],'allocation_sha256':sha(P/'allocation.json'),'predictor_sha256':sha(P/'predict.py'),'exporter_sha256':sha(__file__),'inputs':inputs,'targets_decoded':False,'WP_queries':0}
        write(d/'checkpoint.json',manifest)
    write(P/'prepared.json',{'status':'BOUND_TWO_FOLDS_BEFORE_PREDICTION','inputs':inputs,'reference':refs,'row_order_sha256':orderhash,'fold_manifests':{str(f):sha(P/f'fold{f}/checkpoint.json') for f in (0,1)},'allocation_sha256':sha(P/'allocation.json'),'seconds':time.monotonic()-START,'limits':limits,'targets_decoded':False,'WP_queries':0})
    print(json.dumps({'status':'PREPARED','seconds':time.monotonic()-START}),flush=True)
if __name__=='__main__':main()
