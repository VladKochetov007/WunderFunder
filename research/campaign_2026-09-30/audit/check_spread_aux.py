"""Independent synthetic loss/gradient, cache/provenance and raw-price checks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
ACTOR = AUDIT.parent / "spread_aux"


def digest(path: Path) -> str:
    """Hash a file using bounded working memory."""
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): d.update(block)
    return d.hexdigest()


def main() -> None:
    """Exercise actor handoff without applying its trainer patch or fitting."""
    start = time.monotonic()
    scope=os.environ['WNN_SCOPE_UNIT']+'.scope'
    properties=subprocess.check_output(['systemctl','--user','show',scope,'-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec','-p','Slice'],text=True)
    scope_properties=dict(line.split('=',1) for line in properties.splitlines())
    assert scope_properties['MemoryMax']=='2147483648'
    assert scope_properties['MemorySwapMax']=='0'
    assert scope_properties['CPUQuotaPerSecUSec']=='1s'
    assert scope_properties['Slice']=='wnn-campaign.slice'
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("actor_spread_support",ACTOR/"trainer_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator().manual_seed(926)
    base = torch.randn(2,7,25,generator=generator)
    proxy = torch.randint(12,(2,7),generator=generator)
    current = (proxy+3)%12
    branch = torch.randint(3,(2,7),generator=generator)
    classes = torch.randint(4,(2,7),generator=generator)
    weight = torch.tensor([[0.,.2,1.,.5,3.,0.,.9],[1.,.5,0.,.7,2.,.1,1.]])
    masks = [torch.ones(2,7,dtype=torch.bool),torch.zeros(2,7,dtype=torch.bool),torch.arange(14).reshape(2,7)%3!=0]
    cases = []
    for name, labels in [("proxy_known",proxy),("current_known",current)]:
        for known in masks:
            actual_input = base.clone().requires_grad_()
            encoded = torch.where(known,current,255).to(torch.uint8)
            actual = module.structural_loss(actual_input,labels,branch,classes,weight,2,12,encoded)
            actual_grad = torch.autograd.grad(actual,actual_input)[0]
            oracle_input = base.clone().requires_grad_()
            # Independent log-softmax/gather CE, original denominator retained.
            term = torch.zeros_like(weight)
            for lo,hi,target,mask in [(2,14,labels,known),(14,17,branch,torch.ones_like(known)),(17,21,classes,torch.ones_like(known))]:
                ce = -torch.log_softmax(oracle_input[...,lo:hi],-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
                term += ce*mask
            expected = (term*weight).sum()/weight.sum().clamp_min(1e-8)
            expected_grad = torch.autograd.grad(expected,oracle_input)[0]
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(actual_grad,expected_grad,atol=2e-7,rtol=2e-6)
            assert torch.count_nonzero(actual_grad[...,2:14][~known]) == 0
            assert torch.count_nonzero(actual_grad[...,:2]) == 0
            assert torch.count_nonzero(actual_grad[...,21:]) == 0
            cases.append({"arm":name,"known":int(known.sum()),"loss":float(actual),"denominator":float(weight.sum())})
    # All-known reproduces legacy; all-unknown preserves branch/class gradients.
    a,b = [base.clone().requires_grad_() for _ in range(2)]
    legacy = module.structural_loss(a,proxy,branch,classes,weight,2,12,None)
    all_known = module.structural_loss(b,proxy,branch,classes,weight,2,12,current.to(torch.uint8))
    assert torch.equal(legacy,all_known)
    assert torch.equal(torch.autograd.grad(legacy,a)[0],torch.autograd.grad(all_known,b)[0])
    x,y = [base.clone().requires_grad_() for _ in range(2)]
    masked = module.structural_loss(x,proxy,branch,classes,weight,2,12,torch.full((2,7),255,dtype=torch.uint8))
    ordinary = module.structural_loss(y,proxy,branch,classes,weight,2,12,None)
    gx,gy = torch.autograd.grad(masked,x)[0],torch.autograd.grad(ordinary,y)[0]
    assert torch.equal(gx[...,14:21],gy[...,14:21])
    zero = module.structural_loss(base,proxy,branch,classes,torch.zeros_like(weight),2,12,current.to(torch.uint8))
    assert float(zero)==0
    rejections=[]
    with tempfile.TemporaryDirectory(prefix="spread_critic_") as temp:
        fixture=Path(temp)
        (fixture/"selection.json").write_text('{}')
        (fixture/"interior_code_table.json").write_text('[]')
        metadata={"status":"complete","supervision_only":True,"N_SP_AUX":12,"split_manifest_sha256":"split","fit_train_groups":[7],"fit_valid_groups":[7],"evaluation_valid_groups":[8],"unknown":255,"dtype":"uint8","selection_sha256":digest(fixture/"selection.json"),"map_sha256":digest(fixture/"interior_code_table.json"),"files":{}}
        for split in ["train","valid"]:
            p=fixture/f"spread_{split}.npy";np.save(p,np.zeros((1,20000),dtype=np.uint8))
            metadata["files"][split]={"file":p.name,"sha256":digest(p),"row_groups":[7],"shape":[1,20000]}
        def write(m:dict)->None: (fixture/"metadata.json").write_text(json.dumps(m))
        write(metadata);module.validate_spread_cache(fixture,"split",[7],[7],[8])
        try: module.PackedSpreadCache(fixture,"valid").read(8)
        except ValueError: rejections.append("evaluation_cache_read")
        else: raise AssertionError("outside membership read accepted")
        changes=[("split",lambda m:m.update(split_manifest_sha256="other")),("files",lambda m:m["files"].pop("valid")),("bins",lambda m:m.update(N_SP_AUX=13)),("duplicates",lambda m:m.update(fit_train_groups=[7,7])),("eval_overlap",lambda m:m.update(evaluation_valid_groups=[7])),("shape",lambda m:m["files"]["train"].update(shape=[1,19999])),("order",lambda m:m["files"]["train"].update(row_groups=[6])),("arrayhash",lambda m:m["files"]["train"].update(sha256="bad")),("maphash",lambda m:m.update(map_sha256="bad"))]
        for name,change in changes:
            altered=copy.deepcopy(metadata);change(altered);write(altered)
            try: module.validate_spread_cache(fixture,"split",[7],[7],[8])
            except ValueError: rejections.append(name)
            else: raise AssertionError(f"accepted {name}")
        bad=np.zeros((1,20000),dtype=np.uint8);bad[0,113]=12;np.save(fixture/"spread_train.npy",bad)
        altered=copy.deepcopy(metadata);altered["files"]["train"]["sha256"]=digest(fixture/"spread_train.npy");write(altered)
        try:module.validate_spread_cache(fixture,"split",[7],[7],[8])
        except ValueError:rejections.append("self_consistent_invalid_label")
        else:raise AssertionError("accepted invalid12 bin")
    selection=json.loads((ACTOR/"selection.json").read_text())
    split=json.loads((ROOT/selection["split_manifest_path"]).read_text())
    assert digest(ROOT/selection["split_manifest_path"])==selection["split_manifest_sha256"]
    for relative,sha in selection["source_sha256"].items(): assert digest(ROOT/relative)==sha
    for relative,identity in selection["parquet_identities"].items():
        stat=(ROOT/relative).stat();assert stat.st_size==identity["bytes"] and stat.st_mtime_ns==identity["mtime_ns"]
    component={source:{r["row_group"]:r["component"] for r in rows} for source,rows in split["sequences"].items()}
    evaluated={component["valid"][g] for g in split["evaluation"]["valid"]}
    fitting={component[s][g] for s in ['train','valid'] for g in split['fit'][s]}
    assert not fitting & evaluated
    contract=json.loads((ACTOR/'preregistration.json').read_text())
    salt=contract['selection_salt'];choices={}
    for group in split['fit']['train']: choices.setdefault(component['train'][group],[]).append(group)
    rank=lambda kind,value:hashlib.sha256(f'{salt}:{kind}:{value}'.encode()).digest()
    components=sorted(choices,key=lambda c:rank('component',c))[:512]
    selected=[min(choices[c],key=lambda g:rank('group',g)) for c in components]
    assert selected==selection['map_fit_train_groups'] and components==selection['map_fit_components']
    assert len(set(components))==512 and not set(components)&evaluated
    votes=json.loads((ACTOR/'all_code_votes.json').read_text())
    accepted=[]
    for row in votes:
        ranked=sorted(row['all_votes'],key=lambda v:(-v['votes'],v['class'],v['spread_ticks']))
        winner=ranked[0];total=sum(v['votes'] for v in ranked)
        assert row['fit_total_votes']==total and row['fit_vote_count']==winner['votes']
        assert (row['class'],row['spread_ticks'])==(winner['class'],winner['spread_ticks'])
        keep=winner['votes']>=3 and winner['votes']/total>=.95
        assert row['accepted']==keep
        if keep: accepted.append({k:row[k] for k in ['code_float32_round4_hex','class','spread_ticks','fit_vote_count','fit_total_votes']})
    assert accepted==json.loads((ACTOR/'interior_code_table.json').read_text())
    provenance=module.validate_spread_cache(ACTOR,selection["split_manifest_sha256"],selection["fit_train_groups"],selection["fit_valid_groups"],selection["evaluation_valid_groups"])
    table={bytes.fromhex(v["code_float32_round4_hex"]):min(int(v["spread_ticks"]),12)-1 for v in json.loads((ACTOR/"interior_code_table.json").read_text())}
    raw_checks=[]
    columns=[f"i0_p{i}" for i in range(22)]
    for split in ["train","valid"]:
        reader=pq.ParquetFile(ROOT/f"datasets/{split}.parquet")
        cache=module.PackedSpreadCache(ACTOR,split)
        groups=selection[f"fit_{split}_groups"]
        for index in [0,len(groups)//2,len(groups)-1]:
            group=groups[index];raw=reader.read_row_group(group,columns=columns,use_threads=False)
            prices=np.column_stack([raw[c].to_numpy() for c in columns]).astype(np.float32)
            codes=np.ascontiguousarray(np.round(prices,4),dtype='<f4').view('V88').reshape(-1)
            independent=np.fromiter((table.get(bytes(c),255) for c in codes),dtype=np.uint8,count=20000)
            np.testing.assert_array_equal(independent,cache.read(group))
            raw_checks.append({"source":split,"group":group,"known":int((independent!=255).sum())})
    integration=json.loads((ACTOR/"integration_manifest.json").read_text())
    assert digest(ROOT/'scripts/train_rnn.py')==integration['trainer_base_sha256']
    assert digest(ACTOR/'trainer_support.py')==integration['support_source_sha256']
    assert digest(ACTOR/'integration.patch')==integration['patch_sha256']
    subprocess.run(['git','apply','--check',str(ACTOR/'integration.patch')],cwd=ROOT,check=True)
    assert not torch.cuda.is_initialized()
    result={"status":"pass","scope_properties":scope_properties,"synthetic_cases":cases,"all_known_legacy_loss_gradient_exact":True,"all_unknown_branch_class_gradient_exact":True,"zero_weight_loss":True,"rejections":rejections,"selection_independently_rederived_512_components":True,"recorded_votes_independently_aggregated":len(votes),"accepted_codes":len(accepted),"raw_target_vote_replay":False,"raw_price_replays":raw_checks,"provenance":provenance,"trainer_live_unpatched":True,"cuda_initialized":False,"source_sha256":digest(Path(__file__)),"support_sha256":digest(ACTOR/'trainer_support.py'),"integration_manifest_sha256":digest(ACTOR/'integration_manifest.json'),"wall_seconds":time.monotonic()-start,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (AUDIT/'spread_aux_independent_checks_v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['synthetic_cases','provenance']}))


if __name__ == '__main__':main()
