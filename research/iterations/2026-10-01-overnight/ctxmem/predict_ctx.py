"""Fixed epoch1 raw-input-only OOF prediction, with no metric or label access."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import train_ctx as trainer
from wnn.data import SequenceFile
from model import ContextMemoryModel
from memory import metadata
from recipe import sha,plan,HERE,ROOT,verify_dataset_identity


class InputSequences(Dataset):
    def __init__(self,groups):self.groups=list(groups);self.reader=None
    def __len__(self):return len(self.groups)
    def __getitem__(self,index):
        if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/valid.parquet')
        return torch.from_numpy(self.reader.read(int(self.groups[index]),targets=False,mask=False).features)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    args=parser.parse_args();frozen=plan();job=next(j for j in frozen['jobs'] if j['name']==args.run)
    for name,expected in frozen['code_sha256'].items():assert sha(HERE/name)==expected,name
    input_hashes_before=verify_dataset_identity()
    trainer.configure_torch();checkpoint=ROOT/job['endpoint'];ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['args'];assert cfg['ctx_plan_sha256']==sha(HERE/'execution-plan.json')
    assert ck['epoch']==1 and cfg['valid_fold']==job['fold'] and cfg['seed']==job['seed'] and cfg['ctx_arm']==job['arm']
    expected=metadata(job['arm']);expected['source_fold']=job['fold'];assert cfg['context_memory']==expected
    assert ck['n_in']==211 and ck['readout'] is None and ck['readout_tilted'] is False
    model=ContextMemoryModel(arm=job['arm']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    assert sha(ROOT/'cache/chain_folds.npz')==frozen['chain_folds_sha256']
    # NPZ lazy access names only row_group; no target/scoring arrays materialized.
    with np.load(ROOT/'cache/valid_targets.npz') as archive:groups=archive['row_group']
    with np.load(ROOT/'cache/chain_folds.npz') as cf:
        chosen=cf['row_group'][(cf['split']=='valid')&(cf['fold']==job['fold'])]
    positions=np.flatnonzero(np.isin(groups,chosen));selected=groups[positions]
    loader=DataLoader(InputSequences(selected),batch_size=16,shuffle=False,num_workers=2,prefetch_factor=1)
    output=ROOT/job['prediction'];receipt=output.with_suffix('.inputs.json');temporary=output.with_name(output.name+'.partial')
    assert not any(p.exists() for p in (output,receipt,temporary));output.parent.mkdir(parents=True,exist_ok=True)
    out=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(groups),20000,2));out[:]=np.nan;written=0
    for raw in loader:
        pred=trainer.predict(model,raw,4000,'cuda');assert np.isfinite(pred).all()
        out[positions[written:written+len(pred)]]=pred;written+=len(pred)
    assert written==len(selected);out.flush();del out
    input_hashes_after=verify_dataset_identity();assert input_hashes_after==input_hashes_before
    temporary.replace(output)
    result={'run':args.run,'epoch':1,'fold':job['fold'],'seed':job['seed'],'arm':job['arm'],
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'prediction_path':str(output),'prediction_sha256':sha(output),'plan_sha256':sha(HERE/'execution-plan.json'),
            'script_sha256':sha(Path(__file__)),'feature_metadata':expected,'row_groups':selected.tolist(),
            'rows':positions.tolist(),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),
            'chain_folds_sha256':frozen['chain_folds_sha256'],'outside_fold':'NaN',
            'input_hashes_before':input_hashes_before,'input_hashes_after':input_hashes_after,
            'targets_read':False,'masks_read':False,'WP_queries':0,'metrics_computed':0}
    with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'run':args.run,'epoch':1,'sequences':written,'WP_queries':0}))


if __name__=='__main__':main()
