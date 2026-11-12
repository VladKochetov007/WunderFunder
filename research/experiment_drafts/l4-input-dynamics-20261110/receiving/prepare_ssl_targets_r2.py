"""Compute the canonical train-only moments in parallel with ordered FP64 reduction."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

R=Path('/data/lc-epanov_s/tabdata-next70-20261110-r1')
S=R/'source-ssl-r1'
sys.path.insert(0,str(S/'src'))


@lru_cache(maxsize=1)
def source_file() -> object:
    """Open only the official train file once per spawned process."""
    from wnn.data import SequenceFile
    return SequenceFile(R/'project/datasets/train.parquet')


def group_moments(group: int) -> tuple[int, object, object, int]:
    """Return the same per-sequence sum/square vectors as the serial canonical loop."""
    import numpy as np
    from wnn.training.sequence_pretraining import future_innovation_matrix

    values=source_file().read(group,targets=False,mask=False).features
    targets=future_innovation_matrix(values)
    return group,targets.sum(axis=0),np.square(targets).sum(axis=0),len(targets)


def finalize(total: object, square: object, count: int) -> tuple[object,object]:
    """Match the canonical FP64 population formula and FP32/std-floor serialization."""
    import numpy as np
    from wnn.training.sequence_normalization import NORMALIZATION_STD_FLOOR

    mean64=total/count
    variance64=np.maximum(square/count-np.square(mean64),0.0)
    std64=np.maximum(np.sqrt(variance64),NORMALIZATION_STD_FLOOR)
    return np.ascontiguousarray(mean64.astype(np.float32)),np.ascontiguousarray(std64.astype(np.float32))


def main() -> None:
    """Verify serial parity, reduce fitting groups in their original order, then publish."""
    import numpy as np
    from wnn.data import SEQUENCE_LENGTH
    from wnn.training import sequence_normalization as norm
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_data import build_data
    from wnn.training.sequence_monitor_identity import membership_sha256
    from wnn.training.sequence_pretraining import FUTURE_INNOVATION_CONTRACT,TARGET_START,TARGET_STOP,future_innovation_matrix,prepare_future_innovation_records
    from wnn.training.sequence_provenance import canonical_sha256,source_identity

    control=R/'control/root-ssl-r2'
    control.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    config=load_config(R/'ssl-inputs-r1/configs/ssl_pretrain_s203201.json')
    data=build_data(config,R/'project',R/'project/chronological_primary.json',None,None)
    members=list(data.fit.members)
    assert len(members)==10325 and all(s=='train' for s,_ in members) and len(set(members))==10325
    source_fingerprints={'datasets/train.parquet':data.provenance['sources']['datasets/train.parquet']}
    contract=FUTURE_INNOVATION_CONTRACT|{'sequence_rows':SEQUENCE_LENGTH,'row_start':TARGET_START,'row_stop_exclusive':TARGET_STOP,'fit_members_sha256':membership_sha256(members),'source_fingerprints':source_fingerprints}
    identity=norm.build_normalization_identity(source_fingerprints,data.split.sha256,members,'all_fit',336,identity_extra=contract)
    identity_sha=norm.normalization_identity_sha256(identity)
    cache=R/'project/cache/sequence_future_innovations'/f'{identity_sha}.json'
    assert not cache.exists(), 'Do not replace a completed moments cache'
    total=np.zeros(336,dtype=np.float64); square=np.zeros(336,dtype=np.float64); count=0
    serial_members=members[:8]

    def serial_reader(source: str,group: int) -> object:
        """Use the unchanged full target matrix for an actual-data reference."""
        assert source=='train'
        return future_innovation_matrix(source_file().read(group,targets=False,mask=False).features)

    expected_mean,expected_std=norm.compute_population_moments(serial_members,serial_reader,feature_count=336)
    print(json.dumps({'phase':'parallel_train_only_target_preparation','workers':16,'groups':len(members),'reference_groups':8}),flush=True)
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,(group,sums,squares,rows) in enumerate(pool.map(group_moments,[g for _,g in members],chunksize=4)):
            assert group==members[index][1] and rows==TARGET_STOP-TARGET_START
            total+=sums; square+=squares; count+=rows
            if index==7:
                mean8,std8=finalize(total,square,count)
                assert np.array_equal(mean8,expected_mean) and np.array_equal(std8,expected_std)
                print(json.dumps({'strict_fp32_serial_parallel_parity':'passed','groups':8}),flush=True)
            if (index+1)%128==0:
                print(json.dumps({'groups_completed':index+1,'groups_total':len(members),'elapsed_seconds':time.monotonic()-started}),flush=True)
    mean,std=finalize(total,square,count)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    payload={'identity':identity,'identity_sha256':identity_sha,'mean':mean.tolist(),'std':std.tolist(),'moments_sha256':norm.moments_sha256(mean,std)}
    cache.parent.mkdir(parents=True,exist_ok=True)
    assert norm._write_cache_no_overwrite(cache,norm._canonical_json(payload))
    decoded=norm._decode_cache(cache,identity,identity_sha)
    assert np.array_equal(decoded.mean,mean) and np.array_equal(decoded.std,std)
    innovation=prepare_future_innovation_records(data,R/'project')
    assert innovation['moments_sha256']==payload['moments_sha256'] and innovation['normalization_identity_sha256']==identity_sha
    data.normalization()
    baselines=[]
    for index in range(8):
        item=data.fit[index]
        baselines.append(float(np.square(item['hints'][item['mask']].numpy(),dtype=np.float64).mean()))
    value={'schema':'wnn.l4-next70-ssl-target-preparation.v1','status':'complete','source_sha256':source_identity(S,S/'scripts/train_sequence.py')['sha256'],'config_sha256':canonical_sha256(config),'preparation_implementation':'parallel per-sequence sums and squares; original sequence-ordered FP64 reduction; canonical FP32 formula/cache decoder','preparation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'workers':16,'strict_serial_parallel_reference_groups':8,'strict_serial_parallel_parity':True,'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'innovation':innovation,'input_normalization':data.provenance['normalization'],'fitting_sequence_count':10325,'validation_fitting_members':0,'target_rows':count,'diagnostic_normalized_zero_prediction_mse_first8_fit_sequences':baselines,'diagnostic_scope':'fitting only, no new fit, not validation quality','elapsed_seconds':time.monotonic()-started}
    path=control/'target-preparation-receipt.json'
    assert not path.exists()
    path.write_text(json.dumps(value,sort_keys=True,indent=2))
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    main()
