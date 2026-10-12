import importlib.util, hashlib, json, sys
from pathlib import Path
root=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
helper=root/'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/cpu_shared_backbone_r1/benchmark_shared_f0.py'
spec=importlib.util.spec_from_file_location('shared_bench',helper)
mod=importlib.util.module_from_spec(spec); sys.modules['shared_bench']=mod; spec.loader.exec_module(mod)
pairs=[
(0,'oldz4_h230_student_f0_s3_r3','664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb','oldz4_h230_hft_f0_s3_r5','2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8'),
(1,'oldz4_h230_student_f1_s3_r4','cd8886589e8664082da1f27aee4f2c4925b61c41da8c740598831ab2f774bbd6','oldz4_h230_hft_f1_s3_r2','41e7aa3fb27024f2612f8f6b8f4232ca446ba92022dbf7ba90893ffee2292410')]
res=[]
for fold,srun,ssha,hrun,hsha in pairs:
 sp=root/f'artifacts/runs/{srun}/model_ep1.pt'; hp=root/f'artifacts/runs/{hrun}/model_ep0.pt'
 sw,sc=mod.load_checkpoint_weights(sp,ssha); hw,hc=mod.load_checkpoint_weights(hp,hsha)
 changed=sorted(k for k in sc['model'] if not __import__('torch').equal(sc['model'][k],hc['model'][k]))
 config_s=mod.canonical_inference_config(sc); config_h=mod.canonical_inference_config(hc)
 res.append({'fold':fold,'student_sha256':ssha,'hft_sha256':hsha,'student_epoch':sc['epoch'],'hft_epoch':hc['epoch'],'student_provenance':sc.get('provenance'),'hft_provenance':hc.get('provenance'),'changed_tensors':changed,'config_equal':config_s==config_h,'config':config_s,'served_shapes':[sw['head.2.weight'].shape,hw['head.2.weight'].shape]})
print(json.dumps(res,default=list,sort_keys=True))
