import json
import sys
import torch
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/src')
from wnn.training.dynamics_auxiliary import dynamics_targets_torch, HORIZONS

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
raw = torch.zeros((1, 20000, 112), dtype=torch.float32)
raw[:,100,:] = 10000
raw[:,15000::2,:] = .01
out, mask = dynamics_targets_torch(raw, torch.zeros(112), torch.ones(112), torch.ones((1,20000), dtype=torch.bool))
rows = []
for j, h in enumerate(HORIZONS):
    t = 16000
    delta = raw[0,t+1:t+h+1].double()-raw[0,t:t+h].double()
    expected = float(torch.log(delta.square().mean()+1e-8))
    actual = float(out[0,t,j*26+1])
    rows.append({'horizon':h, 'position':t, 'expected_log_energy_float64_direct_window':expected, 'actual_fp32_target':actual, 'absolute_error':abs(actual-expected), 'valid':bool(mask[0,t,j])})
record = {'status':'NUMERICAL_TARGET_CHECK', 'fixture':'20000rows; early10000pulse; latealternating0/.01; all112features; mean0/std1', 'rows':rows, 'gpu_seconds':0,'new_fits':0}
with open('/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R1.json','x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
