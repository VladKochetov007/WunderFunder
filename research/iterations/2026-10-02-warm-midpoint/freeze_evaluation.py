"""Freeze independent grader and existing sums before midpoint predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=HERE/'training-allocation.json';allocation=json.loads(a.read_text());assert allocation['execution_authorized']
 assert not (HERE/'execution-01').exists() and not (HERE/'launch.json').exists()
 sources={str((HERE/'evaluate.py').relative_to(ROOT)):'79a92747abd1983ca575eee2c479bd3dd0a5ac3eeaaaf3c585885440e28af885',
  'research/iterations/2026-10-01-ensemble-backbone/evaluate.py':'c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9',
  'research/iterations/2026-10-02-ordinary-warm/evaluate.py':'f29b97efa4e767cb85f83e9c44dcee678ed4b84e9f9e28dce18e31f22d6cf865',
  'research/iterations/2026-10-02-ordinary-warm-seed4/evaluate.py':'adcf8d422dd6bc63bf76b5c72e087d830c9984c1e8ca03d96f72789239fdf6c5'}
 for p,h in sources.items():assert sha(ROOT/p)==h,p
 assert allocation['sources'][str((HERE/'evaluate.py').relative_to(ROOT))]==sources[str((HERE/'evaluate.py').relative_to(ROOT))]
 legacy='573cd5cbaaddbda844abd4fd6736fe01de01726b2ceb118ed1b5b4d1682b8ff8';assert sha(ROOT/'research/iterations/2026-10-01-late-atom-head/input-bindings.json')==legacy
 inputs=[HERE/'prepare-01/checkpoints.json']+[ROOT/f'research/iterations/2026-10-02-{branch}/evaluation-01/{name}' for branch in ('ordinary-warm','ordinary-warm-seed4') for name in ('report.json','receipt.json','sufficient.npz')]
 r=dict(status='FROZEN_BEFORE_METRIC_READOUT',utc=datetime.now(timezone.utc).isoformat(),training_allocation_sha256=sha(a),source_sha256=sources,legacy_bindings_sha256=legacy,
  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},resource_budget=dict(CPU_quota=200,CPU_affinity=[28,29],host_memory_bytes=4*2**30,swap=0,timeout_seconds=600,GPU=False,metric_readouts=1),
  expected_report_status='COMPLETE_FIXED_WARM_MIDPOINT_SCREEN',root_contract_sha256=sha(HERE/'root-contract.json'),verification='No fits, onefixedhalf, fourconstructedmodels and fourtargetfreeOOF; no weightsearch or seedselection.')
 out=HERE/'evaluation-source.json'
 with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=r['status'],evaluation_source_sha256=sha(out))))
if __name__=='__main__':main()
