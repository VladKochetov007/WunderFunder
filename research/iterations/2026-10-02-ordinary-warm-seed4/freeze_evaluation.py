"""Root-invoked source freeze after allocation; no prediction/target/metric reads."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=HERE/'training-allocation.json';allocation=json.loads(a.read_text());assert allocation['execution_authorized']
 assert not (HERE/'execution-01').exists() and not (HERE/'launch.json').exists(),'Freeze before launch'
 sources={str((HERE/'evaluate.py').relative_to(ROOT)):'adcf8d422dd6bc63bf76b5c72e087d830c9984c1e8ca03d96f72789239fdf6c5',
          'research/iterations/2026-10-01-ensemble-backbone/evaluate.py':'c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'}
 for rel,h in sources.items():assert sha(ROOT/rel)==h
 assert allocation['sources'][str((HERE/'evaluate.py').relative_to(ROOT))]==sources[str((HERE/'evaluate.py').relative_to(ROOT))]
 legacy='573cd5cbaaddbda844abd4fd6736fe01de01726b2ceb118ed1b5b4d1682b8ff8'
 assert sha(ROOT/'research/iterations/2026-10-01-late-atom-head/input-bindings.json')==legacy
 f=dict(status='FROZEN_BEFORE_METRIC_READOUT',utc=datetime.now(timezone.utc).isoformat(),
  training_allocation_sha256=sha(a),source_sha256=sources,legacy_bindings_sha256=legacy,
  resource_budget=dict(CPU_quota=200,CPU_affinity=[28,29],host_memory_bytes=4*2**30,swap=0,timeout_seconds=600,GPU=False,metric_readouts=1),
  expected_report_status='COMPLETE_FIXED_ORDINARY_WARM_SEED4_SCREEN',
  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [(ROOT/'research/iterations/2026-10-01-continuation/student-seed4/evaluation-v2-01'/name) for name in ('report.json','sufficient.npz')]},
  comparisons=dict(primary=['ordinary_warm_seed4_4','original_seed4_4'],operational=['ordinary_warm_seed4_4','current_student4']),
  verification='Fixed4fits2OOF;103args,ep0/ep1,initial-state/normalization parity,ordinary-trunk H2 freezing and own-parent lineage; B pooled metric and whole-chain paired draws unchanged. No target or metric reads during source freeze.')
 out=HERE/'evaluation-source.json';assert not out.is_symlink()
 with out.open('x') as stream:json.dump(f,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(status=f['status'],evaluation_source_sha256=sha(out))))
if __name__=='__main__':main()
