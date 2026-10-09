from pathlib import Path
import json,hashlib
main=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
work=Path('/home/vlad/development/wunder/worktrees/sol-diversity-delivery-20261009')
base=Path('research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009')
out=work/base/'matrix'; out.mkdir(exist_ok=True)
sources=['ASTRA_PROPOSAL_R1.md','LUNA_LOSS_ARCHITECTURE_R1.md','LUNA_SERVING_COMPUTE_R1.md','FREEBUFF_PUBLIC_PROPOSAL_R1.md','GROK47_PROPOSAL_R1.md','ARTIFACT_RETENTION_R1.md']
source_records=[{'path':str(base/name),'sha256':hashlib.sha256((main/base/name).read_bytes()).hexdigest()} for name in sources]
cards=[
('D01',1,'loss','AdamW','wmse vs huber delta1','OneCycle','GRU256x2; 350 complete training sequences','TPU/GPU matched FP32, then native FP16/BF16','A2; Luna loss1; serving2'),
('D02',2,'optimizer×loss','AdamW vs Muon','wmse vs huber delta1','OneCycle','GRU256x2; same complete component sample','Four endpoints; explicit interaction with D01','A3; Freebuff1; Luna5'),
('D03',3,'activation','matched control','','matched control','GELU vs fixed_mix readout; same GRU','Measure complementarity at fixed initialization/order','Luna6; A3'),
('D04',3,'architecture width','matched control','','matched control','GRU256→384→512→768 teacher ladder','Each rung requires device profile; student compression path','serving capacity; Freebuff1'),
('D05',3,'architecture causal multiscale','matched control','','matched control','GRU vs causal TCN; declared context/receptive field','Teacher first; carry convolution history; serving unresolved','Luna4'),
('D06',3,'architecture alternate recurrent','matched control','','matched control','GRU vs LSTM','Independent init shape policy and serving path audit','serving4'),
('D07',3,'history slow/fast state','matched control','','matched control','Causal slow/fast state teacher or correction','Preserve histories; match compute; no chunk state reset','A4'),
('D08',3,'sampling component bags','matched control','','matched control','Uniform vs fixed complete-component bag/bootstrap','Fixed outer evaluation; same update budget; repeat exposure recorded','A9'),
('D09',3,'sampling stratified order','matched control','','matched control','Complete sequences; input-only strata; no replacement','Contrast order only; class balance replacement is separate','Luna7'),
('D10',4,'sampling capped difficulty specialist','matched control','','matched control','Complete components; frozen OOF difficulty weights','Requires clean fitting-side OOF residuals; no evaluation-based sampling','A9'),
('D11',2,'scheduler cosine','matched control','','OneCycle vs warmup+cosine','Same GRU and component order','Actual LR/beta traces, fixed terminal checkpoint','retention scheduler; ADR0023'),
('D12',2,'scheduler WSD','matched control','','OneCycle vs WSD','Same GRU and component order','Warm/stable/decay budgets explicit; preserve late checkpoints','retention scheduler; ADR0023'),
('D13',3,'scheduler plateau/noise','matched control','','OneCycle vs plateau/noise response','Same GRU and component order','Response uses fit-side signal only; predeclare finite policy','retention scheduler; ADR0023'),
('D14',3,'training target range','matched control','clipped vs raw residual target','matched control','GRU; official clipped-label magnitude weights unchanged','Needs explicit trainer treatment; official eval clip remains2','Luna3'),
('D15',3,'pooled objective','matched control','mix vs wmse_gcorr / gcorr_pool','matched control','GRU; fixed pooled weighting/decay semantics','Separate memory-bearing objective arm; per-component sufficient stats','Luna2'),
('D16',2,'static blend conditional value','','','','Existing ancestry-valid frozen prediction bank','Per-target cross-fitted shrinkage; equal mixture control','A1; Luna8; Freebuff3'),
('D17',3,'causal learned blend/gate','','','','Current-output or causal-prefix gate','Component meta-crossfit; static mixture default/control','A5; Freebuff4; serving8'),
('D18',2,'diverse teacher distillation','matched control','supervised + declared distill','matched control','Student128→256→384; heterogeneous vs anchor-only targets','Own weights; shared ancestry; full teacher masses/clipping explicit','A7; serving5'),
('D19',3,'contrast-preserving distillation','matched control','mean vs common/contrast heads','matched control','Same student trunk; redundant-head control','Fitting-side affine projection/RMS only; compare complete endpoint','A8'),
('D20',3,'neural residual boosting','matched control','MSE vs metric-gradient correction','matched control','One-stage causal correction; fixed small step','Nested OOF targets exclude evaluated components through every ancestor','A6; Freebuff2/5; serving7'),
('D21',3,'teacher family/target mixtures','matched control','target-specific family targets','matched control','Heterogeneous recurrent teachers and two students','Reconstruct own reference teacher stages/seed ancestry/blend first','ADR0023; Freebuff1'),
('D22',2,'fold robustness/scale confirmation','paired original','','original recipe','Fold1/full membership after subset screens','Previously exposed folds remain exposed; not new discovery arms','research objective; subset protocol'),
('D23',3,'serving compression/quantization','','','','Selected student or complete ensemble endpoint','Measure full deterministic CPU replay, reset, native parity and ZIP bytes','serving9'),
]
rows=[]
for ident,p,mechanism,optim,loss,sched,sample,criteria,basis in cards:
 rows.append(dict(experiment_id=ident,priority=p,status='planned_not_admitted',mechanism=mechanism,optim=optim,loss=loss,sched=sched,train_sample=sample,hypothesis=criteria,basis=basis,parent_ids={'D02':['D01'],'D19':['D18'],'D20':['D16'],'D21':['D18'],'D23':['D18']}.get(ident,[]),retention='checkpoint+norm+provenance+fixed float32 NPY/sidecar+rolling full resume state+LR/beta/unscaled-grad/clip/skip trace',evaluation_exposure='exposed; unchanged fixed outer membership',serving_path='student/compression or measured complete endpoint',budget='requires separate root admission',wp=None))
(out/'planned_experiments.json').write_text(json.dumps({'scope':'Curated mechanism contrasts, not Cartesian sweep or compute admission. Full append-only hypothesis/experiment ledgers remain backlog.','sources':source_records,'backlog_ledgers':['research/hypotheses.jsonl','research/experiments.jsonl','research/bandit_tree/journal.jsonl'],'experiments':rows},indent=2)+'\n')
check=main/base/'LUNA_STACKING_ROOT_INDEPENDENT_CHECK_R1.json'; receipt=json.loads(check.read_text()); anchor=receipt['metrics']['P4']['wp']; endpoints=[]
for name,metrics in receipt['metrics'].items():
 endpoints.append(dict(endpoint=name,status='independently_recomputed_fixed_predictions',**metrics,delta_vs_anchor=metrics['wp']-anchor,anchor='P4',panel_sequences=468,panel_components=141,evaluation_exposure='exposed fold0',source=str(check.relative_to(main)),source_sha256=hashlib.sha256(check.read_bytes()).hexdigest(),recipe_note='Z4 here is seed4 fold0 student endpoint; excludes all-data seed99/Z4 submission',limits=receipt['limits'],public_forecast=None))
(out/'reviewed_endpoints.json').write_text(json.dumps({'metric':'global_wp_clip2_v1','endpoints':endpoints},indent=2)+'\n')
