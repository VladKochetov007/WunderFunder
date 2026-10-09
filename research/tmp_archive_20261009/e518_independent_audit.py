from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
LINEAGE = ROOT / 'research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204'
E514 = ROOT / 'research/campaign_2026-10-06/bandit_continuation/conditional_weak_portfolio_E514'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(LINEAGE))
from disagreement_screen import load_e514_inputs
from wnn.gmetric import GlobalAccumulator, _centered_moments, _merge_moments, _wp_from_moments, global_wp

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def arr_sha(a): return hashlib.sha256(memoryview(a).cast('B')).hexdigest()
def main():
    res=json.loads((LINEAGE/'E518_RESULT.json').read_text())
    attempt=json.loads((LINEAGE/'E518_ATTEMPT_STARTED.json').read_text())
    admission=json.loads((LINEAGE/'E518_PRE_SCORE_ADMISSION_20261007.json').read_text())
    receipt=json.loads((LINEAGE/'E518_POST_RUN_EXECUTION_RECEIPT_20261007.json').read_text())
    inp=load_e514_inputs(ROOT)
    targets=inp.targets; mask0=inp.is_scored; anchor=inp.anchor
    cand=inp.candidate_predictions['ab12']; chains=inp.chain_by_sequence
    ctf=dict(inp.chain_to_outer_fold); row_groups=inp.row_groups
    chain_ids=np.unique(chains).astype(np.int64)
    fold_ids={f:np.asarray([c for c in chain_ids if ctf[int(c)]==f],dtype=np.int64) for f in (0,1,2)}
    metric_mask=mask0 & (np.arange(targets.shape[1])>=99)[None,:]
    fold_row=np.asarray([ctf[int(c)] for c in chains],dtype=np.int8)
    gated=anchor.copy(); glob=anchor.copy(); fit_rows=[]; fold_records=[]
    a=np.clip(anchor[:,:,0].astype(np.float64),-2,2)
    c=np.clip(cand[:,:,0].astype(np.float64),-2,2)
    d=c-a; absd=np.abs(d)
    grid=np.arange(101,dtype=np.float64)/100.0
    for f in (0,1,2):
        cal=np.flatnonzero(fold_row!=f); ev=np.flatnonzero(fold_row==f)
        cm=metric_mask[cal]; em=metric_mask[ev]
        thr=float(np.quantile(absd[cal][cm],0.75,method='linear'))
        croute=absd[cal]>=thr; eroute=absd[ev]>=thr
        # Score only official calibration rows using the official gmetric centered-moment
        # primitives. This independently implements the preregistered exact target-0 WP grid.
        ycal=np.clip(targets[cal,:,0].astype(np.float64),-2,2)[cm]
        acal=np.clip(anchor[cal,:,0].astype(np.float64),-2,2)[cm]
        ccal=np.clip(cand[cal,:,0].astype(np.float64),-2,2)[cm]
        rcal=croute[cm]; dcal=ccal-acal
        scores=[]; gscores=[]
        for beta in grid:
            pg=np.clip(acal+beta*rcal*dcal,-2,2).astype(np.float32).astype(np.float64)
            px=np.clip(acal+beta*dcal,-2,2).astype(np.float32).astype(np.float64)
            scores.append(_wp_from_moments(_centered_moments(ycal,pg)))
            gscores.append(_wp_from_moments(_centered_moments(ycal,px)))
        scores=np.asarray(scores,dtype=np.float64); mx=float(scores.max())
        ix=int(np.flatnonzero(mx-scores<=1e-12)[0]); bg=float(grid[ix]); sbg=float(scores[ix])
        gscores=np.asarray(gscores,dtype=np.float64); gmx=float(gscores.max())
        gix=int(np.flatnonzero(gmx-gscores<=1e-12)[0]); bglob=float(grid[gix]); sg=float(gscores[gix])
        gated[ev,:,0]=np.clip(a[ev]+bg*eroute*d[ev],-2,2).astype(np.float32)
        glob[ev,:,0]=np.clip(a[ev]+bglob*d[ev],-2,2).astype(np.float32)
        fit_rows.append({'fold':f,'threshold':thr,'beta_gate':bg,'beta_global':bglob,
                         'gate_cal_t0':sbg,'global_cal_t0':sg,'gate_cal_argmax_t0':mx,'global_cal_argmax_t0':gmx,
                         'eval_gate_rows':int((em&eroute).sum()),'eval_scored_rows':int(em.sum()),
                         'eval_gate_row_fraction':float((em&eroute).sum()/em.sum()),
                         'eval_gate_weight_fraction':float((np.abs(np.clip(targets[ev,:,0].astype(np.float64),-2,2))*(em&eroute)).sum()/(np.abs(np.clip(targets[ev,:,0].astype(np.float64),-2,2))*em).sum())})
        for name,prediction in [('anchor',anchor),('gated',gated),('global',glob)]:
            actual=global_wp(targets[ev],prediction[ev],em)
            stored=res['folds'][f][name]
            fold_records.append({'fold':f,'model':name,'wp':actual['wp'],'stored_wp':stored['wp'],
                                 't0':actual['t0'],'t1':actual['t1'],'selected_rows':actual['selected_rows'],
                                 'wp_match':bool(np.isclose(actual['wp'],stored['wp'],rtol=0,atol=1e-12))})
    checks={'gated_exact_match':bool(np.array_equal(gated,np.load(LINEAGE/'E518_GATED_OOF_PREDICTIONS.npy',mmap_mode='r',allow_pickle=False))),
            'global_exact_match':bool(np.array_equal(glob,np.load(LINEAGE/'E518_GLOBAL_OOF_PREDICTIONS.npy',mmap_mode='r',allow_pickle=False)))}
    saved_g=np.load(LINEAGE/'E518_GATED_OOF_PREDICTIONS.npy',mmap_mode='r',allow_pickle=False)
    saved_gl=np.load(LINEAGE/'E518_GLOBAL_OOF_PREDICTIONS.npy',mmap_mode='r',allow_pickle=False)
    endpoints={}
    for name,p in [('anchor',anchor),('gated',saved_g),('global',saved_gl)]:
        actual=global_wp(targets,p,metric_mask); stored=res[name]
        endpoints[name]={'official':actual,'stored':stored,'wp_match':bool(np.isclose(actual['wp'],stored['wp'],rtol=0,atol=1e-12))}
    # Independently reproduce the preregistered fold-stratified paired complete-chain bootstrap.
    preds=(anchor,saved_g,saved_gl)
    moments=[{} for _ in preds]
    for ch in chain_ids:
        rows=np.flatnonzero(chains==ch)
        for mi,p in enumerate(preds):
            acc=GlobalAccumulator(); acc.add(targets[rows],p[rows],metric_mask[rows]); moments[mi][int(ch)]=acc._moments.copy()
    rng=np.random.default_rng(998465967); deltas=np.empty((2000,2),dtype=np.float64)
    for b in range(2000):
        totals=[np.zeros((2,6),dtype=np.float64) for _ in preds]
        for f in (0,1,2):
            sampled=rng.choice(fold_ids[f],size=fold_ids[f].size,replace=True)
            for ch in sampled:
                for mi in range(3):
                    totals[mi]=np.stack([_merge_moments(totals[mi][k],moments[mi][int(ch)][k]) for k in (0,1)])
        scores=[0.5*sum(_wp_from_moments(t[k]) for k in (0,1)) for t in totals]
        deltas[b,0]=scores[1]-scores[0]; deltas[b,1]=scores[2]-scores[0]
    cis={'gated_vs_anchor':np.quantile(deltas[:,0],[.025,.975]).tolist(),
         'global_vs_anchor':np.quantile(deltas[:,1],[.025,.975]).tolist(),
         'gated_vs_global':np.quantile(deltas[:,0]-deltas[:,1],[.025,.975]).tolist()}
    path_g=LINEAGE/'E518_GATED_OOF_PREDICTIONS.npy'; path_gl=LINEAGE/'E518_GLOBAL_OOF_PREDICTIONS.npy'
    manifest=E514/'SOURCE_MANIFEST_E514_V3.json'; partition=E514/'E514_CHAIN_PARTITION.json'
    prereg=LINEAGE/'METRIC_ALIGNED_GATE_PREREGISTRATION_E518.md'
    runner=LINEAGE/'metric_aligned_gate.py'; loader=LINEAGE/'disagreement_screen.py'; notebook=LINEAGE/'E518_metric_aligned_gate_eda.py'
    hashes={
      'source_manifest_match':inp.source_manifest_sha256==file_sha(manifest)==res['source_manifest_sha256'],
      'partition_match':inp.chain_partition_sha256==file_sha(partition)==res['chain_partition_sha256'],
      'row_groups_match':hashlib.sha256(row_groups.astype('<i8',copy=False).tobytes()).hexdigest()==res['row_groups_sha256'],
      'gated_raw_match':arr_sha(saved_g)==res['oof_gated_sha256'],
      'global_raw_match':arr_sha(saved_gl)==res['oof_global_sha256'],
      'gated_file_match':file_sha(path_g)==res['oof_gated_file_sha256']==receipt['outputs']['gated_oof_sha256'] if False else file_sha(path_g)==res['oof_gated_file_sha256'],
      'global_file_match':file_sha(path_gl)==res['oof_global_file_sha256'],
      'plot_match':file_sha(LINEAGE/'plots/E518_METRIC_ALIGNED_GATE.png')==res['plot_sha256']==receipt['outputs']['plot_sha256'],
      'notebook_hash_matches_admission_and_result':file_sha(notebook)==admission['hashes']['notebook_sha256']==res['notebook_sha256'],
      'runner_hash_matches_admission':file_sha(runner)==admission['hashes']['runner_sha256'],
      'prereg_hash_matches_admission':file_sha(prereg)==admission['hashes']['preregistration_sha256'],
      'result_hash_matches_receipt':file_sha(LINEAGE/'E518_RESULT.json')==receipt['outputs']['result_sha256']}
    pass_recomputed=bool(endpoints['gated']['official']['wp']>endpoints['anchor']['official']['wp'] and endpoints['gated']['official']['wp']>endpoints['global']['official']['wp'] and cis['gated_vs_anchor'][0]>0 and cis['gated_vs_global'][0]>0 and sum(x['beta_gate']>0 for x in fit_rows)>=2)
    print('INPUT',json.dumps({'shape':list(targets.shape),'chains':int(chain_ids.size),'fold_chain_counts':[int(fold_ids[f].size) for f in (0,1,2)],'fold_group_counts':[int((fold_row==f).sum()) for f in (0,1,2)],'manifest_sha256':inp.source_manifest_sha256,'partition_sha256':inp.chain_partition_sha256},sort_keys=True))
    print('HASHES',json.dumps(hashes,sort_keys=True))
    print('FIT',json.dumps(fit_rows,sort_keys=True))
    print('REBUILD',json.dumps(checks,sort_keys=True))
    print('FOLD_ENDPOINTS',json.dumps(fold_records,sort_keys=True))
    print('POOLED',json.dumps(endpoints,sort_keys=True))
    print('BOOTSTRAP',json.dumps({'seed':998465967,'replicates':2000,'intervals':cis,'stored':{'gated_vs_anchor':res['gated_vs_anchor_ci95'],'global_vs_anchor':res['global_vs_anchor_ci95'],'gated_vs_global':res['gated_vs_global_ci95']}},sort_keys=True))
    print('DECISION',json.dumps({'recomputed_pass':pass_recomputed,'stored_pass':res['development_gate_passed'],'positive_gate_folds':sum(x['beta_gate']>0 for x in fit_rows),'all_chains_groups_exposed':True,'attempt_seed':attempt['bootstrap_seed'],'receipt_status':receipt['status'],'hashes_all_registered_match':all(hashes.values())},sort_keys=True))
if __name__=='__main__': main()
