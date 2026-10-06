"""Generated-only independent metric/aggregation controls. No study arrays loaded."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ensemble_backbone_evaluate',HERE/'evaluate.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
fs=importlib.util.spec_from_file_location('canonical_fold_eval',E.ROOT/'scripts/fold_eval.py')
F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F)


class GeneratedMetricChecks(unittest.TestCase):
    def test_sums_match_frozen_independent_implementation(self):
        rng=np.random.default_rng(415)
        y=rng.normal(size=(400,2))*2
        p=rng.normal(size=(400,2))*.8+.7*y
        take=rng.random(400)>.2
        own=E.sufficient(y[take],p[take])
        oracle=F.sums(y[None],p[None],take[None])[0]
        np.testing.assert_allclose(own,oracle,rtol=1e-13,atol=1e-12)
        self.assertAlmostEqual(float(E.metric(own)['WP']),F.pooled(oracle),places=13)
        yc,pc=np.clip(y[take],-2,2),np.clip(p[take],-2,2)
        direct=[]
        for t in (0,1):
            w=abs(yc[:,t]);dy=yc[:,t]-np.average(yc[:,t],weights=w);dp=pc[:,t]-np.average(pc[:,t],weights=w)
            direct.append(np.dot(w,dy*dp)/np.sqrt(np.dot(w,dy*dy)*np.dot(w,dp*dp)))
        np.testing.assert_allclose(E.metric(own)['rho'],direct,rtol=1e-12,atol=1e-12)

    def test_literal_blend_and_clip_order(self):
        rng=np.random.default_rng(76)
        parts=[rng.normal(size=(100,2)).astype(np.float32) for _ in range(4)]
        np.testing.assert_array_equal(E.blend(parts),(((parts[0]+parts[1])+parts[2])+parts[3])/4)
        parts=[np.full((3,2),v,np.float32) for v in (8,8,-8,0)]
        np.testing.assert_array_equal(E.blend(parts),np.full((3,2),2,np.float32))
        wrong=np.mean([np.clip(p,-2,2) for p in parts],axis=0)
        self.assertTrue(np.all(wrong!=E.blend(parts)))

    def test_zero_and_constant_target_conventions(self):
        rng=np.random.default_rng(5)
        p=rng.normal(size=(100,2))
        for y in (np.zeros_like(p),np.ones_like(p)):
            s=E.sufficient(y,p)
            self.assertEqual(E.metric(s)['WP'],0)
            self.assertFalse(E.metric(s)['valid'].any())
            self.assertEqual(F.pooled(s),0)
        self.assertEqual(E.metric(E.sufficient(p[:0],p[:0]))['WP'],0)

    def test_whole_chain_paired_bootstrap_and_pooling(self):
        rng=np.random.default_rng(840)
        keys=np.array([(f,100*f+j,j%4) for f in (0,1) for j in range(12)])
        stats=[];counts=[]
        for f,c,cl in keys:
            n=50+c%17
            y=rng.normal(size=(n,2))+(1 if f else -1)
            p=.5*y+rng.normal(size=(n,2))*.5
            s=E.sufficient(y,p)
            stats.append(np.stack([s,s]));counts.append(n)
        stats=np.asarray(stats);counts=np.asarray(counts)
        draw,strata=E.resamples(keys,n=50)
        for f in (0,1):
            for cl in range(4):
                k=(keys[:,0]==f)&(keys[:,2]==cl)
                np.testing.assert_array_equal(draw[:,k].sum(1),np.full(50,k.sum()))
        out=E.comparison(stats,counts,draw)
        self.assertEqual(out['delta_WP'],0)
        np.testing.assert_array_equal(out['paired_chain_CI']['interval95'],[0,0])
        self.assertAlmostEqual(out['control']['WP'],F.pooled(stats[:,1].sum(0)),places=13)
        pooled=out['control']['WP']
        averaged=np.mean([F.pooled(stats[k,1].sum(0)) for k in (keys[:,0]==0,keys[:,0]==1)])
        self.assertGreater(abs(pooled-averaged),.01)

    def test_primary_and_operational_are_prespecified_distinct_contrasts(self):
        rng=np.random.default_rng(119)
        keys=np.array([(f,100*f+j,j%4) for f in (0,1) for j in range(8)])
        stats=[];counts=[]
        for f,c,cl in keys:
            n=40+c%7
            y=rng.normal(size=(n,2))
            # All three fixed profiles share rows. No separate bootstrap can turn
            # the secondary reference into a more favorable selectable control.
            noise=rng.normal(size=(n,2))
            ps=[y+.5*noise,y+.8*noise,y+1.1*noise]
            ss=np.stack([E.sufficient(y,p) for p in ps])
            stats.append(np.stack([ss,ss]));counts.append([n,n])
        stats=np.array(stats);counts=np.array(counts)
        out=E.describe(stats,counts,keys,bootstrap=70)
        self.assertEqual(list(out['comparisons']),['primary','operational'])
        sums=stats[:,0].sum(0)
        for label,control in [('primary',1),('operational',2)]:
            r=out['comparisons'][label]['scopes']['all']
            self.assertAlmostEqual(r['delta_WP'],F.pooled(sums[0])-F.pooled(sums[control]),places=13)
            self.assertEqual(r['paired_chain_CI']['valid_draws'],70)
            draw,_=E.resamples(keys,n=70)
            oracle=[]
            for d in draw:
                repeats=np.repeat(np.arange(len(keys)),d)
                q=stats[repeats,0].sum(0)
                oracle.append(F.pooled(q[0])-F.pooled(q[control]))
            np.testing.assert_allclose(r['paired_chain_CI']['interval95'],np.quantile(oracle,[.025,.975]),atol=2e-14,rtol=0)
        self.assertEqual(out['comparisons']['primary']['control'],'fresh_control4')
        self.assertEqual(out['comparisons']['operational']['control'],'current_student4')

    def test_degenerate_targets_do_not_create_false_valid_intervals(self):
        y=np.array([[-1,0],[1,0],[-.5,0],[.8,0]],float)
        p=np.array([[-.8,.4],[.9,-.3],[-.4,.1],[.6,.2]],float)
        s=E.sufficient(y,p)
        stats=np.tile(s[None,None],(2,2,1,1))
        draws=np.array([[2,0],[0,2],[1,1]],np.int16)
        r=E.comparison(stats,np.array([4,4]),draws)
        self.assertEqual(r['paired_chain_CI'],{'valid_draws':0,'interval95':None})
        self.assertEqual(r['paired_target_CI'][0]['valid_draws'],3)
        self.assertEqual(r['paired_target_CI'][1],{'valid_draws':0,'interval95':None})
        np.testing.assert_array_equal(r['paired_target_CI'][0]['interval95'],[0,0])


if __name__=='__main__':unittest.main(verbosity=2)
