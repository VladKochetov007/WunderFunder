"""Small generated controls for the independent replay analysis only."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

S=importlib.util.spec_from_file_location('context_analysis',Path(__file__).with_name('analysis.py'))
A=importlib.util.module_from_spec(S);S.loader.exec_module(A)


class GeneratedChecks(unittest.TestCase):
    def test_original_float32_blend_order_and_postblend_clipping(self):
        rng=np.random.default_rng(81)
        m=rng.normal(size=(100,2)).astype(np.float32)
        partners=rng.normal(size=(100,3,2)).astype(np.float32)
        expected=sum([m,partners[:,0],partners[:,1],partners[:,2]])/4
        np.testing.assert_array_equal(A.blend(m,partners),expected)
        p=np.array([[[8.,8.],[-8.,-8.],[0.,0.]]],np.float32)
        out=A.blend(np.array([[8.,8.]],np.float32),p)
        np.testing.assert_array_equal(out,[[2.,2.]])

    def test_pooled_statistic_is_not_mean_group_score(self):
        rng=np.random.default_rng(72)
        y=rng.normal(size=(300,2))
        p=.5*y+rng.normal(size=y.shape)
        group=np.r_[np.zeros(40,int),np.ones(260,int)]
        y[:40]+=4;p[:40]-=3
        s=A.sums_by_group(y,p[:,None],group,2)
        pooled=A.metric(s.sum(0))['score'][0]
        yc,pc=np.clip(y,-2,2),np.clip(p,-2,2)
        rho=[]
        for t in (0,1):
            w=abs(yc[:,t]);my=np.average(yc[:,t],weights=w);mp=np.average(pc[:,t],weights=w)
            dy,dp=yc[:,t]-my,pc[:,t]-mp
            rho.append(np.dot(w,dy*dp)/np.sqrt(np.dot(w,dy*dy)*np.dot(w,dp*dp)))
        self.assertAlmostEqual(pooled,np.mean(rho),places=13)
        self.assertGreater(abs(pooled-A.metric(s)['score'].mean()),.01)

    def test_reference_energy_and_postclip_gradient(self):
        rng=np.random.default_rng(32)
        y=rng.uniform(-1.4,1.4,size=(500,2))
        p=.5*y+rng.normal(0,.3,size=y.shape)
        direction=rng.normal(0,.1,size=y.shape)
        eps=1e-5
        pp=np.stack([p,p+eps*direction,p-eps*direction],axis=1)
        ss=A.sums_by_group(y,pp,np.zeros(len(y),int),1).sum(0)
        par=A.metric(ss[0])
        error,linear,_,_=A.fixed_rows(y,pp,par)
        self.assertAlmostEqual(.25*np.sum(error[:,0].sum(0)/par['W']),1-par['score'],places=12)
        predicted=.5*np.sum(linear[:,1].sum(0)/par['W'])/eps
        exact=(A.metric(ss)['score'][1]-A.metric(ss)['score'][2])/(2*eps)
        self.assertAlmostEqual(predicted,exact,places=8)

    def test_paired_bootstrap_identical_arms_exact_zero(self):
        rng=np.random.default_rng(47)
        n=384
        md=np.zeros((n,7),int)
        md[:,0]=np.repeat(np.arange(12),32)
        md[:,1]=md[:,0]
        md[:,2]=4096+np.tile(np.arange(32),12)
        md[:,3]=md[:,0]
        md[:,4]=md[:,0]%2
        md[:,5]=2+(md[:,0]//2)%2
        md[:,6]=np.tile(np.r_[np.arange(16),64+np.arange(16)],12)
        y=rng.normal(size=(n,2)).astype(np.float32)
        p=(.6*y+rng.normal(0,.4,size=y.shape)).astype(np.float32)
        pp=np.repeat(p[:,None,:],3,axis=1)
        result,arrays=A.analyze(y,pp,md,np.ones(n,bool),bootstrap=30)
        for scope in result['scopes'].values():
            for contrast in scope['contrasts'].values():
                self.assertEqual(contrast['WP_difference'],0)
                np.testing.assert_array_equal(contrast['conditional_WP_CI']['interval95'],[0,0])
                self.assertEqual(contrast['fixed_energy_difference'],0)
                self.assertEqual(contrast['postclip_first_order_WP_contribution'],0)
        self.assertEqual(len(arrays['chain_metadata']),12)


if __name__=='__main__':unittest.main(verbosity=2)
