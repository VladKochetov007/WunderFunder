"""Generated metadata only: interruption is not permission to repeat or alter a job."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('resumed_binding_builder',HERE/'build_resumed_bindings.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)


def fixture():
    old=M.B+'/execution-01';new=M.B+'/execution-02';jobs=[]
    for f in (0,1):
        for arm in ('control','candidate'):
            for stage in ('parent','H2','OOF'):
                name=f'eb_{arm}_s3_{"parent" if stage=="parent" else "h2"}_f{f}'
                j={'name':name,'stage':stage,'argv':['python','fixed_script',name,stage],
                   'timeout_seconds':3600 if stage=='parent' else 1200,
                   'required_outputs':[f'artifacts/runs/{name}/fixed.pt'],
                   'require_absent':[f'artifacts/runs/{name}']}
                if stage!='OOF':
                    check=f'{old}/{name}.check.json'
                    j['post_argv']=['python','fixed_checker',name,'--output',check]
                    j['required_outputs'].append(check);j['require_absent'].append(check)
                jobs.append(j)
    first={'jobs':jobs,'execution_directory':old,'success_status':M.SUCCESS}
    remaining=[]
    for j in jobs[5:]:
        j=copy.deepcopy(j)
        for field in ('post_argv','required_outputs','require_absent'):
            if field in j:j[field]=M.remap(j[field],old,new)
        remaining.append(j)
    second={'jobs':remaining,'execution_directory':new,'success_status':M.SUFFIX_SUCCESS}
    def done(selected,status,nfit,noof,failure):
        return {'jobs':[dict(name=j['name'],stage=j['stage'],exit_code=0,failure=None,remaining_scope_pids=[]) for j in selected],
                'status':status,'failure':failure,'fits':nfit,'OOF_jobs':noof,'metric_calls':0,'retries':0}
    stopped=done(jobs[:5],'FAILED_STOP_NO_RETRY',4,1,M.ORIGINAL_FAILURE)
    finished=done(remaining,M.SUFFIX_SUCCESS,4,3,None)
    return first,stopped,second,finished


class ResumptionChecks(unittest.TestCase):
    def test_exact_prefix_suffix_coverage(self):
        args=fixture();result=M.validate_segments(*args)
        self.assertEqual(list(map(M.key,result)),list(map(M.key,args[0]['jobs'])))
        self.assertEqual(len(set(map(M.key,result))),12)

    def test_only_receipt_destinations_change(self):
        a=fixture();a[2]['jobs'][1]['argv'].append('--different-learning-rate')
        with self.assertRaises(AssertionError):M.validate_segments(*a)
        a=fixture();a[2]['jobs'][1]['post_argv'][1]='different_checker'
        with self.assertRaises(AssertionError):M.validate_segments(*a)
        a=fixture();a[2]['jobs'][0]['timeout_seconds']+=1
        with self.assertRaises(AssertionError):M.validate_segments(*a)

    def test_repeated_or_missing_job_rejected(self):
        a=fixture();a[3]['jobs'][0]=copy.deepcopy(a[1]['jobs'][-1])
        with self.assertRaises(AssertionError):M.validate_segments(*a)
        a=fixture();a[3]['jobs'].pop()
        with self.assertRaises(AssertionError):M.validate_segments(*a)
        a=fixture();a[3]['jobs'][1],a[3]['jobs'][2]=a[3]['jobs'][2],a[3]['jobs'][1]
        with self.assertRaises(AssertionError):M.validate_segments(*a)

    def test_scientific_failure_active_process_or_retry_rejected(self):
        for field,value in [('exit_code',1),('failure','OOM'),('remaining_scope_pids',[123])]:
            a=fixture();a[1]['jobs'][-1][field]=value
            with self.assertRaises(AssertionError):M.validate_segments(*a)
        for which in (1,3):
            for field in ('retries','metric_calls'):
                a=fixture();a[which][field]=1
                with self.assertRaises(AssertionError):M.validate_segments(*a)

    def test_different_interruption_not_silently_reclassified(self):
        a=fixture();a[1]['failure']='arbitrary failed training step'
        with self.assertRaises(AssertionError):M.validate_segments(*a)
        a=fixture();a[1]['status']=M.SUCCESS
        with self.assertRaises(AssertionError):M.validate_segments(*a)

    def test_frozen_sources_cannot_be_replaced(self):
        name=M.B+'/evaluate.py';base={name:'a'*64}
        self.assertEqual(M.merge_sources(base,dict(base),{M.B+'/new_receipt_builder.py':'b'*64})[name],'a'*64)
        with self.assertRaises(AssertionError):M.merge_sources(base,{name:'b'*64},{})
        with self.assertRaises(AssertionError):M.merge_sources(base,base,{'src/wnn/new_science.py':'b'*64})


if __name__=='__main__':unittest.main(verbosity=2)
