"""Generated-only bridge check; root allocates CPU before executing this file."""
import argparse, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import torch
from native_adapter import NativeMinimal, pack_state, sha, NATIVE, LIB_SHA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(False)
    start=time.monotonic();data=NATIVE/'generated-01'
    inventory=data/'generated-before.json'
    assert sha(inventory)=='d3a5de284973d2d5712b0eac9ac2f9668bb14c42518b910f023a50b579be58fd'
    records=json.loads(inventory.read_text())
    for name in ('min23.npz','min2.npz','generated-inputs.npz'):
        assert sha(data/name)==records[name]['sha256']
    s23=dict(np.load(data/'min23.npz'));s2=dict(np.load(data/'min2.npz'))
    flat=pack_state(s23,23);np.testing.assert_array_equal(flat,pack_state(s2,2))
    benchmark=NATIVE/'check_benchmark.py'
    assert sha(benchmark)=='4c3a8a6fdd9cce0d4a99c8a32649160a03a6e604c923b86ce9eb8849a0781ecb'
    spec=importlib.util.spec_from_file_location('original_generated_benchmark',benchmark)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    model=old.MinimalRNN(n_out=2).eval()
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in s2.items()},strict=True)
    np.testing.assert_array_equal(flat,old.pack(model,'min'))
    _,lib=old.libsetup(data);reference=old.Min(lib,model,'optimized')
    raw=np.load(data/'generated-inputs.npz')['raw'];checks=['O23_trim_equals_O2','flat_order_equals_original']
    try:
        with NativeMinimal(flat) as m:
            reference.reset();want=reference.run(raw,True);got=m.run(raw)
            np.testing.assert_array_equal(got,want);np.testing.assert_array_equal(m.state(),reference.state())
            checks.append('original_wrapper_output_and_state_bitexact')
            end=m.state();m.reset()
            cuts=[0,1,63,64,511,1999,2000,len(raw)]
            pieces=[m.run(raw[lo:hi]) for lo,hi in zip(cuts[:-1],cuts[1:])]
            np.testing.assert_array_equal(np.concatenate(pieces),got);np.testing.assert_array_equal(m.state(),end)
            checks.append('raw_chunk_carry_bitexact');m.reset()
            np.testing.assert_array_equal(m.run(raw),got);checks.append('raw_reset_bitexact')
            changed=raw.copy();changed[511:]+=1;m.reset()
            np.testing.assert_array_equal(m.run(changed)[:511],got[:511]);checks.append('raw_prefix_bitexact')
            for bad in (raw[:2].astype('float64'),raw[:2,:111],np.full((1,112),np.nan,np.float32)):
                try:m.run(bad)
                except ValueError:pass
                else:raise AssertionError('Invalid input accepted')
            checks.append('dtype_shape_finite_rejections')
    finally:reference.close()
    result=dict(status='PASS_GENERATED_ADAPTER_ONLY',checks=checks,rows=len(raw),seconds=time.monotonic()-start,
                optimized_sha256=LIB_SHA,adapter_sha256=sha(Path(__file__).with_name('native_adapter.py')),
                check_sha256=sha(__file__),generated_inventory_sha256=sha(inventory),
                trained_checkpoint_loaded=False,forecast_data=False,GPU=False,quality_claim=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
