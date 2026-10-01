# Generated-only checkpoint adapter preparation during fixed training

Purpose: make eventual trained native drift measurement executable if the fixed forecast screen is useful. This is source/format integration, not another forecasting configuration, checkpoint selection, serving-speed sweep or a quality readout. Current training/allocation/evaluator files and old readiness/archive sources remain unchanged.

One adapter accepts the declared corrected minGRU211/256x2 state, trims the auxiliary23-output head to the first2 serving targets, packs the existing frozen native-v2 ABI, and exposes reset/raw-step/chunk state behavior. Use the already compiled library from readiness/native-v2/generated-01; do not change or recompile kernels. Check layout/output identity against the old generated benchmark wrapper and preserved generated weights/input only. No training dataset or trained checkpoint is read in this preparation.

One bounded actor writes adapter and generated check; root reviews and owns execution. One generated CPU check <=30seconds,one affinity-pinned core outside current trainer affinity,host<=4GiB,swap0,CUDA hidden. Preserve sources,command,logs,result and any failure. No scientific retries automatically. No GPU,forecast fit,OOF,metric,external publication or package modification. Actual trained native inference remains unallocated until the fixed OOF result is assessed.

Passing confirms the adapter layout/stream semantics on the tested generated fixtures only. It does not qualify trained quality or complete submission runtime. Local and SSH preservation included in current campaign collector.
