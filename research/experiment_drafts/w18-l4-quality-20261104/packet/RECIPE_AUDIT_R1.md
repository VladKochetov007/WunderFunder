# Independent Recipe Gate and CPU Preflight R1

Date: 2026-11-04. Disposition: **admit active FROZEN_PANEL_R3 to bounded profiling after root's actual Linux byte-binding gate**. This review covers source/recipe semantics and local CPU preflight. No source edits, SSH, GPU use, fits, scoring, or science approval were performed.

## Decision

- I parsed R2 and R3 and compared every cell's train_args, run_name, and seed: all eight are exact matches. R3 changes the source binding only. Data/split/member identity, endpoint and numerical recipes remain fixed. R1/R2 are unexecuted historical panels.
- All eight CLIs use train source, 10,325 fit groups, full official valid evaluation, fixed train-only FBC seed-5 epoch-3 initializer, two epochs and native --batch-order seeded. Seeds 1001/1002 are paired across control and each arm. Shared fields match after removing run name, seed, and the declared arm fields.
- --batch-order seeded is accepted natively. The loader uses EpochOrderSampler keyed by seed and epoch, so changed feature width no longer shifts sequence order through global Torch RNG consumption. This is a new matched cohort recipe, not bitwise replay of historical W18 with default random order.
- Ordinary train-source membership sets extra_valid=None; fit_preprocessing passes this to fit_research_feature_prior. That helper accepts a nonempty validation list if miswired, so the run-time provenance gate must assert serialized prior fit members contain train only.
- Causal innovations adds 80 columns to the 211-column W18 input (291 total). Warm-start expansion uses zero new input weights and mean-zero/std-one normalization, preserving the parent function at initialization.
- Alpha .5 changes the main blend from 30% label / 70% teacher to 50% / 50%. Family weight .01 stays positive and retains the same output shape.
- Arm A has an initialization nuisance: eight new family-output rows draw from the current model-construction RNG state; the wider input projection changes RNG consumption before those rows are created. Interpret A as a paired two-seed conditional quality screen, not a clean causal estimate of feature effect. B/C are cleaner one-factor contrasts.

## Data, baseline, and evaluator

The separate BASELINE_AUDIT_R1 admits cached train-only W18 and TL3 controls after exact chronological fit membership, checkpoint/teacher ancestry, actual-byte closure and full-valid metric checks. The official boundary has 10,325 eligible fit groups, 1,873 valid sequences and 3,528,273 scored valid rows. The strongest eligible fixed historical reference is the canonical TL3 blend at pooled WP 0.6923774245652725; its W18 member is 0.6914165059769162. These are exposed validation results, not fresh confirmation or seed-matched controls. The baseline packet contains exact hashes, score components, W18 argv, ancestry closure and the qualified scratch fallback.

The 10,325 fit count excludes 282 train-component-overlap groups from 10,607 physical train sequences. The panel keeps the official chronological manifest, does not substitute the physical-train count, and has no submitted train+valid W18 ancestor. Every fit must score the complete official valid set with the frozen evaluator.

## Source trace

Local review HEAD f1c4f07c99328d6847e2f3e3864120688d45c99b and root-reported staged source commit 4c57bee9227866c85407e50852973a70f53bf484 have identical reviewed recipe files (git diff --stat over these paths was empty).

- src/wnn/training/data_membership.py:37,138: train-only extra_valid and chronological fit assertion.
- src/wnn/training/setup_preprocessing.py:49; src/wnn/training/inputs.py:187: prior call and optional validation-member argument.
- src/wnn/research_features.py:42,338,366: 80 feature names and causal scan carrying state over 512-row internal blocks.
- src/wnn/training/batches.py:70; src/wnn/training/evaluation.py:33-35: full-sequence feature construction before recurrent chunk slicing.
- src/wnn/training/inheritance.py:94; src/wnn/training/setup_model.py:135-201: append-only expansion and zero/0/1 initialization.
- src/wnn/training_dynamics.py:31; src/wnn/training/data_loaders.py:123-130; src/wnn/training_sampling.py:64-96; src/wnn/training/epoch.py:35-42: accepted seeded sampler.
- src/wnn/training/losses.py:74,94; src/wnn/training/setup_model.py:108-112: alpha equation, positive family loss, and output-head construction.
- src/wnn/distillation.py:123-150: family rows initialized from current model RNG state, establishing the arm-A nuisance.

## CPU commands and results

Environment: C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031\.venv\Scripts\python.exe, Python 3.13.15, Torch 2.14, CPU only, no CUDA calls.

- Production membership/prior synthetic guard: PASS; 40 synthetic train IDs, five valid IDs, extra_valid=None, only train parquet opened, exactly 32 deterministic train-only prior members.
- Innovation extension: PASS; 211-to-291 input, 80 new weights zero, 80 new means zero and stds one; parent output preserved at rtol=atol=1e-6 for arbitrary added features.
- Existing causal feature suite: 8 passed / 1 skipped. The real 20,000-row scan skipped because local datasets/*.parquet is absent. Windows lacks Unix resource, imported by trainer profiling; the test command injects an in-memory stub for collection only.
- Frozen panel lint: PASS; eight cells, identical R2/R3 CLI identities, same parent/endpoint/membership, only declared mechanisms and paired seeds.

### Existing causal-feature suite

Run from the worktree root:

~~~powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031\.venv\Scripts\python.exe' -c "import sys,types; s=types.ModuleType('resource'); s.RUSAGE_SELF=0; s.getrusage=lambda _:types.SimpleNamespace(ru_maxrss=0); sys.modules['resource']=s; import pytest; raise SystemExit(pytest.main(['-p','no:cacheprovider','tests/test_causal_moment_features.py']))"
~~~

Observed: 8 passed, 1 skipped in 7.57s; skip reason: local datasets/*.parquet absent.

### Synthetic train-only prior and innovation extension test

This was executed from the worktree root using the same venv. It uses synthetic rows, not official data or receiving-node bytes.

~~~python
from pathlib import Path
from types import SimpleNamespace
import sys, types
import numpy as np
import torch
shim = types.ModuleType('resource')
shim.RUSAGE_SELF = 0
shim.getrusage = lambda _: SimpleNamespace(ru_maxrss=0)
sys.modules['resource'] = shim
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.data_membership import select_membership
from wnn.training.inputs import fit_research_feature_prior
from wnn.training.inheritance import expand_research_feature_state, can_extend_research_features
from wnn.nn import RNNModel

train_groups = np.arange(100, 140, dtype=np.int64)
valid_groups = np.arange(500, 505, dtype=np.int64)
class Split:
    schema = CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
    train_groups = train_groups
    valid_groups = valid_groups
    evaluation_groups = np.arange(600, 605, dtype=np.int64)
    def assert_fit_groups(self, train, valid):
        assert set(map(int, train)) == set(map(int, self.train_groups))
        assert len(valid) == 0, 'validation fitting membership was populated'
class FakeSequenceFile:
    def __init__(self):
        self.calls = []
        self.rows = np.random.default_rng(12).normal(size=(1024, 112)).astype(np.float32)
    def read(self, group, *, targets, mask):
        assert targets is False and mask is False
        assert int(group) in set(map(int, train_groups)), f'non-train group read: {group}'
        self.calls.append(int(group))
        return SimpleNamespace(features=self.rows)
reader = FakeSequenceFile()
class Env:
    root = Path('C:/fake-tabdata')
    corrupt_train_groups = set()
    def sequence_file(self, path):
        assert path.name == 'train.parquet', f'non-training parquet opened: {path}'
        return reader
env = Env()
args = SimpleNamespace(valid_fold=-1, train_source='train', include_heldout_valid=False,
                       eval_source='dev', full_valid=True, valid_seqs=32,
                       valid_train_fraction=1.0, fit_subset=None, train_classes='',
                       train_seqs=40, seed=1001, class_balance=False)
data = SimpleNamespace(campaign_split=Split(), eligible_valid_count=0, extra_valid=None,
                        fit_subset=None, final_fit_membership=None, inner_crossfit=None,
                        n_valid=0, sampler=None, train_idx=np.empty(0, dtype=np.int64),
                        train_split='train', valid_idx=np.empty(0, dtype=np.int64))
run = SimpleNamespace(environment=env, options=SimpleNamespace(args=args), data=data)
select_membership(run)
assert data.extra_valid is None
prior = fit_research_feature_prior(data.train_split, data.train_idx,
                                   None if data.extra_valid is None else np.asarray(data.extra_valid),
                                   args.seed, environment=env)
assert len(prior['fit_members']) == 32
assert all(item['split'] == 'train' and item['row_group'] in set(train_groups)
           for item in prior['fit_members'])
assert len(reader.calls) == 32 and len(set(reader.calls)) == 32
print('prior_membership_guard=PASS train_fit_groups=40 sampled_prior_groups=32 valid_opened=0')

parent = RNNModel(211, hidden=8, layers=1).eval()
child = RNNModel(291, hidden=8, layers=1).eval()
assert can_extend_research_features('none', 'causal_innovations')
expanded = expand_research_feature_state(parent.state_dict(), child.state_dict(),
                                         'none', 'causal_innovations', 211)
child.load_state_dict(expanded)
assert torch.count_nonzero(child.mean[211:]).item() == 0
assert torch.all(child.std[211:] == 1)
assert torch.count_nonzero(child.inp.weight[:, 211:]).item() == 0
base = torch.randn(2, 24, 211)
extra = torch.randn(2, 24, 80)
with torch.no_grad():
    p, _ = parent(base)
    c, _ = child(torch.cat((base, extra), dim=-1))
torch.testing.assert_close(c, p, rtol=1e-6, atol=1e-6)
print('innovation_extension=PASS parent_width=211 child_width=291 zero_input_weights=80 zero_mean_unit_std=80 parent_function_preserved=1e-6')
~~~

Observed:

    prior_membership_guard=PASS train_fit_groups=40 sampled_prior_groups=32 valid_opened=0
    innovation_extension=PASS parent_width=211 child_width=291 zero_input_weights=80 zero_mean_unit_std=80 parent_function_preserved=1e-6

## Recommendation

Proceed with R3 profiling only after root verifies the 315-file Linux numeric-source binding and four execution-controller hashes against the actual node checkout. Keep a run-time assertion rejecting any valid group in the serialized arm-A prior. Then run the fixed eight recipes and score every official valid row. Report paired per-seed conditional screen results and preserve the arm-A initialization nuisance in interpretation. Historical valid metrics remain development references.
