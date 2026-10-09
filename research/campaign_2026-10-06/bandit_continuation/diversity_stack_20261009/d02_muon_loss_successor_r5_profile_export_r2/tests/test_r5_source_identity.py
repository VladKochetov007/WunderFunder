from __future__ import annotations

from pathlib import Path


def test_r5_bootstrap_and_worker_bind_the_r5_source_dataset() -> None:
    """Keep the sealed input search and emitted receipt on the same dataset."""
    root = Path(__file__).resolve().parents[1]
    expected = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r5"
    template = (root / "kernel_template.py").read_text(encoding="utf-8")
    worker = (root / "code/scripts/loss_pair_gpu_worker.py").read_text(encoding="utf-8")
    sealer = (root / "seal_d02_candidate_r5.py").read_text(encoding="utf-8")

    assert f'SOURCE_DATASET = "{expected}"' in template
    assert f'SOURCE_DATASET = "{expected}"' in worker
    assert f'SOURCE_DATASET_ID = "{expected}"' in sealer
    assert "code-20261009-r4" not in template
    assert "code-20261009-r4" not in worker
