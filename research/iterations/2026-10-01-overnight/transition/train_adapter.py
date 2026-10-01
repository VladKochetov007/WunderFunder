"""Isolated TRANSHEAD-01 trainer, preserving base trainer/objective/optimizer code."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch
import train_rnn as base
from features import SCHEMA, RAW_SLICE, batch_features, validate_nuisance
from model import FeatureHeadModel


def skip_evaluation_preserving_rng(model, loader, chunk, device):
    # _BaseDataLoaderIter draws a base seed even for sequential evaluation. Preserve
    # that CPU RNG advance so next epoch's training shuffle matches original trainer.
    torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    return {"weighted_pearson": 0.0, "evaluation_skipped": True,
            "reason": "placeholder only; no data evaluated; use fixed model_ep1.pt"}


def feature_metadata(path, arm):
    raw = Path(path).read_bytes()
    result = json.loads(raw)
    validate_nuisance(result["fitted"])
    if result["schema"] != SCHEMA:
        raise ValueError("nuisance schema mismatch")
    return {"schema": SCHEMA, "arm": arm, "nuisance_sha256": hashlib.sha256(raw).hexdigest(),
            "source_fold": result["fold"], "nuisance": result["fitted"]}


def install(metadata):
    original_sequences = base.Sequences
    class FeatureSequences(original_sequences):
        def __getitem__(self, index):
            x, *rest = super().__getitem__(index)
            extra = batch_features(x.numpy()[:, RAW_SLICE], metadata["nuisance"], metadata["arm"])
            return (torch.cat([x, torch.from_numpy(extra)], dim=-1), *rest)
    def factory(args_dict, n_in, n_out):
        if args_dict.get("cell", "gru") != "gru" or args_dict.get("enc", "linear") != "linear":
            raise ValueError("TRANSHEAD01 supports plain linear GRU only")
        if n_in != 211 or args_dict.get("class_heads"):
            raise ValueError("TRANSHEAD01 requires ordinary211input parent without class residuals")
        args_dict["transition_features"] = metadata
        args_dict["evaluation_policy"] = "none; fixed epoch1; RNG-compatible evaluation placeholder"
        return FeatureHeadModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out)
    base.Sequences = FeatureSequences
    base.make_model = factory
    base.evaluate = skip_evaluation_preserving_rng
    # The base loader budget otherwise assumes 112+target columns, omitting extra68.
    base.SEQ_BYTES = 20000*(112+68+3)*4


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature-arm", choices=["F", "T"], required=True)
    parser.add_argument("--feature-nuisance", type=Path, required=True)
    options, rest = parser.parse_known_args()
    metadata = feature_metadata(options.feature_nuisance, options.feature_arm)
    def value(flag, default=None):
        return rest[rest.index(flag)+1] if flag in rest else default
    fold = int(value("--valid-fold", "-1"))
    if metadata["source_fold"] != fold:
        raise ValueError("feature nuisance fold differs from model training fold")
    if value("--train-source") != "valid" or value("--epochs") != "2" or value("--freeze") != "inp,gru":
        raise ValueError("fixed experiment requires valid source,2epochs,frozen inp,gru")
    if not value("--init") or value("--loss") != "gcorr" or "--mask-weight" not in rest:
        raise ValueError("fixed experiment requires parent,gcorr,true mask")
    forbidden = ["--heuristics", "--quant-feats", "--online-norm", "--prop-input", "--lookahead", "--class-heads",
                 "--level-aug", "--level-drop", "--state-reset", "--distill", "--eval-ema", "--raw-only"]
    if any(flag in rest for flag in forbidden):
        raise ValueError("additional experiment-changing option is unsupported")
    # Outputs must be fresh; never overwrite an existing immutable run.
    name = value("--name")
    if not name or (ROOT/"artifacts/runs"/name).exists():
        raise ValueError("missing name or output run already exists")
    install(metadata)
    sys.argv = [str(ROOT/"scripts/train_rnn.py"), *rest]
    base.main()

if __name__ == "__main__":
    main()
