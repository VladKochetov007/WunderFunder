"""Recompute paired first-wave effects on the frozen exposed F0 components."""
from __future__ import annotations
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from wnn.evaluation import paired_component_bootstrap, paired_component_moments, _score_component_moments
from wnn.gmetric import WARMUP
ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            d.update(chunk)
    return d.hexdigest()

def main() -> None:
    split_path=ROOT/"research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split=json.loads(split_path.read_text()); groups=split["evaluation"]["valid"]
    component_of={x["row_group"]:x["component"] for x in split["sequences"]["valid"]}
    with np.load(ROOT/"cache/valid_targets.npz",allow_pickle=False) as cache:
        order=list(map(int,cache["row_group"])); positions={g:i for i,g in enumerate(order)}
        indices=np.array([positions[g] for g in groups]); target=cache["y"][indices]; mask=cache["mask"][indices]
    mask[:,:WARMUP]=False
    components=np.array([component_of[g] for g in groups])
    control3="clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1"
    control4="clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1"
    comparisons=[("amplitude_s3","clean_t3_f0_amplitude1_s3_a9_t3r5_r3_p3",control3,.6811949693747386,.6811553603333363),
                 ("trajectory_mean_s3","clean_t3_f0_control_s3_swa345_strategy_r1",control3,.6811988703460603,.6811553603333363),
                 ("trajectory_mean_s4","clean_t3_f0_control_s4_swa345_strategy_r1",control4,.6813158771373331,.6812584217032286)]
    results=[]
    for label,candidate,reference,known_a,known_b in comparisons:
        arrays=[]; bindings=[]
        for run in (candidate,reference):
            path=ROOT/"cache"/f"pred_{run}_valid.npy"; side=json.loads(path.with_suffix(".json").read_text())
            assert side["run"]==run and side["row_groups"]==order and side["split_manifest_sha256"]==sha(split_path)
            assert side["prediction_sha256"]==sha(path)
            arrays.append(np.load(path,mmap_mode="r",allow_pickle=False)[indices])
            bindings.append({"run":run,"sha256":side["prediction_sha256"],"checkpoint_sha256":side["checkpoint_sha256"]})
        a,b=paired_component_moments(target,arrays[0],arrays[1],mask)
        score_a=_score_component_moments(a,np.ones(len(a))); score_b=_score_component_moments(b,np.ones(len(b)))
        assert abs(score_a-known_a)<1e-12 and abs(score_b-known_b)<1e-12
        bootstrap=paired_component_bootstrap(a,b,components,n_boot=2000,seed=20261013,centered=True)
        results.append({"comparison":label,"candidate_wp":score_a,"control_wp":score_b,"delta":score_a-score_b,"paired_component_bootstrap":bootstrap,"bindings":bindings})
    record={"created_utc":datetime.now(UTC).isoformat(),"results":results,"evaluation_groups":len(groups),"components":len(set(components.tolist())),"selected_rows":int(mask.sum()),"split_sha256":sha(split_path),"source_sha256":{p:sha(ROOT/p) for p in ["src/wnn/evaluation.py","src/wnn/gmetric.py"]},"analysis_source_sha256":sha(Path(__file__)),"exposure":"Repeatedly inspected F0; conditional fixed-prediction uncertainty only. This does not estimate fitting, teacher, selection or public-transfer uncertainty."}
    (BASE/"RESULTS_R1.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(results))
if __name__=="__main__":main()
