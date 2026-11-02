from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from data_adapter import ROOT,CACHE,load_grid,write_json,digest


def verify(full):
    frozen=json.loads((ROOT/"frozen_code.json").read_text(encoding="utf-8"))
    for name,expected in frozen["sha256"].items():
        assert digest(ROOT/name)==expected, f"Frozen code differs: {name}"
    config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    checkpoint=json.loads((ROOT/"branch_checkpoint.json").read_text(encoding="utf-8"))
    split=json.loads((ROOT/"split_manifest.json").read_text(encoding="utf-8"))
    times,gpus,_=load_grid()
    eligible=np.load(CACHE/"label_eligible.npy")
    paths=sorted((ROOT/"branch_scores").glob("origin_*.parquet"))
    assert len(paths)==len(checkpoint["completed_origins"])
    assert len(paths)==(len(split["origins"]) if full else 1)
    rows=0;outputs={}
    fusion_checks=0
    for path in paths:
        frame=pd.read_parquet(path)
        origin=int(frame.origin_bin.iloc[0])
        bins=np.arange(origin,min(origin+288,split["end_bin_exclusive"]))
        assert np.array_equal(frame.score_bin.to_numpy(),np.repeat(bins,len(gpus)))
        assert np.array_equal(frame.gpu_idx.to_numpy(),np.tile(np.arange(len(gpus)),len(bins)))
        assert np.array_equal(frame.timestamp_ns.to_numpy(),np.repeat(times[bins],len(gpus)))
        for col in ["p_b2_raw",*[f"p_b1_{a}_raw" for a in config["arms"]]]:
            values=frame[col].to_numpy()
            assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
        rows+=len(frame);outputs[str(path.relative_to(ROOT))]=digest(path)
        if full:
            for cadence in [5,30]:
                selected=(frame.score_bin.to_numpy()-split["start_bin"])%(cadence//5)==0
                raw=frame.loc[selected]
                fused_path=ROOT/"fusion_scores"/f"{cadence}m"/path.name
                fused=pd.read_parquet(fused_path)
                assert len(fused)==len(raw)
                assert np.array_equal(fused.score_bin.to_numpy(),raw.score_bin.to_numpy())
                assert np.array_equal(fused.gpu_idx.to_numpy(),raw.gpu_idx.to_numpy())
                lam=np.minimum(.6,1./(raw.xid_count_30d_at_origin.to_numpy().astype(np.float32)+1.))
                b2=raw.p_b2_raw.to_numpy()
                for arm in config["arms"]:
                    expected=lam*fused[f"{arm}_B1_inertia"].to_numpy()+(1-lam)*b2
                    np.testing.assert_array_equal(fused[f"{arm}_fusion_main"].to_numpy(),expected)
                    expected=lam*raw[f"p_b1_{arm}_raw"].to_numpy()+(1-lam)*b2
                    np.testing.assert_array_equal(fused[f"{arm}_fusion_no_inertia"].to_numpy(),expected)
                    fusion_checks+=2
                outputs[str(fused_path.relative_to(ROOT))]=digest(fused_path)
    fits=checkpoint["fit_trace"]
    assert sum(t["fit_count"] for t in fits)==6*len(paths)
    for t in fits:
        assert t["latest_train_bin"]<=t["origin"]-config["purge_hours"]*12
        assert t["label_maturity_latest_bin"]<=t["origin"]
        assert t["train_end_bin_exclusive"]-t["train_start_bin"]==t["training_days"]*288
        if t["arm"] in ["A","B","shared"]:
            assert t["training_days"]==21
    for a,c in checkpoint["controllers"].items():
        for t in c["adst_history"]:
            assert t["feedback_latest_score_bin"]<=t["update_origin"]-432
            assert t["feedback_label_maturity_bin"]<=t["update_origin"]
    pd.DataFrame(dict(gpu_idx=np.arange(len(gpus)),gpu_id=gpus)).to_csv(ROOT/"gpu_mapping.csv",index=False)
    if full:
        assert (ROOT/"evaluation_manifest.json").exists()
        summary=pd.read_csv(ROOT/"metrics_summary.csv")
        for cadence in [5,30]:
            for a in config["arms"]:
                row=summary[(summary.cadence_minutes==cadence)&(summary.split=="terminal")&(summary.model==f"{a}/fusion_main")]
                assert len(row)==1
        assert not summary[(summary.split=="terminal")&summary.model.str.contains("dev_")].shape[0]
    result=dict(status="PASS",mode="full" if full else "pilot",origins=len(paths),rows=rows,
                actual_branch_estimator_fits=sum(t["fit_count"] for t in fits),
                full_cluster_gpus=len(gpus),source_only_old_scores_used=False,
                shared_label_mask=True,label_eligible_bins=int(eligible.sum()),
                full_tape_fusion_identity_checks=fusion_checks,outputs_sha256=outputs)
    write_json(ROOT/("final_validation.json" if full else "pilot_validation.json"),result)
    print(json.dumps(result,ensure_ascii=True))


if __name__=="__main__":
    verify(sys.argv[1]=="full")
