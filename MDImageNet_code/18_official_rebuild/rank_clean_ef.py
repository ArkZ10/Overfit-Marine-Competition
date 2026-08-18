#!/usr/bin/env python3
"""Conservative E/F candidate reranker, evaluated image-level out of fold on val_fit.

Official matches are positives. Duplicates, wrong-class overlaps, and poor-localization
overlaps are negatives. Predictions unmatched to any annotation are *not* used as negatives,
because the clean audit showed that many are real unlabeled objects.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENS=ROOT/"MDImageNet_code"/"12_ensemble"
sys.path.insert(0,str(ENS))
from calibrate import apply_temperature  # noqa:E402
from pycocotools.coco import COCO  # noqa:E402
from pycocotools.cocoeval import COCOeval  # noqa:E402

DATA=HERE/"data"; PREDS=ENS/"preds"; SCORES=ENS/"scores"
GT_PATH=DATA/"val_fit.json"
FEATURE_NAMES=["raw_logit","e_logit","f_logit","support","ef_iou","score_min",
               "score_max","score_gap","area_frac","log_aspect","category"]


def iou_matrix(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    a2=a[:,:2]+a[:,2:]; b2=b[:,:2]+b[:,2:]
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a2[:,None,:],b2[None,:,:])
    wh=np.clip(hi-lo,0,None); inter=wh[...,0]*wh[...,1]
    aa=(a[:,2]*a[:,3])[:,None]; bb=(b[:,2]*b[:,3])[None,:]
    return inter/(aa+bb-inter+1e-9)


def logit(x):
    x=np.clip(x,1e-6,1-1e-6); return np.log(x/(1-x))


def group(rows):
    out=defaultdict(list)
    for x in rows: out[x["image_id"]].append(x)
    return out


def ap50(gt,dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(dets); ev=COCOeval(gt,dt,"bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])


def member_features(candidate, members, temps):
    best=[]
    for rows,T in zip(members,temps):
        same=[x for x in rows if x["category_id"]==candidate["category_id"]]
        if not same: best.append((0.0,None)); continue
        ious=iou_matrix([candidate["bbox"]],[x["bbox"] for x in same])[0]
        j=int(np.argmax(ious))
        if ious[j]<0.3: best.append((0.0,None)); continue
        best.append((float(apply_temperature(np.array([same[j]["score"]]),T)[0]),same[j]))
    es,fs=best[0][0],best[1][0]
    both=best[0][1] is not None and best[1][1] is not None
    ef_iou=float(iou_matrix([best[0][1]["bbox"]],[best[1][1]["bbox"]])[0,0]) if both else 0.0
    present=[s for s in (es,fs) if s>0]
    return es,fs,sum(x[1] is not None for x in best),ef_iou


def build_rows(gt_data,candidates,e_dets,f_dets):
    gt_by=group(gt_data["annotations"]); e_by=group(e_dets); f_by=group(f_dets)
    dims={x["id"]:(x["width"],x["height"]) for x in gt_data["images"]}
    temps=[json.loads((SCORES/"clean_e.calib.json").read_text())["temperature"],
           json.loads((SCORES/"clean_f.calib.json").read_text())["temperature"]]
    rows=[]
    for image_id,cs in group(candidates).items():
        gs=gt_by.get(image_id,[]); gb=[x["bbox"] for x in gs]; claimed=set()
        width,height=dims[image_id]
        for c in sorted(cs,key=lambda x:-x["score"]):
            ious=iou_matrix([c["bbox"]],gb)[0] if gb else np.zeros(0)
            same=[j for j,g in enumerate(gs) if g["category_id"]==c["category_id"]]
            good=[j for j in same if ious[j]>=0.5]
            free=[j for j in good if j not in claimed]
            if free:
                j=max(free,key=lambda x:ious[x]); claimed.add(j); outcome="tp"; label=1
            elif good: outcome="duplicate"; label=0
            elif len(ious) and ious.max()>=0.5: outcome="class_confusion"; label=0
            elif same and ious[same].max()>=0.1: outcome="localization"; label=0
            else: outcome="ambiguous_unmatched"; label=-1
            es,fs,support,ef_iou=member_features(c,[e_by.get(image_id,[]),f_by.get(image_id,[])],temps)
            x,y,w,h=c["bbox"]; scores=[z for z in (es,fs) if z>0]
            feats=[logit(c["score"]),logit(es) if es else -13.8,logit(fs) if fs else -13.8,
                   support,ef_iou,min(scores) if scores else 0,max(scores) if scores else 0,
                   abs(es-fs),w*h/(width*height),np.log(max(w/h,1e-6)),c["category_id"]]
            rows.append({"det":c,"image_id":image_id,"label":label,"outcome":outcome,"x":feats})
    return rows


def main():
    gt_data=json.loads(GT_PATH.read_text()); ids={x["id"] for x in gt_data["images"]}
    load=lambda p:[x for x in json.loads(p.read_text()) if x["image_id"] in ids]
    candidates=load(PREDS/"clean_ef_frozen.clean_val.json")
    e=load(PREDS/"clean_e.clean_val.json"); f=load(PREDS/"clean_f.clean_val.json")
    rows=build_rows(gt_data,candidates,e,f)
    labeled=[r for r in rows if r["label"]>=0]
    counts=defaultdict(int)
    for r in rows: counts[r["outcome"]]+=1
    image_ids=sorted(ids); random.Random(42).shuffle(image_ids)
    folds={im:k%5 for k,im in enumerate(image_ids)}
    oof=np.zeros(len(rows)); models=[]
    for fold in range(5):
        train=[r for r in labeled if folds[r["image_id"]]!=fold]
        test_idx=[i for i,r in enumerate(rows) if folds[r["image_id"]]==fold]
        X=np.asarray([r["x"] for r in train]); y=np.asarray([r["label"] for r in train])
        model=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.05,max_leaf_nodes=15,
                min_samples_leaf=30,l2_regularization=2.0,categorical_features=[10],random_state=42)
        model.fit(X,y); oof[test_idx]=model.predict_proba(np.asarray([rows[i]["x"] for i in test_idx]))[:,1]
        models.append(model)
    with contextlib.redirect_stdout(io.StringIO()): gt=COCO(str(GT_PATH))
    baseline=ap50(gt,candidates); trials=[]
    for alpha in (0.0,0.25,0.5,1.0,2.0):
        out=[]
        for r,p in zip(rows,oof):
            score=float(1/(1+np.exp(-(logit(r["det"]["score"])+alpha*logit(p))))) if alpha else r["det"]["score"]
            out.append({**r["det"],"score":score})
        trials.append({"alpha":alpha,"oof_ap50":ap50(gt,out)})
        if alpha==max((x["alpha"] for x in trials),default=alpha): pass
    best=max(trials,key=lambda x:x["oof_ap50"])
    best_dump=[]
    for r,p in zip(rows,oof):
        score=float(1/(1+np.exp(-(logit(r["det"]["score"])+best["alpha"]*logit(p)))))
        best_dump.append({**r["det"],"score":score})
    (PREDS/"clean_ef_ranker_oof.val_fit.json").write_text(json.dumps(best_dump))
    result={"baseline_fit_ap50":baseline,"outcome_counts":dict(counts),"n_labeled":len(labeled),
            "n_ambiguous_ignored":len(rows)-len(labeled),"features":FEATURE_NAMES,"trials":trials,
            "best":best,"warning":"OOF val_fit research result; val_sel not opened."}
    (SCORES/"clean_ef_ranker_oof.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
