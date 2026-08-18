#!/usr/bin/env python3
"""Compare clean EFS WBF, cross-model NMS, and consensus-WBF/unique-union hybrid."""

from __future__ import annotations
import contextlib, io, itertools, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
ENS=ROOT/"MDImageNet_code"/"12_ensemble"; sys.path.insert(0,str(ENS))
from wbf_fuse import fuse, load_dims, normalize_scores  # noqa:E402
from pycocotools.coco import COCO  # noqa:E402
from pycocotools.cocoeval import COCOeval  # noqa:E402

DATA=HERE/"data"; PREDS=ENS/"preds"; SCORES=ENS/"scores"
NAMES=("clean_e","clean_f","clean_s")

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
    inter=ix*iy; return inter/(aw*ah+bw*bh-inter+1e-9)

def ap50(gt,dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(dets); ev=COCOeval(gt,dt,"bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])

def calibrated(paths):
    return [normalize_scores(json.loads(p.read_text()),"temperature",name)
            for p,name in zip(paths,NAMES)]

def cross_nms(models,iou_thr,score_thr):
    grouped=defaultdict(list)
    for source,dets in enumerate(models):
        for d in dets:
            if d["score"]>=score_thr: grouped[(d["image_id"],d["category_id"])].append((source,d))
    out=[]
    for rows in grouped.values():
        kept=[]
        for source,d in sorted(rows,key=lambda x:-x[1]["score"]):
            # Suppress only cross-model duplicates; each detector retains its native predictions.
            if any(source!=ks and iou(d["bbox"],kd["bbox"])>=iou_thr for ks,kd in kept): continue
            kept.append((source,d)); out.append(d)
    return out

def hybrid(wbf,models,match_iou,unique_score_thr,final_iou=0.7):
    source=defaultdict(list)
    for model_id,dets in enumerate(models):
        for d in dets: source[(d["image_id"],d["category_id"])].append((model_id,d))
    candidates=[]
    for f in wbf:
        matches=defaultdict(list)
        for model_id,d in source.get((f["image_id"],f["category_id"]),[]):
            ov=iou(f["bbox"],d["bbox"])
            if ov>=match_iou: matches[model_id].append((d,ov))
        if len(matches)>=2:
            candidates.append(f)
        elif matches:
            best=max((d for rows in matches.values() for d,_ in rows),key=lambda d:d["score"])
            if best["score"]>=unique_score_thr: candidates.append(best)
    # Standard same-class NMS after merging consensus and restored singleton boxes.
    grouped=defaultdict(list)
    for d in candidates: grouped[(d["image_id"],d["category_id"])].append(d)
    out=[]
    for rows in grouped.values():
        keep=[]
        for d in sorted(rows,key=lambda x:-x["score"]):
            if any(iou(d["bbox"],x["bbox"])>=final_iou for x in keep): continue
            keep.append(d); out.append(d)
    return out

def evaluate_split(split,recipe=None):
    gt_path=DATA/f"val_{split}.json"; paths=[PREDS/f"{n}.clean_val_{split}.json" for n in NAMES]
    with contextlib.redirect_stdout(io.StringIO()): gt=COCO(str(gt_path))
    models=calibrated(paths); dims=load_dims(gt_path); rows=[]
    if recipe is not None:
        settings=[recipe]
    else:
        settings=[]
        for iou_thr,weights in itertools.product((0.60,0.65,0.70),
                ([1.1,1.0,1.0],[1.1,0.9,1.0],[1.2,0.9,1.0])):
            settings.append({"method":"wbf","iou":iou_thr,"weights":weights})
            for unique in (0.01,0.03,0.05,0.10):
                settings.append({"method":"hybrid","iou":iou_thr,"weights":weights,"unique":unique})
        for iou_thr,score in itertools.product((0.50,0.60,0.70),(0.01,0.03,0.05,0.10)):
            settings.append({"method":"cross_nms","iou":iou_thr,"score":score})
    wbf_cache={}
    for cfg in settings:
        if cfg["method"] in ("wbf","hybrid"):
            key=(cfg["iou"],tuple(cfg["weights"]))
            if key not in wbf_cache:
                wbf_cache[key]=fuse(paths,dims,cfg["iou"],0.001,cfg["weights"],"temperature","avg")
            dets=wbf_cache[key] if cfg["method"]=="wbf" else hybrid(
                wbf_cache[key],models,cfg["iou"],cfg["unique"])
        else: dets=cross_nms(models,cfg["iou"],cfg["score"])
        row={**cfg,f"{split}_ap50":ap50(gt,dets),"n_boxes":len(dets)}; rows.append(row)
        print(json.dumps(row),flush=True)
    return rows

def main():
    fit=evaluate_split("fit"); fit.sort(key=lambda x:-x["fit_ap50"]); best=fit[0]
    sel=evaluate_split("sel",best)[0]
    result={"best_frozen_on_fit":best,"selection_result":sel,"all_fit_trials":fit,
            "note":"val_sel opened once after recipe freeze"}
    (SCORES/"clean_efs_fusion_comparison.json").write_text(json.dumps(result,indent=2))
    print("BEST",json.dumps(result,indent=2))

if __name__=="__main__": main()
