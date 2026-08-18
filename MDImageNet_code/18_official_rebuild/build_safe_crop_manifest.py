#!/usr/bin/env python3
"""Build a three-state crop manifest without unsafe random-background labels.

States:
  positive   official annotated boxes (target 0..33)
  ambiguous  same-class teacher consensus unmatched to official GT (target -1 / ignore)
  negative   sampled regions that avoid GT and every teacher proposal above --veto-conf

The script writes crop coordinates and supervision state only; it does not duplicate images.
For detector training, ambiguous boxes should mask classification/background loss and must not
supervise regression.  For a crop classifier, omit ambiguous rows from CE loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou_matrix(a, b):
    a=np.asarray(a,dtype=float).reshape(-1,4); b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    a2=a[:,:2]+a[:,2:]; b2=b[:,:2]+b[:,2:]
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a2[:,None,:],b2[None,:,:])
    wh=np.clip(hi-lo,0,None); inter=wh[...,0]*wh[...,1]
    aa=(a[:,2]*a[:,3])[:,None]; bb=(b[:,2]*b[:,3])[None,:]
    return inter/(aa+bb-inter+1e-9)


def group(rows):
    out=defaultdict(list)
    for x in rows: out[x["image_id"]].append(x)
    return out


def consensus(dumps, gt_by, min_conf, match_iou):
    """Return conservative all-teacher, same-class consensus unmatched to any GT."""
    by_model=[group([d for d in ds if d["score"]>=min_conf]) for ds in dumps]
    result=[]
    image_ids=set.intersection(*(set(x) for x in by_model)) if by_model else set()
    for image_id in sorted(image_ids):
        base=by_model[0][image_id]
        gt_boxes=[a["bbox"] for a in gt_by.get(image_id,[])]
        for d in base:
            if gt_boxes and iou_matrix([d["bbox"]],gt_boxes).max()>=match_iou: continue
            matches=[]
            for other in by_model[1:]:
                candidates=[x for x in other.get(image_id,[]) if x["category_id"]==d["category_id"]]
                if not candidates: break
                ious=iou_matrix([d["bbox"]],[x["bbox"] for x in candidates])[0]
                j=int(np.argmax(ious))
                if ious[j]<match_iou: break
                matches.append((candidates[j],float(ious[j])))
            else:
                support=min([d["score"]]+[x[0]["score"] for x in matches])
                result.append({**d,"support":support,
                               "mean_teacher_iou":float(np.mean([x[1] for x in matches]))})
    # Suppress duplicate consensus regions within class, highest minimum support first.
    kept=[]
    for d in sorted(result,key=lambda x:-x["support"]):
        overlap=[x for x in kept if x["image_id"]==d["image_id"] and
                 x["category_id"]==d["category_id"]]
        if overlap and iou_matrix([d["bbox"]],[x["bbox"] for x in overlap]).max()>=match_iou:
            continue
        kept.append(d)
    return kept


def sample_negatives(rng, image, gt_boxes, veto_boxes, sizes, count, reject_iou):
    out=[]; width,height=image["width"],image["height"]
    for _ in range(count*100):
        if len(out)>=count: break
        sw,sh=rng.choice(sizes)
        scale=rng.uniform(0.8,1.25); w=min(width,max(32,sw*scale)); h=min(height,max(32,sh*scale))
        x=rng.uniform(0,max(0,width-w)); y=rng.uniform(0,max(0,height-h)); box=[x,y,w,h]
        blockers=gt_boxes+veto_boxes+[x["bbox"] for x in out]
        if blockers and iou_matrix([box],blockers).max()>=reject_iou: continue
        out.append({"bbox":box})
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt",type=Path,required=True)
    ap.add_argument("--dumps",type=Path,nargs="+",required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--consensus-conf",type=float,default=0.05)
    ap.add_argument("--veto-conf",type=float,default=0.01,
                    help="any teacher proposal above this vetoes a random negative")
    ap.add_argument("--match-iou",type=float,default=0.5)
    ap.add_argument("--reject-iou",type=float,default=0.1)
    ap.add_argument("--neg-per-image",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    gt=json.loads(args.gt.read_text()); dumps=[json.loads(p.read_text()) for p in args.dumps]
    valid_ids={im["id"] for im in gt["images"]}
    for p,ds in zip(args.dumps,dumps):
        extra={d["image_id"] for d in ds}-valid_ids
        if extra: raise SystemExit(f"{p}: predictions outside GT manifest")
    gt_by=group(gt["annotations"]); images={im["id"]:im for im in gt["images"]}
    ambiguous=consensus(dumps,gt_by,args.consensus_conf,args.match_iou)
    amb_by=group(ambiguous)
    veto_by=[]
    for ds in dumps: veto_by.append(group([d for d in ds if d["score"]>=args.veto_conf]))
    sizes=[(a["bbox"][2],a["bbox"][3]) for a in gt["annotations"]] or [(224,224)]
    rows=[]
    for a in gt["annotations"]:
        rows.append({"image_id":a["image_id"],"file_name":images[a["image_id"]]["file_name"],
                     "state":"positive","target":a["category_id"],"candidate_class":"",
                     "x":a["bbox"][0],"y":a["bbox"][1],"w":a["bbox"][2],"h":a["bbox"][3],
                     "support":"official"})
    for d in ambiguous:
        rows.append({"image_id":d["image_id"],"file_name":images[d["image_id"]]["file_name"],
                     "state":"ambiguous","target":-1,"candidate_class":d["category_id"],
                     "x":d["bbox"][0],"y":d["bbox"][1],"w":d["bbox"][2],"h":d["bbox"][3],
                     "support":round(d["support"],6)})
    rng=random.Random(args.seed); shortfall=0
    for image_id,image in images.items():
        gt_boxes=[x["bbox"] for x in gt_by.get(image_id,[])]
        veto_boxes=[]
        for by in veto_by: veto_boxes.extend(x["bbox"] for x in by.get(image_id,[]))
        veto_boxes.extend(x["bbox"] for x in amb_by.get(image_id,[]))
        negs=sample_negatives(rng,image,gt_boxes,veto_boxes,sizes,args.neg_per_image,args.reject_iou)
        shortfall+=args.neg_per_image-len(negs)
        for d in negs:
            b=d["bbox"]
            rows.append({"image_id":image_id,"file_name":image["file_name"],"state":"negative",
                         "target":34,"candidate_class":"","x":round(b[0],3),"y":round(b[1],3),
                         "w":round(b[2],3),"h":round(b[3],3),"support":"teacher_vetoed"})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    counts={state:sum(x["state"]==state for x in rows) for state in ("positive","ambiguous","negative")}
    meta={"gt":str(args.gt),"dumps":[str(x) for x in args.dumps],"counts":counts,
          "negative_shortfall":shortfall,"parameters":vars(args)}
    meta["parameters"]={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v
                        for k,v in meta["parameters"].items()}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({**counts,"negative_shortfall":shortfall,"out":str(args.out)},indent=2))


if __name__=="__main__": main()
