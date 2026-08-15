"""RTMDet checkpoint -> shared 12_ensemble dump format (called by dump_preds.py).

  --tta uses mmdet's built-in RTMDet TTA (3 scales x 2 flips, from rtmdet_tta.py).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "rtmdet_l_marine.py"
CONF = 0.001
MAX_DET = 300
BATCH = 16


def run_inference(weights, img_files, stem_to_id, device="cuda:0", config=None,
                  tta=False, batch=BATCH, imgsz=None):
    import torch
    from mmdet.apis import inference_detector, init_detector
    from mmengine.registry import init_default_scope

    init_default_scope("mmdet")
    cfg_path = str(config or CONFIG)

    # torch>=2.6 defaults torch.load(weights_only=True). mmengine checkpoints embed a
    # message_hub (HistoryBuffer, numpy reconstructors, ...) that the strict unpickler
    # rejects, and allowlisting them one by one is endless. These checkpoints are ones
    # we produced, so relax the flag only for the duration of this load.
    _orig_load = torch.load

    def _trusted_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _trusted_load
    try:
        model = init_detector(cfg_path, str(weights), device=device)
    finally:
        torch.load = _orig_load

    if tta:  # mmdet ships RTMDet TTA (3 scales x 2 flips) in rtmdet_tta.py
        from mmengine.config import Config
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import MODELS
        cfg = Config.fromfile(cfg_path)
        model.cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        tta_model = MODELS.build(cfg.tta_model)
        tta_model.module = model
        model = revert_sync_batchnorm(tta_model)

    if imgsz is not None:  # rescale the test pipeline's Resize for multi-scale TTA
        for op in model.cfg.test_dataloader.dataset.pipeline:
            if op.get("type") == "Resize":
                op["scale"] = (imgsz, imgsz)
        model.cfg.test_pipeline = model.cfg.test_dataloader.dataset.pipeline

    paths = [str(p) for p in img_files]
    stems = [Path(p).stem for p in paths]
    dets = []
    for i in range(0, len(paths), batch):
        chunk, chunk_stems = paths[i:i + batch], stems[i:i + batch]
        with torch.no_grad():
            results = inference_detector(model, chunk)
        if not isinstance(results, list):
            results = [results]
        for r, s in zip(results, chunk_stems):
            if s not in stem_to_id:
                continue
            iid = stem_to_id[s]
            pi = r.pred_instances
            keep = pi.scores.argsort(descending=True)[:MAX_DET]
            bx = pi.bboxes[keep].float().cpu().numpy()
            sc = pi.scores[keep].float().cpu().numpy()
            lb = pi.labels[keep].cpu().numpy()
            for (x1, y1, x2, y2), s_, l_ in zip(bx, sc, lb):
                if s_ < CONF:
                    continue
                dets.append({
                    "image_id": iid,
                    "category_id": int(l_),
                    "bbox": [round(float(x1), 3), round(float(y1), 3),
                             round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                    "score": round(float(s_), 6),
                })
    return dets
