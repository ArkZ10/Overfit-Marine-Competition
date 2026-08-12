#!/usr/bin/env python3
"""Raw cost metrics for all three detectors + the ensemble sum (NetScore inputs).

  python3 measure_costs.py

Records params, checkpoint MB, GFLOPs @640, and ms/img (batch 1) per model, then
sums them - an ensemble pays every model's cost at inference.
"""
import json
import time
from pathlib import Path

import torch

from paths12 import DETECTOR_A_WEIGHTS, IMGSZ, RUNS_DIR, SCORES_DIR, SEED  # noqa: I001

torch.manual_seed(SEED)
RUNS_LAT = 50
WARMUP = 10


def latency(net, imgsz=IMGSZ):
    x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    with torch.no_grad():
        for _ in range(WARMUP):
            net(x) if not isinstance(net, list) else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS_LAT):
            net(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / RUNS_LAT * 1000


def gflops(net, imgsz=IMGSZ):
    try:
        from ultralytics.utils.torch_utils import get_flops
        g = get_flops(net, imgsz)
        if g:
            return g
    except Exception:
        pass
    try:
        import thop
        macs, _ = thop.profile(net, inputs=(torch.zeros(1, 3, imgsz, imgsz).cuda(),), verbose=False)
        return macs * 2 / 1e9
    except Exception as e:
        print(f"  GFLOPs unavailable: {type(e).__name__}")
        return None


def measure_ultralytics(kind, weights, name):
    from ultralytics import RTDETR, YOLO
    m = (RTDETR if kind == "rtdetr" else YOLO)(str(weights))
    net = m.model.float().eval().cuda()
    return {
        "name": name,
        "params_total": sum(p.numel() for p in net.parameters()),
        "checkpoint_mb": round(Path(weights).stat().st_size / 1024**2, 2),
        "gflops_640": round(g, 2) if (g := gflops(net)) else None,
        "latency_ms_bs1": round(latency(net), 3),
    }


def measure_frcnn(weights, name):
    from frcnn.train_frcnn import build_model
    device = torch.device("cuda")
    net = build_model(device)
    ck = torch.load(weights, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()
    # torchvision detectors take a list of CHW tensors
    x = [torch.randn(3, IMGSZ, IMGSZ, device="cuda")]
    with torch.no_grad():
        for _ in range(WARMUP):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS_LAT):
            net(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / RUNS_LAT * 1000
    g = None
    try:
        import thop
        macs, _ = thop.profile(net, inputs=(x,), verbose=False)
        g = macs * 2 / 1e9
    except Exception as e:
        print(f"  FRCNN GFLOPs unavailable: {type(e).__name__}")
    return {
        "name": name,
        "params_total": sum(p.numel() for p in net.parameters()),
        "checkpoint_mb": round(Path(weights).stat().st_size / 1024**2, 2),
        "gflops_640": round(g, 2) if g else None,
        "latency_ms_bs1": round(ms, 3),
    }


if __name__ == "__main__":
    out = []
    print("measuring detector A (YOLOv11m)...")
    out.append(measure_ultralytics("yolo", DETECTOR_A_WEIGHTS, "y11m_control"))
    print("measuring detector B (RT-DETR-l)...")
    out.append(measure_ultralytics("rtdetr", RUNS_DIR / "rtdetr_l" / "weights" / "best.pt", "rtdetr_l"))
    print("measuring detector C (Faster R-CNN)...")
    out.append(measure_frcnn(RUNS_DIR / "frcnn_r50v2" / "best.pth", "frcnn_r50v2"))

    ens = {
        "name": "ensemble_sum",
        "params_total": sum(r["params_total"] for r in out),
        "checkpoint_mb": round(sum(r["checkpoint_mb"] for r in out), 2),
        "gflops_640": round(sum(r["gflops_640"] for r in out if r["gflops_640"]), 2),
        "latency_ms_bs1": round(sum(r["latency_ms_bs1"] for r in out), 3),
    }
    out.append(ens)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    (SCORES_DIR / "costs.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'model':<16} {'params':>12} {'ckpt MB':>9} {'GFLOPs':>9} {'ms/img':>8}")
    for r in out:
        print(f"{r['name']:<16} {r['params_total']:>12,} {r['checkpoint_mb']:>9.1f} "
              f"{(r['gflops_640'] or 0):>9.1f} {r['latency_ms_bs1']:>8.2f}")
    print(f"\nwrote {SCORES_DIR / 'costs.json'}")
