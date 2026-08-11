#!/usr/bin/env python3
"""Record raw cost metrics for a checkpoint: params, checkpoint MB, GFLOPs @640, ms/img.

Raw numbers only - these feed a NetScore calculation later.

  python3 measure_model.py --weights runs/y11m_dino_p0p3/weights/best.pt --name y11m_dino_p0p3
"""
import argparse
import json
import time
from pathlib import Path

import torch

from paths import SCORES_DIR, SEED  # noqa: I001
import dino3_modules  # noqa: E402

torch.manual_seed(SEED)


def measure(weights: Path, name: str, imgsz: int = 640, runs: int = 100, warmup: int = 20):
    dino3_modules.register()
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    net = yolo.model.float().eval().cuda()

    total = sum(p.numel() for p in net.parameters())
    # requires_grad is False on every tensor in a saved checkpoint, so it cannot tell us
    # what was trainable. Derive it from the architecture: the frozen part is the ViT trunk.
    frozen = sum(p.numel() for n, p in net.named_parameters() if ".trunk.vit." in n)
    trainable = total - frozen
    ckpt_mb = Path(weights).stat().st_size / 1024**2

    # GFLOPs
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = get_flops(net, imgsz)
    except Exception as e:
        print(f"  ultralytics get_flops failed: {type(e).__name__}: {str(e)[:80]}")
    if not gflops:
        try:
            import thop

            macs, _ = thop.profile(net, inputs=(torch.zeros(1, 3, imgsz, imgsz).cuda(),), verbose=False)
            gflops = macs * 2 / 1e9
        except Exception as e:
            print(f"  thop failed: {type(e).__name__}: {str(e)[:80]}")

    # latency, batch 1
    x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            net(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / runs * 1000

    result = {
        "name": name,
        "weights": str(weights),
        "params_total": total,
        "params_trainable": trainable,
        "params_frozen": frozen,
        "checkpoint_mb": round(ckpt_mb, 2),
        "gflops_640": round(gflops, 2) if gflops else None,
        "latency_ms_per_img_bs1": round(ms, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"{name}_cost.json"
    out.write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"  {k:26} {v}")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    measure(a.weights, a.name, a.imgsz)
