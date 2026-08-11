#!/usr/bin/env python3
"""Phase 2 feasibility spike for candidate (c).

Success criterion: one forward pass + one training step on a 20-image subset of
our own data, DINOv3 ViT-B loaded frozen, no NaN loss.
"""
import random

import numpy as np
import torch

from paths import IMPROVE_DIR, SEED, variant_tree  # noqa: I001
import dino3_modules  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CFG = IMPROVE_DIR / "yolo11m-dino3-p0p3.yaml"
SUBSET_LIST = IMPROVE_DIR / "rfs" / "spike20.txt"
SUBSET_YAML = IMPROVE_DIR / "data_spike20.yaml"


def make_subset():
    """20 train images, reusing the control symlink tree so labels resolve."""
    from make_rfs_list import write_list, write_yaml

    train_imgs = sorted((variant_tree("control") / "images" / "train").glob("*.jpg"))[:20]
    assert len(train_imgs) == 20, f"expected 20 images, got {len(train_imgs)}"
    write_list(SUBSET_LIST, [str(p) for p in train_imgs])
    write_yaml(SUBSET_YAML, variant_tree("control"), SUBSET_LIST, SUBSET_LIST)
    return SUBSET_YAML


def main():
    dino3_modules.register()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    print("=" * 70)
    print("STEP 1: build model from yaml")
    model = DetectionModel(str(CFG), ch=3, nc=34, verbose=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  total params     : {total:,}")
    print(f"  trainable        : {trainable:,}")
    print(f"  frozen (ViT)     : {frozen:,}")
    assert frozen > 80e6, f"ViT does not look frozen (only {frozen:,} frozen params)"

    print("\nSTEP 2: forward pass @ 640")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 640, 640))
    shapes = [tuple(o.shape) for o in (out if isinstance(out, (list, tuple)) else [out])[:1]]
    print(f"  output[0] shape  : {shapes}")

    print("\nSTEP 3: ViT runs exactly once per forward")
    trunk = model.model[0].trunk
    calls = {"n": 0}
    orig = trunk.forward

    def counting(x):
        calls["n"] += 1
        return orig(x)

    trunk.forward = counting
    with torch.no_grad():
        model(torch.randn(1, 3, 640, 640))
    trunk.forward = orig
    print(f"  ViT forward calls: {calls['n']}")
    assert calls["n"] == 1, f"expected 1 ViT pass, got {calls['n']}"

    print("\nSTEP 4: gates zero-initialised => starts as plain YOLO11m")
    print(f"  P0 gate = {model.model[0].gate.item():.4f}   P3 gate = {model.model[6].gate.item():.4f}")

    print("\nSTEP 5: deepcopy (EMA) keeps trunk sharing intact")
    import copy

    ema = copy.deepcopy(model)
    shared = ema.model[6]._src[0] is ema.model[0]
    print(f"  EMA P3 back-ref points at EMA P0: {shared}")
    assert shared, "deepcopy broke the P0<->P3 link; EMA validation would fail"
    del ema

    print("\nSTEP 6: one real training step on 20 of our images")
    data = make_subset()
    vit_ref = model.model[0].trunk.vit.blocks[0].attn.qkv.weight
    before = vit_ref.detach().clone()

    m = YOLO(str(CFG), task="detect")
    m.model = model

    grad_state = {}

    def check_frozen(trainer):
        params = dict(trainer.model.named_parameters())
        vit = {k: v for k, v in params.items() if ".trunk.vit." in k}
        grad_state["n_vit"] = len(vit)
        grad_state["n_grad_on"] = sum(1 for v in vit.values() if v.requires_grad)

    m.add_callback("on_train_start", check_frozen)

    results = m.train(
        data=str(data),
        epochs=1,
        imgsz=640,
        batch=2,
        device="0",
        workers=0,
        seed=SEED,
        freeze=dino3_modules.FREEZE_SPEC,
        val=False,
        plots=False,
        save=False,
        project=str(IMPROVE_DIR / "runs"),
        name="_spike",
        exist_ok=True,
        verbose=False,
    )
    print(f"  training step completed: {results is not None}")
    print(f"  ViT tensors: {grad_state.get('n_vit')}, with requires_grad=True: {grad_state.get('n_grad_on')}")
    assert grad_state.get("n_vit", 0) > 0, "no ViT params seen by the trainer"
    assert grad_state["n_grad_on"] == 0, (
        f"{grad_state['n_grad_on']} ViT tensors still trainable - the ViT is NOT frozen"
    )

    after = model.model[0].trunk.vit.blocks[0].attn.qkv.weight.detach()
    drift = (after.float() - before.float()).abs().max().item()
    print(f"  max |ΔW| on a ViT weight after the step: {drift:.3e}")
    assert drift == 0.0, f"ViT weights changed by {drift:.3e}; it is being trained"

    import csv
    from pathlib import Path

    csv_path = Path(IMPROVE_DIR / "runs" / "_spike" / "results.csv")
    if csv_path.exists():
        row = list(csv.DictReader(open(csv_path)))[-1]
        losses = {k.strip(): v for k, v in row.items() if "loss" in k}
        print(f"  losses: {losses}")
        bad = [k for k, v in losses.items() if v.strip().lower() in {"nan", "inf", "-inf"}]
        assert not bad, f"NaN/Inf in losses: {bad}"

    print("\n" + "=" * 70)
    print("SPIKE PASSED")


if __name__ == "__main__":
    main()
