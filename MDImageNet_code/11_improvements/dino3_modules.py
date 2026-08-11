"""DINOv3 -> YOLO11 injection modules (Phase 2, candidate (c)).

DINO-YOLO-style injection (arXiv:2510.25140): a frozen DINOv3 ViT-B/16 supplies
features at two points - P0 (input preprocessing) and P3 (mid-backbone). The YOLO
weights and the small fusion adapters train; the ViT never does.

Weights come from the timm mirror `vit_base_patch16_dinov3.lvd1689m`, which is
UNGATED. The canonical `facebook/dinov3-vitb16-pretrain-lvd1689m` repo is
gated=manual and needs an accepted licence + HF token, so it cannot be used here.

Design notes
------------
* The ViT runs ONCE per forward. `DINO3Preprocessor` (layer 0) owns the trunk,
  computes the feature map, and stashes it; `DINO3P3Fusion` reads that stash
  through a non-registered back-reference, so there is no second ViT pass and no
  second copy of the 85.6M frozen parameters in the checkpoint.
* The back-reference is wired at construction time via `_BUILD_REGISTRY`, because
  ultralytics' parse_model builds layers sequentially and gives a layer no handle
  on its predecessors. P0 must appear before the P3 fusion in the yaml.
* `copy.deepcopy` (used for the EMA model) keeps the sharing intact: deepcopy's
  memo maps the registered submodule and the back-reference to the same new
  object, so the EMA copy stashes and reads on its own trunk.
* Both adapters are gated by a zero-initialised scalar, so at step 0 the network
  is numerically identical to plain YOLO11m and training starts from the
  pretrained optimum rather than a perturbed one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DINO = "vit_base_patch16_dinov3.lvd1689m"
DINO_EMBED_DIM = 768
DINO_PATCH = 16

# MUST be passed as ultralytics' `freeze=` argument. Setting requires_grad=False at
# construction is NOT enough: BaseTrainer._setup_train walks named_parameters() and
# force-enables grad on anything not matched by `freeze`, which would silently train
# the ViT. Entries are matched as the substring f"model.{entry}." in parameter names.
FREEZE_SPEC = ["0.trunk.vit"]

# parse_model builds layers in order; P0 drops itself here so the P3 layer can find it.
_BUILD_REGISTRY: dict[str, "DINO3Preprocessor"] = {}


class DINOv3Trunk(nn.Module):
    """Frozen DINOv3 ViT-B/16 emitting a dense feature map (B, 768, H/16, W/16)."""

    def __init__(self, model_name: str = DEFAULT_DINO):
        super().__init__()
        import timm

        self.vit = timm.create_model(
            model_name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.embed_dim = getattr(self.vit, "embed_dim", DINO_EMBED_DIM)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.eval()

    def train(self, mode: bool = True):
        """Stay in eval mode permanently - the trunk is frozen."""
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gh, gw = h // DINO_PATCH, w // DINO_PATCH
        tokens = self.vit.forward_features(x)  # (B, n_prefix + gh*gw, C)
        n_prefix = tokens.shape[1] - gh * gw  # CLS + register tokens
        patches = tokens[:, n_prefix:, :]
        return patches.transpose(1, 2).reshape(b, self.embed_dim, gh, gw).contiguous()


class DINO3Preprocessor(nn.Module):
    """P0 injection: enrich the input image with DINOv3 features.

    yaml: [-1, 1, DINO3Preprocessor, ['<timm-name>', True]]   # 3 -> 3 channels
    """

    def __init__(self, model_name: str = DEFAULT_DINO, freeze: bool = True, key: str = "p0"):
        super().__init__()
        self.trunk = DINOv3Trunk(model_name)
        if not freeze:
            raise ValueError("Phase 2 spec requires the ViT frozen; freeze=False is unsupported")
        self.proj = nn.Sequential(
            nn.Conv2d(self.trunk.embed_dim, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self._stash: torch.Tensor | None = None
        _BUILD_REGISTRY[key] = self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x).to(x.dtype)
        self._stash = feats  # consumed by DINO3P3Fusion later in this same forward
        delta = F.interpolate(self.proj(feats), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.gate * delta


class DINO3P3Fusion(nn.Module):
    """P3 injection: fuse the stashed DINOv3 features into the P3 feature map.

    yaml: [-1, 1, DINO3P3Fusion, [256, '<timm-name>', True]]  # c1 -> c1 channels
    """

    def __init__(self, c1: int, model_name: str = DEFAULT_DINO, freeze: bool = True, key: str = "p0"):
        super().__init__()
        src = _BUILD_REGISTRY.get(key)
        if src is None:
            raise RuntimeError(
                f"DINO3P3Fusion found no DINO3Preprocessor registered under key '{key}'. "
                "The P0 layer must be declared before the P3 layer in the model yaml."
            )
        # plain list => not registered as a submodule => trunk weights stored once,
        # while deepcopy's memo still keeps EMA copies pointing at their own P0.
        self._src = [src]
        self.c1 = c1
        self.proj = nn.Sequential(
            nn.Conv2d(src.trunk.embed_dim, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._src[0]._stash
        if feats is None:
            raise RuntimeError("DINO3P3Fusion ran before DINO3Preprocessor populated its stash")
        if x.shape[1] != self.c1:
            raise ValueError(f"DINO3P3Fusion expected {self.c1} channels, got {x.shape[1]}")
        up = F.interpolate(self.proj(feats.to(x.dtype)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.gate * up


def register():
    """Expose the modules to ultralytics' parse_model, which resolves names via globals()."""
    from ultralytics.nn import tasks

    for cls in (DINOv3Trunk, DINO3Preprocessor, DINO3P3Fusion):
        setattr(tasks, cls.__name__, cls)
    # torch.load(weights_only=True) must be able to reconstruct these on resume
    try:
        torch.serialization.add_safe_globals([DINOv3Trunk, DINO3Preprocessor, DINO3P3Fusion])
    except Exception:
        pass
