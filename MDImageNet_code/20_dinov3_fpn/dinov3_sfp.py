"""DINOv3 ViT backbone with a ViTDet-style simple feature pyramid."""

import torch
from torch import nn
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        LayerNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        LayerNorm2d(out_channels),
    )


@MODELS.register_module()
class DINOv3SimpleFeaturePyramid(BaseModule):
    """Turn one stride-16 DINOv3 feature map into P2--P5 features.

    This follows the Simple Feature Pyramid idea used by ViTDet: learned
    upsampling supplies strides 4 and 8, while identity/downsampling supplies
    strides 16 and 32.  The DINOv3 backbone is fine-tuned with checkpointing.
    """

    def __init__(self,
                 model_name='vit_base_patch16_dinov3.lvd1689m',
                 out_channels=256,
                 pretrained=True,
                 grad_checkpointing=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        if grad_checkpointing and hasattr(self.vit, 'set_grad_checkpointing'):
            self.vit.set_grad_checkpointing(True)
        dim = self.vit.num_features
        self.p2_up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
        )
        self.p3_up = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.p2 = _projection(dim // 4, out_channels)
        self.p3 = _projection(dim // 2, out_channels)
        self.p4 = _projection(dim, out_channels)
        self.p5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                _projection(dim, out_channels))

    def forward(self, x: torch.Tensor):
        # The final normalized block feature is sufficient for SFP and avoids
        # keeping four redundant same-resolution ViT maps alive.
        feats = self.vit.forward_intermediates(
            x, indices=[-1], norm=True, output_fmt='NCHW',
            intermediates_only=True)
        z = feats[0]
        p2 = self.p2(self.p2_up(z))
        p3 = self.p3(self.p3_up(z))
        p4 = self.p4(z)
        p5 = self.p5(z)
        p6 = nn.functional.max_pool2d(p5, kernel_size=1, stride=2)
        return (p2, p3, p4, p5, p6)
