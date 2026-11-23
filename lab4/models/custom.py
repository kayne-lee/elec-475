"""LiteSegNet: our compact student for Step 3.

Encoder : MobileNetV3-Small (ImageNet weights) with the last stage dilated, so the deepest
          features are at stride 16 instead of 32 (as in DeepLabV3 / LR-ASPP).
Context : depthwise-separable ASPP (1x1, three 3x3 atrous branches with rates 2/4/6 and an
          image-pooling branch), 96 channels.
Decoder : two-step decoder with skips from stride 8 (24 ch) and stride 4 (16 ch), each fused
          by a depthwise-separable 3x3 conv, bilinear upsampling, dropout, 1x1 classifier.
Taps    : 'mid' (stride 8, 24 ch) and 'high' (stride 16, 576 ch) for feature KD, matched to
          the teacher's layer2 (512 ch) and layer4 (2048 ch), both at stride 8.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def conv_bn_act(in_ch, out_ch, k=1, groups=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=dilation * (k // 2), dilation=dilation,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.Hardswish(inplace=True))


def ds_conv(in_ch, out_ch, dilation=1):
    """Depthwise 3x3 (optionally atrous) + pointwise 1x1, each followed by BN + h-swish."""
    return nn.Sequential(conv_bn_act(in_ch, in_ch, 3, groups=in_ch, dilation=dilation),
                         conv_bn_act(in_ch, out_ch, 1))


class DSASPP(nn.Module):
    def __init__(self, in_ch, out_ch=96, rates=(2, 4, 6)):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_act(in_ch, out_ch, 1)] +
                                      [ds_conv(in_ch, out_ch, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv_bn_act(out_ch * (len(rates) + 2), out_ch, 1),
                                     nn.Dropout(0.1))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        feats.append(self.pool(x).expand(-1, -1, *x.shape[-2:]))
        return self.project(torch.cat(feats, dim=1))


class LiteSegNet(nn.Module):
    kd_taps = {'mid': 'layer2', 'high': 'layer4'}
    kd_channels = {'mid': (24, 512), 'high': (576, 2048)}

    def __init__(self, num_classes=21, pretrained=True, width=96):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights, dilated=True).features
        self.low = features[:2]     # 16 ch, stride 4
        self.mid = features[2:4]    # 24 ch, stride 8
        self.high = features[4:]    # 576 ch, stride 16 (last stage dilated)
        self.aspp = DSASPP(576, width)
        self.mid_proj = conv_bn_act(24, 32, 1)
        self.fuse8 = ds_conv(width + 32, width)
        self.low_proj = conv_bn_act(16, 24, 1)
        self.fuse4 = ds_conv(width + 24, 64)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.aspp(high)
        y = F.interpolate(y, size=mid.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse8(torch.cat([y, self.mid_proj(mid)], dim=1))
        y = F.interpolate(y, size=low.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse4(torch.cat([y, self.low_proj(low)], dim=1))
        y = self.classifier(self.dropout(y))
        out = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        return {'out': out, 'low': low, 'mid': mid, 'high': high}


if __name__ == '__main__':
    from utils.profiler import ms_per_image

    model = LiteSegNet().eval()
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
    for name in ['low', 'mid', 'high', 'aspp', 'mid_proj', 'fuse8', 'low_proj', 'fuse4',
                 'classifier']:
        print(f'  {name:<10}{sum(p.numel() for p in getattr(model, name).parameters()):>10,}')
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    for k, v in out.items():
        print(f'{k:<5} {tuple(v.shape)}')
    print(f'ms/img (CPU): {ms_per_image(model, torch.device("cpu")):.1f}')
