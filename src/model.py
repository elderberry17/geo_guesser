import torch.nn as nn

from src.data import COUNTRIES


class InvertedResidual(nn.Module):
    """MobileNetV2-style block: expand -> depthwise conv -> project."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class SmallCNN(nn.Module):
    """From-scratch MobileNetV2-style backbone, sized to fit the param budget."""

    STEM_CH = 32
    STAGE_WIDTHS = (56, 112, 192, 320)
    STAGE_DEPTHS = (2, 3, 4, 3)
    EXPAND_RATIO = 4
    HEAD_CH = 576

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.STEM_CH, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.STEM_CH),
            nn.ReLU6(inplace=True),
        )

        stages = []
        in_ch = self.STEM_CH
        for width, depth in zip(self.STAGE_WIDTHS, self.STAGE_DEPTHS):
            stages.append(self._make_stage(in_ch, width, depth, stride=2))
            in_ch = width
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, self.HEAD_CH, 1, bias=False),
            nn.BatchNorm2d(self.HEAD_CH),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = self.HEAD_CH

    def _make_stage(self, in_ch, out_ch, depth, stride):
        blocks = [InvertedResidual(in_ch, out_ch, stride, self.EXPAND_RATIO)]
        for _ in range(depth - 1):
            blocks.append(InvertedResidual(out_ch, out_ch, 1, self.EXPAND_RATIO))
        return nn.Sequential(*blocks)

    def forward(self, pixel_values):
        x = self.stem(pixel_values)
        x = self.stages(x)
        x = self.head(x)
        return self.pool(x).flatten(1)


class GeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNN()
        feat_dim = self.backbone.feat_dim
        self.coord_head = nn.Linear(feat_dim, 2)
        self.country_head = nn.Linear(feat_dim, len(COUNTRIES))

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values)
        return self.coord_head(features), self.country_head(features)


def build_model():
    model = GeoModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    assert total <= 5_000_000, f"Model exceeds 5M params: {total:,}"
    return model
