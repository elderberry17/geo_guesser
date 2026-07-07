import torch.nn as nn
from transformers import ConvNextV2Model

from src.data import COUNTRIES


class GeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNextV2Model.from_pretrained("facebook/convnextv2-atto-1k-224")
        feat_dim = self.backbone.config.hidden_sizes[-1]
        self.coord_head = nn.Linear(feat_dim, 2)
        self.country_head = nn.Linear(feat_dim, len(COUNTRIES))

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values).pooler_output
        return self.coord_head(features), self.country_head(features)


def build_model():
    model = GeoModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    assert total <= 5_000_000, f"Model exceeds 5M params: {total:,}"
    return model
