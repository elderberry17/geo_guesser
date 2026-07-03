import torch
import torch.nn as nn

from src.data import COUNTRIES


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=3.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """Plain ViT trained from scratch: patchify -> [CLS] + pos embed -> transformer blocks."""

    IMG_SIZE = 256
    PATCH_SIZE = 16
    EMBED_DIM = 224
    DEPTH = 6
    HEADS = 4
    MLP_RATIO = 3.0

    def __init__(self):
        super().__init__()
        n_patches = (self.IMG_SIZE // self.PATCH_SIZE) ** 2
        self.patch_embed = nn.Conv2d(
            3, self.EMBED_DIM, kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.EMBED_DIM))
        self.blocks = nn.ModuleList([
            TransformerBlock(self.EMBED_DIM, self.HEADS, self.MLP_RATIO)
            for _ in range(self.DEPTH)
        ])
        self.norm = nn.LayerNorm(self.EMBED_DIM)
        self.feat_dim = self.EMBED_DIM

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


class GeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleViT()
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
