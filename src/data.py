from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

COUNTRIES = [
    "Belarus", "Finland", "France", "Germany", "Iceland",
    "Italy", "Norway", "Poland", "Spain", "Sweden", "Turkey", "United_Kingdom",
]
COUNTRY_TO_IDX = {c: i for i, c in enumerate(COUNTRIES)}


def stratified_split(labels, val_frac, random_state=42):
    """Split `labels` into (train_df, val_df), sampling val_frac per country
    so both splits keep the same country balance. Deterministic given
    val_frac/random_state, so the same val set can be reconstructed later
    (e.g. to score a checkpoint) without needing to persist it."""
    val_df = labels.groupby("country", group_keys=False).apply(
        lambda g: g.sample(frac=val_frac, random_state=random_state), include_groups=False
    )
    val_df = labels.loc[val_df.index]
    train_df = labels.drop(val_df.index)
    return train_df, val_df


class GeoDataset(Dataset):
    def __init__(self, df, img_dir, transform, lat_mean, lat_std, lng_mean, lng_std):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.lat_mean, self.lat_std = lat_mean, lat_std
        self.lng_mean, self.lng_std = lng_mean, lng_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        img = self.transform(img)
        lat = (row["lat"] - self.lat_mean) / self.lat_std
        lng = (row["lng"] - self.lng_mean) / self.lng_std
        coords = torch.tensor([lat, lng], dtype=torch.float32)
        country = torch.tensor(COUNTRY_TO_IDX[row["country"]], dtype=torch.long)
        return img, coords, country


class HoldoutDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = sorted(Path(img_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].name
