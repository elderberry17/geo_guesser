import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import GeoDataset, HoldoutDataset
from src.model import build_model
from src.utils import run_epoch

# unlearn to predict sees (def)
# try to train a model from scratch (cnn / vit)
# different backbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def predict_holdout(model, data_root, val_tf, lat_mean, lat_std, lng_mean, lng_std,
                    batch_size, workers, device):
    holdout_ds = HoldoutDataset(data_root / "holdout_public", val_tf)
    loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    filenames, pred_lats, pred_lngs = [], [], []
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="Predicting holdout"):
            coord_pred, _ = model(pixel_values=imgs.to(device))
            p = coord_pred.cpu().numpy()
            pred_lats.extend(p[:, 0] * lat_std + lat_mean)
            pred_lngs.extend(p[:, 1] * lng_std + lng_mean)
            filenames.extend(names)
    return pd.DataFrame({"filename": filenames, "pred_lat": pred_lats, "pred_lng": pred_lngs})


def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    labels = pd.read_csv(data_root / "train_labels.csv")

    lat_mean, lat_std = labels["lat"].mean(), labels["lat"].std()
    lng_mean, lng_std = labels["lng"].mean(), labels["lng"].std()
    print(f"Lat  mean={lat_mean:.3f}  std={lat_std:.3f}")
    print(f"Lng  mean={lng_mean:.3f}  std={lng_std:.3f}")

    val_df = labels.groupby("country", group_keys=False).apply(
        lambda g: g.sample(frac=args.val_frac, random_state=42), include_groups=False
    )
    val_df = labels.loc[val_df.index]
    train_df = labels.drop(val_df.index)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print(f"Val country distribution:\n{val_df['country'].value_counts(normalize=True)}")

    norm = dict(lat_mean=lat_mean, lat_std=lat_std, lng_mean=lng_mean, lng_std=lng_std)
    train_tf, val_tf = get_transforms()

    pin = device.type == "cuda"
    train_loader = DataLoader(
        GeoDataset(train_df, data_root / "train", train_tf, **norm),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        GeoDataset(val_df, data_root / "train", val_tf, **norm),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pt"
    best_median = float("inf")

    print('loaders:', len(train_loader), len(val_loader))

    for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs):
        tr_loss, tr_km = run_epoch(model, train_loader, optimizer,
                                   device, aux_weight=args.aux_weight, train=True, **norm)
        vl_loss, vl_km = run_epoch(model, val_loader, optimizer,
                                   device, aux_weight=args.aux_weight, train=False, **norm)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train {tr_loss:.2f} / {tr_km:.1f} km  "
              f"val {vl_loss:.2f} / {vl_km:.1f} km")

        if vl_km < best_median:
            best_median = vl_km
            torch.save({"model_state": model.state_dict(), "norm": norm, "epoch": epoch},
                       best_ckpt)
            print(f"  -> best saved ({best_median:.1f} km)")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    print(f"\nBest val median: {best_median:.1f} km")

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    preds = predict_holdout(model, data_root, val_tf, **norm,
                            batch_size=args.batch_size, workers=args.workers, device=device)
    csv_path = out_dir / "predictions.csv"
    preds.to_csv(csv_path, index=False)
    print(f"Saved {len(preds)} predictions → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="geo_dataset")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--wd",          type=float, default=1e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--workers",     type=int,   default=1)
    p.add_argument("--aux_weight",  type=float, default=0.3,
                   help="Weight for country classification auxiliary loss")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
