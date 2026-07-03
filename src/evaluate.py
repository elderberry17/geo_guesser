"""
Standalone inference: load a trained checkpoint and either

  - write predictions.csv for the unlabelled holdout set, or
  - score the labelled validation split (same one train.py held out) so
    checkpoints/architectures can be compared on real metrics.

Usage:
    python -m src.evaluate --checkpoint outputs/best_model.pt --output_dir outputs
    python -m src.evaluate --checkpoint outputs/best_model.pt --val
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import GeoDataset, stratified_split
from src.model import build_model
from src.train import get_transforms, predict_holdout
from src.utils import haversine_km


def score_val_set(model, data_root, val_tf, lat_mean, lat_std, lng_mean, lng_std,
                   val_frac, batch_size, workers, device):
    labels = pd.read_csv(data_root / "train_labels.csv")
    _, val_df = stratified_split(labels, val_frac)

    loader = DataLoader(
        GeoDataset(val_df, data_root / "train", val_tf,
                   lat_mean=lat_mean, lat_std=lat_std, lng_mean=lng_mean, lng_std=lng_std),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )

    model.eval()
    pred_lats, pred_lngs = [], []
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader, desc="Scoring val set"):
            coord_pred, _ = model(pixel_values=imgs.to(device))
            p = coord_pred.cpu().numpy()
            pred_lats.extend(p[:, 0] * lat_std + lat_mean)
            pred_lngs.extend(p[:, 1] * lng_std + lng_mean)

    result = val_df[["filename", "country", "lat", "lng"]].reset_index(drop=True).copy()
    result["pred_lat"] = pred_lats
    result["pred_lng"] = pred_lngs
    result["error_km"] = haversine_km(
        result["lat"].to_numpy(), result["lng"].to_numpy(),
        result["pred_lat"].to_numpy(), result["pred_lng"].to_numpy(),
    )

    metrics = {
        "n": len(result),
        "median_km": float(result["error_km"].median()),
        "mean_km": float(result["error_km"].mean()),
        "acc_200km": float((result["error_km"] < 200).mean()),
        "acc_750km": float((result["error_km"] < 750).mean()),
    }
    return metrics, result


def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm = ckpt["norm"]

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    _, val_tf = get_transforms()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.val:
        metrics, result = score_val_set(model, Path(args.data_root), val_tf, **norm,
                                         val_frac=args.val_frac, batch_size=args.batch_size,
                                         workers=args.workers, device=device)
        print(f"n={metrics['n']}  median={metrics['median_km']:.1f} km  "
              f"mean={metrics['mean_km']:.1f} km  "
              f"<200km={metrics['acc_200km']:.1%}  <750km={metrics['acc_750km']:.1%}")
        csv_path = out_dir / "val_predictions.csv"
        result.to_csv(csv_path, index=False)
        print(f"Saved {len(result)} scored val predictions → {csv_path}")
    else:
        preds = predict_holdout(model, Path(args.data_root), val_tf, **norm,
                                batch_size=args.batch_size, workers=args.workers, device=device)
        csv_path = out_dir / "predictions.csv"
        preds.to_csv(csv_path, index=False)
        print(f"Saved {len(preds)} predictions → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train.py")
    p.add_argument("--data_root",  default="geo_dataset")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers",    type=int, default=1)
    p.add_argument("--val", action="store_true",
                   help="Score the labelled validation split instead of predicting the holdout set")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Must match the val_frac used at training time, to reconstruct the same split")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
