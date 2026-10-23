"""Build the static demo site data in docs/ from out-of-fold predictions.

Every prediction shown comes from cv_pooled_oof_predictions.csv: the 5-fold
cross-validation run of the final recipe, so each image was predicted by a
model that never trained on it.

Usage:
    python scripts/build_demo.py \
        --oof ../geo_guesser_runs/outputs/cv_pooled_oof_predictions.csv \
        --data_root geo_dataset --per_country 50
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def parse_leaderboard(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*\**([^|*]+?)\**\s*\|\s*([^|]+?)\s*\|"
                     r"[^|]*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*\**([\d.]+)\**\s*\|", line)
        if m:
            rows.append({"rank": int(m[1]), "name": m[2].strip(), "backbone": m[3],
                         "change": m[4], "km": float(m[6])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path,
                    default=ROOT.parent / "geo_guesser_runs/outputs/cv_pooled_oof_predictions.csv")
    ap.add_argument("--data_root", type=Path, default=ROOT / "geo_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    ap.add_argument("--per_country", type=int, default=50)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    assert len(oof) == oof.filename.nunique()

    # Round-trip summary over every out-of-fold prediction.
    err = oof.error_km.to_numpy()
    summary = {
        "n": int(len(oof)),
        "median": float(np.median(err)), "mean": float(err.mean()),
        "lt200": float((err < 200).mean()), "lt750": float((err < 750).mean()),
        "countries": [
            {"country": c, "n": int(len(g)), "median": float(g.error_km.median()),
             "mean": float(g.error_km.mean()), "lt200": float((g.error_km < 200).mean())}
            for c, g in oof.groupby("country")
        ],
    }
    summary["countries"].sort(key=lambda r: r["median"])

    # Compact arrays for the explore map: every OOF point.
    countries = sorted(oof.country.unique())
    cidx = {c: i for i, c in enumerate(countries)}
    points = {
        "countries": countries,
        "rows": [[round(r.lat, 4), round(r.lng, 4), round(r.pred_lat, 4),
                  round(r.pred_lng, 4), cidx[r.country], round(r.error_km, 1)]
                 for r in oof.itertuples()],
    }

    # Stratified random sample for play mode (random, not cherry-picked).
    sample = (oof.groupby("country").sample(n=args.per_country, random_state=args.seed)
                 .sample(frac=1, random_state=args.seed))
    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    rounds = []
    for i, r in enumerate(sample.itertuples()):
        name = f"{i:04d}.jpg"
        im = Image.open(args.data_root / "train" / r.filename).convert("RGB")
        im.resize((args.size, args.size), Image.LANCZOS).save(
            img_dir / name, quality=78, optimize=True, progressive=True)
        rounds.append({"img": name, "country": r.country,
                       "lat": round(r.lat, 5), "lng": round(r.lng, 5),
                       "pred_lat": round(r.pred_lat, 5), "pred_lng": round(r.pred_lng, 5),
                       "err": round(r.error_km, 1)})

    data = {"summary": summary, "rounds": rounds,
            "leaderboard": parse_leaderboard(ROOT / "leaderboard.md")}
    (args.out / "data").mkdir(exist_ok=True)
    (args.out / "data" / "demo.json").write_text(json.dumps(data, separators=(",", ":")))
    (args.out / "data" / "points.json").write_text(json.dumps(points, separators=(",", ":")))
    print(f"{len(rounds)} rounds, {len(points['rows'])} points, "
          f"{len(data['leaderboard'])} leaderboard rows, median {summary['median']:.1f} km")


if __name__ == "__main__":
    main()
