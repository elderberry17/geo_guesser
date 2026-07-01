"""
Interactive geo-localisation demo.
Loads best_model.pt, picks a training sample by index, runs inference
in real-time, and shows the image + a folium map with real vs predicted pin.

Usage:
    python demo.py
    python demo.py --data_root geo_dataset --checkpoint outputs/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import folium
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.data import COUNTRIES
from src.model import GeoModel
from src.utils import haversine_km

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VAL_TF = transforms.Compose([
    transforms.Resize(292),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_assets(data_root: Path, checkpoint: Path):
    device = (
        torch.device("mps")  if torch.backends.mps.is_available()  else
        torch.device("cuda") if torch.cuda.is_available()           else
        torch.device("cpu")
    )

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    norm = ckpt["norm"]

    model = GeoModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    labels = pd.read_csv(data_root / "train_labels.csv").reset_index(drop=True)

    return model, device, labels, norm


def make_map(true_lat, true_lng, pred_lat, pred_lng, dist_km: float) -> str:
    mid_lat = (true_lat + pred_lat) / 2
    mid_lng = (true_lng + pred_lng) / 2

    m = folium.Map(location=[mid_lat, mid_lng], zoom_start=5,
                   tiles="CartoDB positron")

    folium.Marker(
        [true_lat, true_lng],
        tooltip=f"Real ({true_lat:.4f}, {true_lng:.4f})",
        icon=folium.Icon(color="green", icon="map-marker"),
    ).add_to(m)

    folium.Marker(
        [pred_lat, pred_lng],
        tooltip=f"Predicted ({pred_lat:.4f}, {pred_lng:.4f})",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(m)

    folium.PolyLine(
        [[true_lat, true_lng], [pred_lat, pred_lng]],
        color="gray", weight=2, dash_array="6",
        tooltip=f"Error: {dist_km:.1f} km",
    ).add_to(m)

    folium.Circle(
        [true_lat, true_lng], radius=15000,
        color="green", fill=True, fill_opacity=0.15,
    ).add_to(m)

    return m._repr_html_()


def build_predict_fn(model, device, labels, norm, img_dir: Path):
    lat_mean = norm["lat_mean"];  lat_std = norm["lat_std"]
    lng_mean = norm["lng_mean"];  lng_std = norm["lng_std"]
    n = len(labels)

    def predict(idx: int):
        idx = max(0, min(int(idx), n - 1))
        row = labels.iloc[idx]

        pil_img = Image.open(img_dir / row["filename"]).convert("RGB")
        tensor  = VAL_TF(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            coord_pred, country_logits = model(pixel_values=tensor)

        p        = coord_pred.cpu().numpy()[0]
        pred_lat = float(p[0] * lat_std + lat_mean)
        pred_lng = float(p[1] * lng_std + lng_mean)
        true_lat = float(row["lat"])
        true_lng = float(row["lng"])

        pred_country = COUNTRIES[int(country_logits.argmax(dim=1).item())]
        dist = float(haversine_km(true_lat, true_lng, pred_lat, pred_lng))

        map_html = make_map(true_lat, true_lng, pred_lat, pred_lng, dist)

        info = (
            f"**Index:** {idx} / {n - 1}\n\n"
            f"**True country:** {row['country']}  |  "
            f"**Predicted country:** {pred_country}\n\n"
            f"**True coords:** {true_lat:.4f}, {true_lng:.4f}\n\n"
            f"**Predicted coords:** {pred_lat:.4f}, {pred_lng:.4f}\n\n"
            f"**Error:** {dist:.1f} km"
        )

        return pil_img, map_html, info

    return predict, n


def build_ui(predict_fn, n_samples: int) -> gr.Blocks:
    with gr.Blocks(title="Geo Demo") as demo:
        gr.Markdown("## Geo-Localisation Interactive Demo\n"
                    "Green = real location · Red = predicted location")

        with gr.Row():
            idx_slider = gr.Slider(0, n_samples - 1, value=0, step=1,
                                   label=f"Sample index (0 – {n_samples - 1})")
            idx_box    = gr.Number(value=0, label="Or type index", precision=0)

        run_btn = gr.Button("Predict", variant="primary")

        with gr.Row():
            img_out  = gr.Image(label="Image", type="pil", height=320)
            map_out  = gr.HTML(label="Map")

        info_out = gr.Markdown()

        def on_slider(v):  return gr.update(value=v)
        def on_box(v):     return gr.update(value=v)

        idx_slider.change(on_slider, idx_slider, idx_box)
        idx_box.change(on_box, idx_box, idx_slider)

        run_btn.click(predict_fn, inputs=idx_slider,
                      outputs=[img_out, map_out, info_out])
        idx_slider.release(predict_fn, inputs=idx_slider,
                           outputs=[img_out, map_out, info_out])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="geo_dataset")
    parser.add_argument("--checkpoint",  default="outputs/best_model.pt")
    parser.add_argument("--port",        type=int, default=7860)
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    checkpoint = Path(args.checkpoint)

    print("Loading model…")
    model, device, labels, norm = load_assets(data_root, checkpoint)
    print(f"Loaded {len(labels)} samples — device: {device}")

    predict_fn, n = build_predict_fn(model, device, labels, norm,
                                     data_root / "train")

    demo = build_ui(predict_fn, n)
    demo.launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
