import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    """Numpy haversine — used for reporting metrics."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def haversine_loss(pred, target, lat_mean, lat_std, lng_mean, lng_std, R=6371.0088):
    """Differentiable mean haversine distance (km) — used as training loss."""
    pred_lat = torch.deg2rad(pred[:, 0] * lat_std + lat_mean)
    pred_lng = torch.deg2rad(pred[:, 1] * lng_std + lng_mean)
    true_lat = torch.deg2rad(target[:, 0] * lat_std + lat_mean)
    true_lng = torch.deg2rad(target[:, 1] * lng_std + lng_mean)

    d = (torch.sin((true_lat - pred_lat) / 2) ** 2
         + torch.cos(pred_lat) * torch.cos(true_lat)
         * torch.sin((true_lng - pred_lng) / 2) ** 2)
    return (2 * R * torch.asin(torch.sqrt(torch.clamp(d, 0, 1)))).mean()


def run_epoch(model, loader, optimizer, device,
              lat_mean, lat_std, lng_mean, lng_std,
              aux_weight=0.3, train=True):
    model.train(train)
    total_loss = 0.0
    pred_lats, pred_lngs = [], []
    true_lats, true_lngs = [], []

    norm = dict(lat_mean=lat_mean, lat_std=lat_std, lng_mean=lng_mean, lng_std=lng_std)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, coords, countries in tqdm(loader, total=len(loader)):
            imgs, coords, countries = imgs.to(device), coords.to(device), countries.to(device)

            coord_pred, country_logits = model(pixel_values=imgs)

            loss = (haversine_loss(coord_pred, coords, **norm)
                    + aux_weight * F.cross_entropy(country_logits, countries))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(imgs)

            p = coord_pred.detach().cpu().numpy()
            t = coords.detach().cpu().numpy()
            pred_lats.extend(p[:, 0] * lat_std + lat_mean)
            pred_lngs.extend(p[:, 1] * lng_std + lng_mean)
            true_lats.extend(t[:, 0] * lat_std + lat_mean)
            true_lngs.extend(t[:, 1] * lng_std + lng_mean)

    avg_loss = total_loss / len(loader.dataset)
    dists = haversine_km(
        np.array(true_lats), np.array(true_lngs),
        np.array(pred_lats), np.array(pred_lngs),
    )
    return avg_loss, float(np.median(dists))
