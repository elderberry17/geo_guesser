import torch
import torch.nn as nn
import torch.nn.functional as F


class LandPrior(nn.Module):
    """Soft "is this plausibly on land" prior.

    A Gaussian mixture (one component per country) fit to the training
    (lat, lng) points. Used to penalize predictions that land somewhere
    the training data never does — e.g. open water between countries.
    """

    def __init__(self, weights, means, covs, threshold):
        super().__init__()
        self.register_buffer("log_weights", torch.log(weights))
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    @classmethod
    def fit(cls, df, country_col="country", lat_col="lat", lng_col="lng",
            reg=1e-3, percentile=1.0):
        """Fit one Gaussian per country and calibrate a plausibility threshold
        as the given low percentile of training points' own log-likelihood."""
        weights, means, covs = [], [], []
        for _, group in df.groupby(country_col):
            pts = torch.tensor(group[[lat_col, lng_col]].to_numpy(), dtype=torch.float32)
            weights.append(len(group) / len(df))
            means.append(pts.mean(dim=0))
            covs.append(torch.cov(pts.T) + reg * torch.eye(2))

        prior = cls(
            weights=torch.tensor(weights, dtype=torch.float32),
            means=torch.stack(means),
            covs=torch.stack(covs),
            threshold=float("-inf"),
        )

        all_pts = torch.tensor(df[[lat_col, lng_col]].to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            train_log_probs = prior.log_prob(all_pts)
        prior.threshold.fill_(torch.quantile(train_log_probs, percentile / 100.0).item())
        return prior

    def log_prob(self, coords):
        """coords: (B, 2) lat/lng in raw degrees -> (B,) log-density under the mixture."""
        dist = torch.distributions.MultivariateNormal(self.means, covariance_matrix=self.covs)
        comp_log_probs = dist.log_prob(coords.unsqueeze(1)) + self.log_weights
        return torch.logsumexp(comp_log_probs, dim=1)

    def penalty(self, coords):
        """Hinge penalty: zero while a prediction is at least as plausible as
        the calibration threshold, grows once it falls below (e.g. in the sea)."""
        return F.relu(self.threshold - self.log_prob(coords)).mean()

    def implausible_rate(self, coords):
        """Fraction of predictions below the plausibility threshold — diagnostic only."""
        return (self.log_prob(coords) < self.threshold).float().mean()
