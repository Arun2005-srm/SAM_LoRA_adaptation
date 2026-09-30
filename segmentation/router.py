import torch
from torch import nn
from torch.nn import functional as F


class TaskVAE(nn.Module):
    def __init__(self, features, hidden, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, features))

    def score(self, features, beta, stochastic=False):
        h = self.enc(features)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-12, 12)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if stochastic else mu
        recon = (self.dec(z) - features).square().mean(-1)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
        return recon + beta * kl


def fit_router(model, train_loader, val_loader, cfg, device):
    """Only current-task features live in this function; no retained feature bank."""
    model.eval()
    def extract(loader):
        return torch.cat([model.features(b["image"].to(device), cfg["temperature"]).cpu()
                          for b in loader])
    train_features, val_features = extract(train_loader), extract(val_loader)
    vae = TaskVAE(model.channels, cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg["learning_rate"])
    history = []
    for epoch in range(cfg["epochs"]):
        total = 0.0
        for idx in torch.randperm(len(train_features)).split(cfg["batch_size"]):
            x = train_features[idx].to(device)
            loss = vae.score(x, cfg["beta"], stochastic=True).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append({"epoch": epoch + 1, "loss": total / len(train_features)})
    vae.eval()
    with torch.no_grad():
        scores = torch.cat([vae.score(x.to(device), cfg["beta"]).cpu()
                            for x in val_features.split(cfg["batch_size"])])
    threshold = float(torch.quantile(scores, cfg["quantile"]))
    hist, edges = torch.histogram(scores, bins=20)
    return {"state": {k: v.cpu() for k, v in vae.state_dict().items()},
            "threshold": threshold, "features": model.channels, "config": cfg,
            "history": history, "calibration_count": len(val_features),
            "calibration_histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
            "scoring": "deterministic_posterior_mean_reconstruction_plus_beta_kl"}


def load_router(artifact, device):
    cfg = artifact["config"]
    vae = TaskVAE(artifact["features"], cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    vae.load_state_dict(artifact["state"])
    return vae.eval()


@torch.no_grad()
def route(features, vaes, artifacts):
    scores = torch.stack([v.score(features, a["config"]["beta"]) for v, a in zip(vaes, artifacts)], dim=1)
    best_score, selected = scores.min(1)
    thresholds = features.new_tensor([a["threshold"] for a in artifacts])
    selected[best_score > thresholds[selected]] = -1
    return selected, scores
