from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LoRAQKV(nn.Module):
    """Freeze fused QKV; adapt Q and V only. Zero update at initialization."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        dim = base.in_features
        self.dim, self.scale, self.enabled = dim, alpha / rank, True
        self.q_a = nn.Linear(dim, rank, bias=False)
        self.q_b = nn.Linear(rank, dim, bias=False)
        self.v_a = nn.Linear(dim, rank, bias=False)
        self.v_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.q_b.weight)
        nn.init.zeros_(self.v_b.weight)

    def forward(self, x):
        result = self.base(x)
        if not self.enabled:
            return result
        q, k, v = result.chunk(3, dim=-1)
        return torch.cat((q + self.scale * self.q_b(self.q_a(x)), k,
                          v + self.scale * self.v_b(self.v_a(x))), dim=-1)


class TinyBlock(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        a = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), -1)
        return x + self.proj(a @ v)


class TinyEncoder(nn.Module):
    """Explicit CPU smoke backend; never substituted for SAM automatically."""
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 24, 4, 4)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        x = self.patch(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2).reshape(b, c, h, w)


class Segmenter(nn.Module):
    def __init__(self, cfg, classes):
        super().__init__()
        self.cfg = cfg
        if cfg["backend"] == "sam":
            checkpoint = cfg.get("checkpoint")
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError("Supply model.checkpoint for real SAM; tiny is only for smoke tests")
            from segment_anything import sam_model_registry
            sam = sam_model_registry[cfg["variant"]](checkpoint=checkpoint)
            self.encoder = sam.image_encoder
            channels = 256
        else:
            self.encoder = TinyEncoder()
            channels = 24
        self.channels = channels
        self.encoder.requires_grad_(False)
        blocks = self.encoder.blocks
        if cfg["last_blocks"] > len(blocks):
            raise ValueError("last_blocks exceeds encoder depth")
        for block in blocks[-cfg["last_blocks"]:]:
            target = block.attn if cfg["backend"] == "sam" else block
            target.qkv = LoRAQKV(target.qkv, cfg["rank"], cfg["alpha"])
        # A native semantic head is required for prompt-free multiclass output.
        # SAM's prompt-conditioned decoder is deliberately not reinterpreted as one.
        self.head = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(64, classes, 1))

    def preprocess(self, images):
        if self.cfg["backend"] == "tiny":
            return images
        h, w = images.shape[-2:]
        size = self.cfg["input_size"]
        scale = size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        x = F.interpolate(images, (nh, nw), mode="bilinear", align_corners=False) * 255
        mean = x.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
        std = x.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
        return F.pad((x - mean) / std, (0, size - nw, 0, size - nh))

    def forward(self, images):
        z = self.encoder(self.preprocess(images))
        logits = self.head(z)
        if self.cfg["backend"] == "sam":
            size = self.cfg["input_size"]
            h, w = images.shape[-2:]
            scale = size / max(h, w)
            logits = F.interpolate(logits, (size, size), mode="bilinear", align_corners=False)
            logits = logits[..., :round(h * scale), :round(w * scale)]
        return F.interpolate(logits, images.shape[-2:], mode="bilinear", align_corners=False)

    @contextmanager
    def adapters_disabled(self):
        modules = [m for m in self.modules() if isinstance(m, LoRAQKV)]
        previous = [m.enabled for m in modules]
        try:
            for m in modules:
                m.enabled = False
            yield
        finally:
            for m, enabled in zip(modules, previous):
                m.enabled = enabled

    @torch.no_grad()
    def features(self, images, temperature=1.0):
        with self.adapters_disabled():
            z = self.encoder(self.preprocess(images)).flatten(2)
            weights = (z.norm(dim=1) / (z.shape[1] * temperature)).softmax(-1)
            return (z * weights[:, None]).sum(-1)

    def adaptation_state(self):
        names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_adaptation(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter/head checkpoint keys do not match architecture")
        self.load_state_dict(state, strict=False)


def segmentation_loss(logits, target, dice_weight=1.0):
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=-100)
    truth = F.one_hot(target.clamp_min(0), logits.shape[1]).permute(0, 3, 1, 2)
    prob = logits.softmax(1) * valid[:, None]
    truth = truth * valid[:, None]
    dims = (0, 2, 3)
    dice = (2 * (prob * truth).sum(dims) + 1e-6) / (prob.sum(dims) + truth.sum(dims) + 1e-6)
    return ce + dice_weight * (1 - dice.mean())
