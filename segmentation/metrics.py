import math

import numpy as np
import torch
from torch.nn import functional as F


def boundary(mask, width):
    padded = F.pad(mask[None, None].float(), (width, width, width, width), value=0)
    eroded = 1 - F.max_pool2d(1 - padded, 2 * width + 1, stride=1)
    return mask & ~(eroded[0, 0] > 0.5)


class Metrics:
    """Dataset confusion counts, not averages of batch-level IoUs.

    The extra prediction column is abstention/unmapped expert output; it counts
    as an error instead of disappearing from the denominator.
    """
    def __init__(self, classes, boundary_ratio=0.02):
        self.classes, self.ratio = classes, boundary_ratio
        self.confusion = np.zeros((classes, classes + 1), dtype=np.int64)
        self.bi = np.zeros(classes, dtype=np.int64)
        self.bu = np.zeros(classes, dtype=np.int64)

    def update(self, prediction, target):
        pred, truth = prediction.detach().cpu().long(), target.detach().cpu().long()
        for p, t in zip(pred, truth):
            valid = t >= 0
            if not valid.any():
                continue
            p = p.clone()
            p[(p < 0) | (p >= self.classes)] = self.classes
            count = torch.bincount(t[valid] * (self.classes + 1) + p[valid],
                                   minlength=self.classes * (self.classes + 1))
            self.confusion += count.reshape(self.classes, self.classes + 1).numpy()
            width = max(1, round(self.ratio * math.hypot(*t.shape)))
            # Exclude neighborhoods touching ignored pixels from boundary scoring.
            invalid = (~valid)[None, None].float()
            safe = valid & (F.max_pool2d(invalid, 2 * width + 1, stride=1, padding=width)[0, 0] == 0)
            for c in range(self.classes):
                pb = boundary((p == c) & valid, width) & safe
                tb = boundary((t == c) & valid, width) & safe
                self.bi[c] += int((pb & tb).sum())
                self.bu[c] += int((pb | tb).sum())

    def compute(self):
        cm = self.confusion
        tp = cm[:, :self.classes].diagonal().astype(float)
        actual, predicted = cm.sum(1), cm.sum(0)[:self.classes]
        union = actual + predicted - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        dice = np.divide(2 * tp, actual + predicted, out=np.zeros_like(tp), where=(actual + predicted) > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
        biou = np.divide(self.bi, self.bu, out=np.zeros_like(tp), where=self.bu > 0)
        present = union > 0
        return {"accuracy": float(tp.sum() / max(cm.sum(), 1)),
                "iou": float(tp.sum() / max(union.sum(), 1)),
                "miou": float(iou[present].mean()) if present.any() else 0.0,
                "dice": float(dice[present].mean()) if present.any() else 0.0,
                "biou": float(biou[self.bu > 0].mean()) if (self.bu > 0).any() else 0.0,
                "per_class_iou": [float(v) if u else None for v, u in zip(iou, union)],
                "per_class_dice": [float(v) if u else None for v, u in zip(dice, union)],
                "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
                "per_class_biou": [float(v) if u else None for v, u in zip(biou, self.bu)],
                "confusion": cm.tolist(), "valid_pixels": int(cm.sum())}
