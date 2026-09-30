import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IGNORE = -100


def discover(cfg):
    root = Path(cfg.get("root", "."))
    records = []
    if cfg.get("manifest"):
        with open(cfg["manifest"], newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                records.append({"id": row["id"], "image": str((root / row["image"]).resolve()),
                                "mask": str((root / row["mask"]).resolve()),
                                "group": row.get("group") or row["id"], "split": row.get("split", "")})
    else:
        images = sorted(root.glob(cfg.get("images", "images/*")))
        masks = sorted(root.glob(cfg.get("masks", "masks/*")))
        masks_by_id = {}
        suffix = cfg.get("mask_suffix", "")
        for p in masks:
            key = p.stem.removesuffix(suffix) if suffix else p.stem
            if key in masks_by_id:
                raise ValueError(f"Duplicate mask stem {key}; use an explicit CSV manifest")
            masks_by_id[key] = p
        for p in images:
            if p.stem not in masks_by_id:
                raise ValueError(f"Missing mask for {p}")
            group = p.stem
            if cfg.get("group_regex"):
                match = re.search(cfg["group_regex"], str(p.relative_to(root)))
                if not match:
                    raise ValueError(f"group_regex does not match {p}")
                group = match.group(1)
            records.append({"id": p.stem, "image": str(p.resolve()),
                            "mask": str(masks_by_id.pop(p.stem).resolve()), "group": group, "split": ""})
        if masks_by_id:
            raise ValueError("Unpaired masks found; restrict the mask pattern or use a manifest")
    ids = [r["id"] for r in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Dataset empty or duplicate IDs; use unique IDs in a CSV manifest")
    for r in records:
        if not Path(r["image"]).is_file() or not Path(r["mask"]).is_file():
            raise FileNotFoundError(f"Missing image/mask for {r['id']}")
    return records


def split_records(records, cfg):
    result = {s: [] for s in ("train", "val", "test")}
    assigned = [bool(r.get("split")) for r in records]
    if any(assigned):
        if not all(assigned):
            raise ValueError("Manifest split column must be filled for every sample or none")
        for r in records:
            if r["split"] not in result:
                raise ValueError(f"Unknown split {r['split']}")
            result[r["split"]].append(r)
    else:
        groups = sorted({r["group"] for r in records})
        random.Random(cfg.get("seed", 42)).shuffle(groups)
        if len(groups) < 3:
            raise ValueError("At least three independent groups are required")
        ratios = cfg.get("ratios", [0.70, 0.15, 0.15])
        n = len(groups)
        ntrain = min(max(1, round(n * ratios[0])), n - 2)
        nval = min(max(1, round(n * ratios[1])), n - ntrain - 1)
        group_split = {g: "train" if i < ntrain else "val" if i < ntrain + nval else "test"
                       for i, g in enumerate(groups)}
        for r in records:
            result[group_split[r["group"]]].append(dict(r, split=group_split[r["group"]]))
    seen = set()
    for split, rows in result.items():
        groups = {r["group"] for r in rows}
        if seen & groups:
            raise ValueError("Spatial/temporal group leakage across splits")
        seen |= groups
        if not rows:
            raise ValueError(f"Empty {split} split")
    return result


def info(path):
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.height, ds.width, ds.crs, tuple(ds.transform)
    with Image.open(path) as im:
        return im.height, im.width, None, None


def read(path, bands=None, window=None):
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path) as ds:
            win = Window(*window) if window else None
            if bands and max(bands) > ds.count:
                raise ValueError(f"Missing selected bands in {path}")
            a = ds.read(indexes=bands, window=win)
            valid = np.all(ds.read_masks(indexes=bands, window=win) > 0, axis=0)
        return np.moveaxis(a, 0, -1), valid
    with Image.open(path) as im:
        if window:
            x, y, w, h = window
            im = im.crop((x, y, x + w, y + h))
        a = np.array(im)
    if a.ndim == 2:
        a = a[..., None]
    if bands:
        if max(bands) > a.shape[-1]:
            raise ValueError(f"Missing selected bands in {path}")
        a = a[..., np.array(bands) - 1]
    return a, np.ones(a.shape[:2], dtype=bool)


def decode_mask(raw, cfg):
    target = np.full(raw.shape[:2], IGNORE, dtype=np.int64)
    recognized = np.zeros(target.shape, dtype=bool)
    if cfg.get("color_map"):
        if raw.shape[-1] < 3:
            raise ValueError("Color labels require RGB mask files")
        for color, index in cfg["color_map"].items():
            rgb = np.array([int(v) for v in str(color).split(",")])
            pixels = np.all(raw[..., :3] == rgb, axis=-1)
            target[pixels] = int(index)
            recognized |= pixels
        for color in cfg.get("ignore_values", []):
            recognized |= np.all(raw[..., :3] == np.array([int(v) for v in str(color).split(",")]), axis=-1)
    else:
        if raw.shape[-1] != 1:
            raise ValueError("Indexed masks must be single-channel; use color_map for RGB masks")
        raw = raw[..., 0]
        for value, index in cfg["label_map"].items():
            pixels = raw == int(value)
            target[pixels] = int(index)
            recognized |= pixels
        for value in cfg.get("ignore_values", []):
            recognized |= raw == int(value)
    if not recognized.all():
        raise ValueError("Mask has unmapped values; explicitly map them or declare ignore_values")
    return target


def offsets(length, size, stride):
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


class SegmentationDataset(Dataset):
    def __init__(self, records, cfg, training=False):
        self.records, self.cfg, self.training = records, cfg, training
        self.tiles = []
        size = cfg.get("tile_size", 512)
        stride = cfg.get("stride", size)
        for i, record in enumerate(records):
            h, w, crs, transform = info(record["image"])
            mh, mw, mcrs, mt = info(record["mask"])
            if (h, w) != (mh, mw):
                raise ValueError(f"Image/mask size mismatch: {record['id']}")
            if crs is not None and mcrs is not None and (crs != mcrs or transform != mt):
                raise ValueError("Raster grids differ; reproject explicitly before training")
            if (crs is None) != (mcrs is None) and not cfg.get("assume_aligned", False):
                raise ValueError("Only one asset is georeferenced; set assume_aligned after verification")
            for y in offsets(h, size, stride):
                for x in offsets(w, size, stride):
                    self.tiles.append((i, x, y, min(size, w - x), min(size, h - y)))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        i, x, y, w, h = self.tiles[index]
        record = self.records[i]
        image, valid = read(record["image"], self.cfg.get("bands", [1, 2, 3]), (x, y, w, h))
        raw, mask_valid = read(record["mask"], window=(x, y, w, h))
        target = decode_mask(raw, self.cfg)
        valid &= mask_valid & np.isfinite(image).all(-1)
        if self.cfg.get("nodata") is not None:
            valid &= ~np.all(image == self.cfg["nodata"], axis=-1)
        target[~valid] = IGNORE
        image = image.astype(np.float32) * self.cfg.get("scale", 1 / 255) + self.cfg.get("offset", 0)
        image[~valid] = 0
        if np.any((image < 0) | (image > 1)):
            raise ValueError("Image values outside [0,1]; set appropriate scale/offset (no silent clipping)")
        if self.training and self.cfg.get("augment", True):
            for axis in (0, 1):
                if random.random() < 0.5:
                    image, target = np.flip(image, axis), np.flip(target, axis)
        size = self.cfg.get("tile_size", 512)
        image = np.pad(image, ((0, size - h), (0, size - w), (0, 0)))
        target = np.pad(target, ((0, size - h), (0, size - w)), constant_values=IGNORE)
        return {"image": torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1),
                "target": torch.from_numpy(np.ascontiguousarray(target)),
                "id": f"{self.cfg['name']}/{record['id']}@{x},{y}",
                "record_index": i, "window": torch.tensor([x, y, w, h])}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, cfg, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["workers"], pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)
