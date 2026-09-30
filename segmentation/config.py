import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


DEFAULTS = {
    "seed": 42, "device": "auto", "output": "runs/experiment",
    "model": {"backend": "sam", "variant": "vit_b", "checkpoint": None,
              "input_size": 1024, "rank": 8, "alpha": 8, "last_blocks": 4},
    "training": {"epochs": 30, "batch_size": 2, "workers": 0,
                 "learning_rate": 0.0001, "weight_decay": 0.01,
                 "amp": True, "grad_clip": 1.0, "dice_weight": 1.0},
    "router": {"enabled": True, "epochs": 10, "batch_size": 64,
               "learning_rate": 0.0005, "latent_dim": 64, "hidden_dim": 128,
               "beta": 16.5, "quantile": 0.97, "temperature": 1.0},
    "evaluation": {"visualizations": 12, "boundary_ratio": 0.02,
                   "unknown_action": "abstain"},
}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = set(DEFAULTS) | {"tasks"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown experiment keys: {set(raw) - allowed}")
    for section in ("model", "training", "router", "evaluation"):
        unknown = set(raw.get(section, {})) - set(DEFAULTS[section])
        if unknown:
            raise ValueError(f"Unknown {section} keys: {unknown}")
    cfg = merge(DEFAULTS, raw)
    tasks = []
    for item in cfg.get("tasks", []):
        if isinstance(item, str):
            p = (path.parent / item).resolve()
            item = yaml.safe_load(p.read_text(encoding="utf-8"))
            base = p.parent
        else:
            base = path.parent
        item = copy.deepcopy(item)
        for key in ("root", "manifest"):
            if item.get(key):
                expanded = os.path.expandvars(str(item[key]))
                if "$" in expanded or "YOUR_" in expanded:
                    raise ValueError(f"Set dataset {key}: {expanded}")
                p = Path(expanded)
                item[key] = str(p if p.is_absolute() else (base / p).resolve())
        validate_dataset(item)
        tasks.append(item)
    cfg["tasks"] = tasks
    if not tasks or len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Provide at least one task with a unique name")
    if cfg["model"]["backend"] not in {"sam", "tiny"}:
        raise ValueError("model.backend must be sam or tiny")
    if cfg["model"]["backend"] == "sam" and cfg["model"]["input_size"] != 1024:
        raise ValueError("Original SAM requires input_size=1024 in this implementation")
    for section, key in (("training", "epochs"), ("training", "batch_size"),
                         ("router", "epochs"), ("model", "rank"), ("model", "last_blocks")):
        if cfg[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    if not 0 < cfg["router"]["quantile"] < 1 or cfg["router"]["temperature"] <= 0:
        raise ValueError("Invalid router quantile or temperature")
    if cfg["evaluation"]["unknown_action"] != "abstain":
        raise ValueError("Semantic unknown-task inference currently supports abstain only")
    cfg["output"] = str(Path(os.path.expandvars(cfg["output"])).resolve())
    if cfg["model"].get("checkpoint"):
        cfg["model"]["checkpoint"] = str(Path(os.path.expandvars(cfg["model"]["checkpoint"])).resolve())
    return cfg


def validate_dataset(d):
    allowed = {"name", "root", "manifest", "images", "masks", "mask_suffix",
               "classes", "label_map", "color_map", "ignore_values", "bands",
               "scale", "offset", "nodata", "split", "group_regex", "tile_size",
               "stride", "augment", "assume_aligned", "pad"}
    if set(d) - allowed:
        raise ValueError(f"Unknown dataset keys: {set(d) - allowed}")
    for key in ("name", "classes"):
        if not d.get(key):
            raise ValueError(f"Dataset requires {key}")
    if len(d["classes"]) < 2 or len(set(d["classes"])) != len(d["classes"]):
        raise ValueError("classes must contain at least two unique names")
    if any(str(c).startswith("YOUR_") for c in d["classes"]):
        raise ValueError("Replace placeholder classes and verify the dataset label_map before training")
    if not re_safe_name(d["name"]):
        raise ValueError("Task names must use only letters, numbers, underscores and hyphens")
    if not d.get("label_map") and not d.get("color_map"):
        raise ValueError("Supply explicit label_map or color_map; no implicit thresholding")
    mapping = d.get("label_map") or d["color_map"]
    if any(int(v) < 0 or int(v) >= len(d["classes"]) for v in mapping.values()):
        raise ValueError("Mapped class IDs must index classes")
    split = d.get("split", {})
    if set(split) - {"ratios", "seed"}:
        raise ValueError("Unknown split keys")
    ratios = split.get("ratios", [0.70, 0.15, 0.15])
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("split.ratios must be three positive values summing to one")
    if len(d.get("bands", [1, 2, 3])) != 3:
        raise ValueError("SAM path requires exactly three selected bands; indices are 1-based")
    if any(b < 1 for b in d.get("bands", [1, 2, 3])):
        raise ValueError("Band indices are 1-based")
    if d.get("scale", 1 / 255) <= 0:
        raise ValueError("scale must be positive")
    tile = d.get("tile_size", 512)
    if tile <= 0 or not 0 < d.get("stride", tile) <= tile:
        raise ValueError("Require 0 < stride <= tile_size")


def re_safe_name(name):
    import re
    return isinstance(name, str) and re.fullmatch(r"[a-zA-Z0-9_-]+", name) is not None


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
