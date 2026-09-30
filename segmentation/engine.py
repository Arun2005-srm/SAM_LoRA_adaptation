import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from .config import dump_json, fingerprint
from .data import SegmentationDataset, discover, split_records, loader, read, decode_mask, IGNORE
from .metrics import Metrics
from .models import Segmenter, segmentation_loss
from .router import fit_router, load_router, route
from .plots import curves, metric_plots, visual, continual_plot, router_plots, scene_distribution, routing_plot


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(cfg):
    value = cfg["device"]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def new_model(cfg, classes, device):
    # Shared tiny backbone must be identical across tasks and reloads.
    with torch.random.fork_rng():
        torch.manual_seed(cfg["seed"])
        return Segmenter(cfg["model"], classes).to(device)


def model_identity(cfg):
    path = cfg["model"].get("checkpoint")
    if cfg["model"]["backend"] == "tiny":
        return f"tiny-random-smoke-seed-{cfg['seed']}"
    if not path or not Path(path).is_file():
        raise FileNotFoundError("A real SAM checkpoint is required; set model.checkpoint")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_epoch(model, batches, device, cfg, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metrics = Metrics(model.head[-1].out_channels, cfg["evaluation"]["boundary_ratio"])
    total_loss = total_pixels = 0
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in batches:
            x, y = batch["image"].to(device), batch["target"].to(device)
            pixels = int((y >= 0).sum())
            if not pixels:
                continue
            with torch.autocast(device_type=device.type, enabled=cfg["training"]["amp"] and device.type == "cuda"):
                logits = model(x)
                loss = segmentation_loss(logits, y, cfg["training"]["dice_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            metrics.update(logits.argmax(1), y)
            total_loss += float(loss.detach()) * pixels
            total_pixels += pixels
    if not total_pixels:
        raise ValueError("Split has no valid labeled pixels")
    synchronize(device)
    result = metrics.compute()
    result["loss"] = total_loss / total_pixels
    return result, time.perf_counter() - start


def history_csv(history, path):
    rows = []
    for h in history:
        row = {k: v for k, v in h.items() if not isinstance(v, dict)}
        for split in ["train", "val"]:
            row.update({f"{split}_{k}": v for k, v in h[split].items() if isinstance(v, (float, int))})
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(cfg, output):
    """Index all tasks, without reading their pixels or computing future statistics."""
    manifests = {}
    for task in cfg["tasks"]:
        split = split_records(discover(task), task.get("split", {}))
        manifests[task["name"]] = split
        dump_json(output / "manifests" / f"{task['name']}.json", split)
    return manifests


def train(cfg):
    output = Path(cfg["output"])
    if (output / "resolved_config.json").exists():
        raise FileExistsError("Run exists. Choose a new output directory; existing runs are never overwritten")
    device = device_for(cfg)
    seed_all(cfg["seed"])
    base_id = model_identity(cfg)
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "resolved_config.json", cfg)
    manifests = prepare(cfg, output)
    dump_json(output / "environment.json", {"torch": str(torch.__version__), "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "base_id": base_id, "backend": cfg["model"]["backend"]})
    bank, continual = [], []
    for index, task in enumerate(cfg["tasks"]):
        name = task["name"]
        stage = output / name
        stage.mkdir()
        seed_all(cfg["seed"] + index)
        split = manifests[name]
        train_data = SegmentationDataset(split["train"], task, training=True)
        val_data = SegmentationDataset(split["val"], task)
        train_loader = loader(train_data, cfg["training"], True, cfg["seed"] + index)
        val_loader = loader(val_data, cfg["training"])
        model = new_model(cfg, len(task["classes"]), device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["training"]["amp"])
        best, history = -1, []
        print(f"\nTask {index + 1}/{len(cfg['tasks'])}: {name} | {len(train_data)} train / {len(val_data)} val patches | trainable={sum(p.numel() for p in params):,}", flush=True)
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            tr, trtime = run_epoch(model, train_loader, device, cfg, optimizer, scaler)
            va, vatime = run_epoch(model, val_loader, device, cfg)
            h = {"epoch": epoch, "train": tr, "val": va, "train_seconds": trtime, "val_seconds": vatime,
                 "epoch_seconds": time.perf_counter() - start, "learning_rate": optimizer.param_groups[0]["lr"],
                 "rss_mb": psutil.Process().memory_info().rss / 2**20,
                 "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                 "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0}
            history.append(h)
            artifact = {"adapter": model.adaptation_state(), "task": task, "model": cfg["model"], "seed": cfg["seed"],
                        "base_id": base_id, "manifest_hash": fingerprint(split), "epoch": epoch, "val_miou": va["miou"]}
            if va["miou"] > best:
                best = va["miou"]
                torch.save(artifact, stage / "best.pt")
            scheduler.step()
            # Current-task recovery checkpoint; never used as a replay buffer.
            torch.save({**artifact, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict()}, stage / "last.pt")
            dump_json(stage / "history.json", history)
            history_csv(history, stage / "history.csv")
            print(f"ep[{epoch}/{cfg['training']['epochs']}] "
                  f"Train acc={tr['accuracy']:.4f} loss={tr['loss']:.4f} IoU={tr['iou']:.4f} mIoU={tr['miou']:.4f} BIoU={tr['biou']:.4f} | "
                  f"Val acc={va['accuracy']:.4f} loss={va['loss']:.4f} IoU={va['iou']:.4f} mIoU={va['miou']:.4f} BIoU={va['biou']:.4f} | "
                  f"time train/val/epoch={trtime:.2f}/{vatime:.2f}/{h['epoch_seconds']:.2f}s | "
                  f"CPU RSS={h['rss_mb']:.1f}MiB GPU peak={h['gpu_peak_allocated_mb']:.1f}MiB reserved={h['gpu_peak_reserved_mb']:.1f}MiB", flush=True)
        curves(history, stage / "plots")
        artifact = torch.load(stage / "best.pt", map_location="cpu", weights_only=False)
        model.load_adaptation(artifact["adapter"])
        metric_plots(history[artifact["epoch"] - 1]["val"], task["classes"], stage / "plots" / "best_validation")
        if cfg["router"]["enabled"]:
            clean_train = loader(SegmentationDataset(split["train"], task), cfg["training"])
            print("Fitting current-task VAE; calibrating on validation features (no test access).", flush=True)
            artifact["router"] = fit_router(model, clean_train, val_loader, cfg["router"], device)
            dump_json(stage / "router_history.json", artifact["router"]["history"])
            router_plots(artifact["router"], stage / "plots")
            torch.save(artifact, stage / "best.pt")
            del clean_train
        bank.append(str((stage / "best.pt").resolve()))
        dump_json(output / "bank.json", bank)
        del train_loader, val_loader, train_data, val_data, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage_result = evaluate_bank(cfg, bank, manifests, output / "continual" / f"stage_{index + 1}", visualizations=0)
        continual.append({name: result["routed" if cfg["router"]["enabled"] else "oracle"] for name, result in stage_result.items()})
        dump_json(output / "continual_metrics.json", continual)
        continual_plot(continual, [t["name"] for t in cfg["tasks"]], output)
    forgetting = {}
    for task in cfg["tasks"][:-1]:
        name = task["name"]
        prior = [row[name]["miou"] for row in continual[:-1] if name in row]
        forgetting[name] = max(prior) - continual[-1][name]["miou"]
    dump_json(output / "forgetting.json", forgetting)
    print(f"Training complete: {output}. Run testing.py for final visualizations.", flush=True)
    return output


def activate(model, artifact, device):
    count = len(artifact["task"]["classes"])
    if model.head[-1].out_channels != count:
        model.head[-1] = nn.Conv2d(64, count, 1).to(device)
    model.load_adaptation(artifact["adapter"])
    model.eval()


@torch.no_grad()
def evaluate_bank(cfg, bank, manifests, output, visualizations=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg)
    artifacts = [torch.load(p, map_location="cpu", weights_only=False) for p in bank]
    if len({a["base_id"] for a in artifacts}) != 1:
        raise ValueError("Experts do not share the same base")
    model = new_model(cfg, len(artifacts[0]["task"]["classes"]), device)
    router_available = all("router" in a for a in artifacts)
    vaes = [load_router(a["router"], device) for a in artifacts] if router_available else []
    limit = cfg["evaluation"]["visualizations"] if visualizations is None else visualizations
    results = {}
    for task_index, artifact in enumerate(artifacts):
        task = artifact["task"]
        classes = task["classes"]
        results[task["name"]] = {}
        for mode in (["oracle", "routed"] if router_available else ["oracle"]):
            folder = output / task["name"] / mode
            folder.mkdir(parents=True, exist_ok=True)
            metrics = Metrics(len(classes), cfg["evaluation"]["boundary_ratio"])
            per_scene, route_counts, route_correct, route_total = [], [0] * (len(bank) + 1), 0, 0
            start = time.perf_counter()
            for scene_index, record in enumerate(manifests[task["name"]]["test"]):
                data = SegmentationDataset([record], task)
                raw, mask_valid = read(record["mask"])
                truth = decode_mask(raw, task)
                height, width = truth.shape
                probabilities = np.zeros((len(classes) + 1, height, width), dtype=np.float32)
                coverage = np.zeros((height, width), dtype=np.float32)
                valid_scene = np.zeros((height, width), dtype=bool)
                for batch in loader(data, cfg["training"]):
                    x = batch["image"].to(device)
                    if mode == "routed":
                        descriptors = model.features(x, cfg["router"]["temperature"])
                        selected, _ = route(descriptors, vaes, [a["router"] for a in artifacts])
                    else:
                        selected = torch.full((len(x),), task_index, device=device)
                    for expert in selected.unique().tolist():
                        positions = (selected == expert).nonzero().flatten()
                        mapped = torch.zeros((len(positions), len(classes) + 1, *x.shape[-2:]), device=device)
                        if expert < 0:
                            mapped[:, -1] = 1
                        else:
                            activate(model, artifacts[expert], device)
                            probs = model(x[positions]).softmax(1)
                            for source, label in enumerate(artifacts[expert]["task"]["classes"]):
                                dest = classes.index(label) if label in classes else len(classes)
                                mapped[:, dest] += probs[:, source]
                        for j, position in enumerate(positions.tolist()):
                            px, py, w, h = batch["window"][position].tolist()
                            probabilities[:, py:py+h, px:px+w] += mapped[j, :, :h, :w].cpu().numpy()
                            coverage[py:py+h, px:px+w] += 1
                            valid_scene[py:py+h, px:px+w] |= batch["target"][position, :h, :w].numpy() >= 0
                            route_counts[expert if expert >= 0 else len(bank)] += 1
                            route_correct += int(expert == task_index)
                            route_total += 1
                if (coverage == 0).any():
                    raise RuntimeError("Tiling left uncovered pixels")
                probabilities /= coverage[None]
                prediction = probabilities.argmax(0)
                truth[~valid_scene | ~mask_valid] = IGNORE
                metrics.update(torch.from_numpy(prediction)[None], torch.from_numpy(truth)[None])
                one = Metrics(len(classes), cfg["evaluation"]["boundary_ratio"])
                one.update(torch.from_numpy(prediction)[None], torch.from_numpy(truth)[None])
                scores = one.compute()
                per_scene.append({"id": record["id"], "miou": scores["miou"], "iou": scores["iou"], "biou": scores["biou"], "accuracy": scores["accuracy"]})
                if scene_index < limit:
                    rgb, _ = read(record["image"], task.get("bands", [1, 2, 3]))
                    rgb = np.clip(rgb.astype(np.float32) * task.get("scale", 1/255) + task.get("offset", 0), 0, 1)
                    visual(rgb, truth, prediction, classes, folder / f"visual_{scene_index+1:03d}.png",
                           f"{task['name']} | {record['id']} | {mode} | mIoU {scores['miou']:.3f}")
                    ImageSave(prediction, record, folder / f"prediction_{scene_index+1:03d}")
            result = metrics.compute()
            result.update({"seconds": time.perf_counter() - start,
                           "routing_accuracy": route_correct / max(route_total, 1),
                           "route_counts": route_counts, "route_labels": [a["task"]["name"] for a in artifacts] + ["unknown"],
                           "visualizations_saved": min(limit, len(per_scene)), "scenes": len(per_scene),
                           "aggregation": "scene probabilities averaged over overlapping windows"})
            dump_json(folder / "metrics.json", result)
            dump_json(folder / "per_scene.json", per_scene)
            if per_scene:
                with open(folder / "per_scene.csv", "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
                    writer.writeheader()
                    writer.writerows(per_scene)
            metric_plots(result, classes, folder)
            scene_distribution(per_scene, folder)
            results[task["name"]][mode] = result
            print(f"Test {task['name']} [{mode}] acc={result['accuracy']:.4f} IoU={result['iou']:.4f} mIoU={result['miou']:.4f} BIoU={result['biou']:.4f}", flush=True)
        if router_available:
            r = results[task["name"]]
            r["routing_gap_miou"] = r["oracle"]["miou"] - r["routed"]["miou"]
    dump_json(output / "summary.json", results)
    routing_plot(results, output)
    return results


def ImageSave(prediction, record, path):
    """Export label IDs, preserving the input raster grid when georeferenced."""
    if Path(record["image"]).suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(record["image"]) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="uint16", nodata=65535)
        # RGB photometric metadata is not valid for a single-channel label raster.
        profile.pop("photometric", None)
        with rasterio.open(path.with_suffix(".tif"), "w", **profile) as dst:
            dst.write(prediction.astype(np.uint16), 1)
    else:
        from PIL import Image
        Image.fromarray(prediction.astype(np.uint16)).save(path.with_suffix(".png"))
