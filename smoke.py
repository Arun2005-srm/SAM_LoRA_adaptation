"""Exercise the entire pipeline with generated data, never claiming SAM accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from segmentation.config import load_config, dump_json
from segmentation.engine import train
from testing import test_run
from inspect_data import inspect


def make_data(root, count=20, tasks=2):
    root = Path(root).resolve()
    datasets = []
    for task in range(tasks):
        directory = root / f"task_{task+1}"
        (directory / "images").mkdir(parents=True, exist_ok=True)
        (directory / "masks").mkdir(exist_ok=True)
        rng = np.random.default_rng(42 + task)
        classes = ["background", "building"] + (["water"] if task else [])
        for i in range(count):
            mask = np.zeros((32, 32), dtype=np.uint8)
            x, y = rng.integers(3, 12, 2)
            mask[y:y+12, x:x+12] = 1
            if task:
                mask[23:29, 2:29] = 2
            palette = np.array([[30, 80, 35], [205, 160, 70], [25, 70, 225]])
            rgb = np.clip(palette[mask] + rng.normal(0, 3, (32, 32, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(directory / "images" / f"scene_{i:03d}.png")
            Image.fromarray(mask).save(directory / "masks" / f"scene_{i:03d}.png")
        datasets.append({"name": f"task_{task+1}", "root": str(directory), "classes": classes,
                         "label_map": {i: i for i in range(len(classes))}, "tile_size": 32,
                         "split": {"ratios": [0.7, 0.15, 0.15], "seed": 42}})
    return datasets


def run_smoke(output):
    torch.set_num_threads(2)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh smoke directory")
    output.mkdir(parents=True)
    tasks = make_data(output / "synthetic")
    raw = {"seed": 42, "device": "cpu", "output": str(output / "run"), "tasks": tasks,
           "model": {"backend": "tiny", "input_size": 32, "rank": 2, "alpha": 2, "last_blocks": 2},
           "training": {"epochs": 2, "batch_size": 4, "workers": 0, "learning_rate": 0.003, "amp": False},
           "router": {"epochs": 3, "hidden_dim": 16, "latent_dim": 4},
           "evaluation": {"visualizations": 2}}
    path = output / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    report = inspect(cfg)
    for task in report.values():
        assert [task[s]["scenes"] for s in ["train", "val", "test"]] == [14, 3, 3]
    train(cfg)
    run = output / "run"
    stages = json.loads((run / "continual_metrics.json").read_text())
    # Adding a task must not change the first expert's known-ID predictions.
    first = json.loads((run / "continual/stage_1/summary.json").read_text())
    last = json.loads((run / "continual/stage_2/summary.json").read_text())
    assert first["task_1"]["oracle"]["confusion"] == last["task_1"]["oracle"]["confusion"]
    for task in tasks:
        history = json.loads((run / task["name"] / "history.json").read_text())
        assert len(history) == 2
        for row in history:
            for split in ["train", "val"]:
                assert all(k in row[split] for k in ["loss", "accuracy", "iou", "miou", "biou"])
            assert row["epoch_seconds"] > 0 and row["rss_mb"] > 0
        checkpoint = torch.load(run / task["name"] / "best.pt", weights_only=False)
        assert not any(k in checkpoint for k in ["images", "masks", "embeddings", "replay_buffer"])
    results = test_run(output / "run", visualizations=2)
    for result in results.values():
        assert result["oracle"]["visualizations_saved"] == 2
        assert result["routed"]["visualizations_saved"] == 2
        assert result["oracle"]["valid_pixels"] == 3 * 32 * 32
    dump_json(output / "smoke_result.json", {"passed": True, "backend": "tiny CPU; NOT pretrained SAM",
              "tasks": 2, "epochs_per_task": 2, "split_scenes": [14, 3, 3], "visualizations_per_task_per_mode": 2,
              "old_expert_preserved": True, "epoch_fields_checked": True, "sample_cache_not_saved": True})
    print(f"SMOKE PASSED: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke")
    run_smoke(parser.parse_args().output)
