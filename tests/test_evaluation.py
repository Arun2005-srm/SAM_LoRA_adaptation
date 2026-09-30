import numpy as np
import torch

from segmentation.config import DEFAULTS, merge
from segmentation.data import discover, split_records, read
from segmentation.engine import new_model, evaluate_bank
from smoke import make_data


def test_stitching_counts_pixels_once_and_never_reads_training_samples(tmp_path, monkeypatch):
    task = make_data(tmp_path / "data", tasks=1)[0]
    task.update(tile_size=24, stride=12)
    cfg = merge(DEFAULTS, {"model": {"backend": "tiny", "input_size": 32, "rank": 2, "last_blocks": 2},
                           "device": "cpu", "tasks": [task], "training": {"batch_size": 2}})
    model = new_model(cfg, 2, torch.device("cpu"))
    checkpoint = tmp_path / "expert.pt"
    torch.save({"base_id": "test", "task": task, "adapter": model.adaptation_state()}, checkpoint)
    splits = split_records(discover(task), {})
    forbidden = {r[k] for s in ["train", "val"] for r in splits[s] for k in ["image", "mask"]}
    from segmentation import data, engine
    def guarded_read(path, *args, **kwargs):
        assert str(path) not in forbidden, "Evaluator attempted to access training/validation pixels"
        return read(path, *args, **kwargs)
    monkeypatch.setattr(data, "read", guarded_read)
    monkeypatch.setattr(engine, "read", guarded_read)
    result = evaluate_bank(cfg, [str(checkpoint)], {task["name"]: splits}, tmp_path / "test", 0)
    assert result[task["name"]]["oracle"]["valid_pixels"] == 3 * 32 * 32
    assert result[task["name"]]["oracle"]["visualizations_saved"] == 0
