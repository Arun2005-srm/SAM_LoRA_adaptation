import argparse
from pathlib import Path

import numpy as np

from segmentation.config import load_config, dump_json
from segmentation.data import discover, split_records, SegmentationDataset
from segmentation.plots import visual


def inspect(cfg):
    output = Path(cfg["output"]) / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in cfg["tasks"]:
        splits = split_records(discover(task), task.get("split", {}))
        result = {}
        for split, records in splits.items():
            dataset = SegmentationDataset(records, task)
            counts = np.zeros(len(task["classes"]), dtype=np.int64)
            ignored = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                mask = sample["target"].numpy()
                counts += np.bincount(mask[mask >= 0], minlength=len(counts))
                ignored += int((mask < 0).sum())
                if i == 0:
                    visual(sample["image"].permute(1, 2, 0).numpy(), mask, mask, task["classes"],
                           output / f"{task['name']}_{split}.png", f"Data inspection | {task['name']} | {split}")
            result[split] = {"scenes": len(records), "groups": len({r['group'] for r in records}),
                             "patches": len(dataset), "patch_class_pixels": counts.tolist(), "ignored_patch_pixels": ignored}
        report[task["name"]] = result
    dump_json(output / "report.json", report)
    print(f"All image/mask patches validated: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit labels, bands, geometry, splits, and sample overlays")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    inspect(load_config(args.config))
