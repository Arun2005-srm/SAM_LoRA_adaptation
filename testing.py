import argparse
import json
from pathlib import Path

from segmentation.engine import evaluate_bank, model_identity


def test_run(run, visualizations=None):
    run = Path(run)
    cfg = json.loads((run / "resolved_config.json").read_text())
    expected = json.loads((run / "environment.json").read_text())["base_id"]
    if model_identity(cfg) != expected:
        raise ValueError("Base checkpoint changed since training")
    bank = json.loads((run / "bank.json").read_text())
    manifests = {t["name"]: json.loads((run / "manifests" / f"{t['name']}.json").read_text()) for t in cfg["tasks"]}
    return evaluate_bank(cfg, bank, manifests, run / "test", visualizations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained bank on fixed held-out test scenes")
    parser.add_argument("--run", required=True)
    parser.add_argument("--visualizations", type=int, default=None, help="N scenes per task and inference mode")
    args = parser.parse_args()
    if args.visualizations is not None and args.visualizations < 0:
        parser.error("visualizations must be nonnegative")
    test_run(args.run, args.visualizations)
