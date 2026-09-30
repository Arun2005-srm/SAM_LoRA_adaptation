import argparse
from segmentation.config import load_config
from segmentation.engine import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential task-specific SAM LoRAs without replay")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))
