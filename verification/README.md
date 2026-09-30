# Local verification

Executed 30 September 2026 on Windows, Python 3.10, CPU-only PyTorch 2.14.0.

- `python -m pytest -q --junitxml=verification/tests.xml`: **19 passed**.
- One upstream rasterio/Affine deprecation warning; no test failures.
- `python smoke.py --output runs/smoke_final`: **passed**.
- Synthetic data: two tasks, 20 scenes each, 14/3/3 train/validation/test.
- Two training epochs and three VAE epochs per task.
- Held-out evaluation after both stages and final standalone testing completed.
- Two visualizations per task per mode: eight final prediction panels total.
- Old expert's oracle confusion matrix unchanged after learning the second task.
- Saved epoch fields, scene pixel counts, and absence of sample caches checked.
- Generated learning curves and prediction panels were visually inspected.
- Python compilation and staged whitespace checks passed.

This verifies the implementation plumbing and a reduced official SAM encoder
integration, **not pretrained SAM performance**. No real remote-sensing dataset,
SAM checkpoint, CUDA training, or GPU benchmark was run. The smoke model is a
small randomly initialized CPU network. Its low synthetic segmentation scores
are expected from a two-epoch smoke run and are not research results.

Dataset YAMLs still need user paths and edition-specific verification. The
SeasoNet label vocabulary/band mapping must be completed before that task runs.
GitHub Actions configuration is provided; remote CI execution is not claimed.
