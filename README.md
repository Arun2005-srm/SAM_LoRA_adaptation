# SAM LoRA adaptation for continual remote-sensing segmentation

A configuration-driven PyTorch project for **prompt-free multiclass semantic
segmentation**, task-specific SAM LoRA adapters, and VAE task routing. Train tasks
sequentially without replay; evaluate both known-task (oracle) and inferred-task
routing. Ordinary paired images, indexed/color masks, and windowed GeoTIFFs share
one trainer.

**Status:** implemented baseline, not a reproduction of CA-SAM results. Real
datasets and pretrained SAM weights are supplied by the user. Dataset YAMLs are
editable templates; verify the exact edition, label IDs and band order. The
SeasoNet template deliberately fails until its placeholder classes are replaced.

## Install

Use Python 3.10+ and install a matching PyTorch/torchvision build for your GPU from
https://pytorch.org/get-started/locally/ first. Then:

```bash
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install -r requirements.txt
```

For a CPU-only smoke environment:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python smoke.py --output runs/smoke
```

The smoke run creates two synthetic tasks, uses an explicitly named **tiny random
CPU encoder**, trains two epochs each, fits VAEs, evaluates all seen tasks,
exports plots and two visualizations per task/mode, and asserts the 14/3/3 scene
split. It does **not** test pretrained SAM accuracy or GPU memory feasibility.
Use a new output directory to repeat it.

## Configure and train

1. Put a pretrained original SAM ViT-B checkpoint under `checkpoints/` and set
   `model.checkpoint` in `configs/experiment.yaml`. No automatic checkpoint download.
2. Edit each dataset YAML: root, file patterns or CSV manifest, class IDs, bands,
   radiometric scale, spatial groups and tiling. Dataset paths resolve relative
   to the dataset YAML; model checkpoint and output paths resolve from the working
   directory. `${ENV_VAR}` placeholders are supported.
3. Select one or more dataset configs in the experiment's `tasks` list.
4. Inspect first, then train and test:

```bash
python inspect_data.py --config configs/experiment.yaml
python train.py --config configs/experiment.yaml
python testing.py --run runs/remote_sensing_001 --visualizations 12
```

Windows PowerShell environment example:

```powershell
$env:LANDCOVERAI_ROOT = 'D:/datasets/landcoverai'
$env:OPENEARTHMAP_ROOT = 'D:/datasets/OpenEarthMap'
$env:OPENEARTHMAP_MANIFEST = 'D:/datasets/OpenEarthMap/manifest.csv'
```

Remove unprepared tasks from `tasks` until their configs are ready. A compatible
new dataset requires only a new YAML and optionally a CSV, not trainer edits.

## Epoch output

```text
ep[1/30] Train acc=... loss=... IoU=... mIoU=... BIoU=... | Val acc=... loss=... IoU=... mIoU=... BIoU=... | time train/val/epoch=.../.../...s | CPU RSS=...MiB GPU peak=...MiB reserved=...MiB
```

- Accuracy is valid-pixel accuracy; **IoU is micro IoU**, mIoU is the unweighted
  mean of classes with nonzero union. Background participates when it is an
  explicitly declared class. Undefined absent-class scores are saved as null.
- BIoU is macro boundary IoU using inner morphological boundary bands of width
  `max(1, round(boundary_ratio * image_diagonal))`. Neighborhoods of ignored
  pixels are excluded. This definition and the width must match for comparisons.
- Loss is cross-entropy plus multiclass soft Dice. Logged epoch loss is the
  valid-pixel-weighted mean of batch losses (Dice itself is batch-dependent).
- Epoch metrics are patch-level. Final test metrics are computed once per scene
  after averaging overlapping patch probabilities, avoiding duplicate pixel counts.
- Train/val times include data loading, forward pass and metric computation;
  train also includes backward/optimizer work. CUDA is synchronized for timings.
  Epoch time excludes plot/checkpoint writing and subsequent VAE fitting.
- CPU RSS is process resident memory sampled at epoch end, **not CPU peak memory**.
  GPU values are PyTorch allocator peaks for the epoch, not whole-device usage.
  CPU-only runs report GPU memory as zero. GPU testing has not been claimed by
  the CPU smoke run.

## Results

```text
runs/experiment/
  resolved_config.json, environment.json, bank.json
  manifests/<task>.json
  <task>/
    best.pt, last.pt
    history.json, history.csv, router_history.json
    plots/
      learning_curves.png       # train/val loss, accuracy, IoU, mIoU, BIoU, Dice
      resources.png             # times, CPU/GPU memory, learning rate
      router_calibration.png    # VAE loss and validation-score histogram
      best_validation/          # per-class bars, normalized confusion matrix
  continual/stage_<n>/         # held-out evaluation after each training stage
  continual_metrics.json, continual_miou.png, forgetting.json
  test/
    summary.json, routing_confusion.png
    <task>/<oracle|routed>/
      metrics.json, per_scene.json, per_scene.csv
      per_class.png, confusion_matrix.png, scene_score_distribution.png
      visual_001.png ...        # image, truth, prediction, overlay, error pixels
      prediction_001.png|tif    # label IDs; georeference preserved for rasters
```

`--visualizations N` means up to N **test scenes per task per mode**, capped at
the number available; actual counts are recorded. Unknown predictions use class
ID `len(classes)`, shown in gray. Ignore pixels are excluded from metrics.
Raster export uses uint16 IDs; 65535 is reserved as nodata.

Plots describe actual measurements. ROC/AUROC and unknown-domain detection curves
are not fabricated from known-task-only data; they require a separate OOD protocol.

## Data and splitting

Folder datasets pair filename stems (optional mask suffix). Repeated stems or
unpaired images/masks fail explicitly. Use a CSV for arbitrary layouts:

```csv
id,image,mask,group
regionA_001,regionA/images/001.tif,regionA/labels/001.tif,regionA
regionB_001,regionB/images/001.tif,regionB/labels/001.tif,regionB
```

An optional `split` column (`train`, `val`, `test`) supplies an existing partition.
Otherwise deterministic **70/15/15 group splitting** is the default. Ratios are
approximate for indivisible groups; actual scene/group counts are reported by
inspection. With no group rule, each source image is one group. For pre-cut tiles
set `group_regex` (first capture is the group) or use a manifest. All seasonal
observations of one location should share a group for spatial generalization.

Split before tiling. Do not randomly divide already-overlapping tiles. Explicit
official partitions are preserved rather than silently changed to 70/15/15.
The same group crossing partitions is rejected. Coordinates are not automatically
used to detect overlapping neighboring scenes; supply correct grouping.

Supported input: PNG/JPEG and other Pillow-readable images, and GeoTIFF via
rasterio. Masks can be integer indexed or explicitly RGB color-mapped. GeoTIFF
reads are windowed, retaining native bit depth and selected **1-based** bands.
Categorical masks are never interpolated or thresholded. Grid mismatches fail;
reprojection is an explicit preprocessing responsibility. Set `assume_aligned`
only after verifying imagery/masks where only one has georeferencing.

Images must map to [0,1] using configured `scale` and `offset`; no silent clipping.
Three bands are required by the current SAM path. Multispectral raw storage is
readable through band selection, but full multispectral model adaptation, HDF5,
Zarr, vector rasterization, and temporal tensors are not implemented readers.
Training adds paired random horizontal/vertical flips. SAM normalization is
applied once inside the model. Invalid/padded targets use -100.

## Model and routing

The pretrained SAM image encoder stays frozen except for separate task-specific
LoRA Q/V updates in its last configured transformer blocks. LoRA B matrices start
at zero. Each task has its own small convolutional semantic head; SAM's original
prompt-conditioned mask decoder is **not** used. This is an explicit adaptation
for prompt-free semantic segmentation, not an unchanged CA-SAM architecture.

The VAE uses parameter-free norm-attention pooled **unadapted** SAM features.
It is trained only on current-task training patches. Validation features set a
configured quantile threshold; test data never fits it. Validation is also used
to select the semantic model, so this is a practical holdout baseline, not the
paper's five-fold threshold calibration or a formal OOD guarantee.

Training scores use stochastic posterior samples. Inference/calibration use
posterior-mean reconstruction + beta KL for deterministic scores. This is an
ELBO-style score, not a task posterior or exact marginal likelihood. Routing
selects minimum score then checks that expert's threshold, following CA-SAM's
selection order. Full frozen features followed by adapted features require two
encoder passes for routed inference; routing computation is included in testing
time. Cross-task raw-score calibration remains a research limitation.

Native label heads remain separate. For scoring wrong-expert predictions only,
exactly matching class names map across tasks; unmatched names count as unknown
errors. Synonyms are not automatically merged. This scoring convention does not
claim that all task label ontologies are equivalent.

Unknown-task routing **abstains**. Disabling LoRA does not magically give SAM a
land-cover semantic head. All-abstained pixels are counted as incorrect rather
than removed from the metric. Known-task-ID evaluation provides the routing-gap
reference. Negative forgetting can indicate improvement; it is not clipped.

## Strict no replay

Only current-task train/validation loaders are constructed for optimization.
Previous images, masks and cached embeddings are not supplied to subsequent
optimization. VAE feature tensors are temporary current-task memory and are not
saved; only aggregate calibration histograms are retained. Past adapters, heads,
VAEs and preprocessing specs are saved. No generative replay or shared learned
normalizer exists. The frozen base remains unchanged.

The evaluator separately reads historical **test** sets after each stage; results
do not update a model, router, scheduler or threshold. Manifests retain file paths
for reproducibility, not sample content. Training pre-indexes future file names
but does not fit future features/statistics. For a physically isolated stream,
provide only its authorized task sequence and evaluate separately.

The baseline stores current-task features in host RAM for VAE fitting and a full
scene probability accumulator during testing. Large scenes/datasets may need
streaming fitting or disk-backed accumulation; these are not hidden behind a
claim of constant memory. Checkpoints are trusted local artifacts (PyTorch load).

## Testing and limitations

Tests cover label decoding, splits/leakage, raster bands, edge coverage, padding,
LoRA identity/gradients, frozen-base preservation, frozen routing features,
adapter round-trip isolation, hand-calculated metrics, ignored-pixel handling,
VAE rejection, tiny-batch overfitting, and integration with the official SAM
`ImageEncoderViT` class using a reduced randomly initialized encoder.

Use `python -m pytest -q` and `python smoke.py --output runs/smoke`.
The latter is the end-to-end training/testing test. No real dataset results,
pretrained ViT-B validation, or GPU benchmark is implied by either.

`best.pt` selects maximum validation mIoU. `last.pt` also preserves optimizer,
scheduler and scaler states for inspection/recovery tooling; automatic interrupted
run resume is not currently exposed. Existing training runs are never overwritten.

## References

- [SAM](https://github.com/facebookresearch/segment-anything)
- [CA-SAM](https://arxiv.org/abs/2511.17201)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Workflow reference](https://github.com/Arun2005-srm/skin_lesion_segmentation_XAI)
