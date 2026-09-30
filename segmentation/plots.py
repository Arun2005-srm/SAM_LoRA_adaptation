from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def curves(history, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, key in zip(axes.flat, ["loss", "accuracy", "iou", "miou", "biou", "dice"]):
        for split, color in [("train", "#126782"), ("val", "#d97706")]:
            ax.plot(epochs, [h[split][key] for h in history], "o-", label=split, color=color)
        ax.set(title=key.upper() if "iou" in key else key.title(), xlabel="Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Training and validation | patch-level metrics", fontsize=15)
    fig.savefig(output / "learning_curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for key in ["train_seconds", "val_seconds", "epoch_seconds"]:
        axes[0].plot(epochs, [h[key] for h in history], "o-", label=key.removesuffix("_seconds"))
    axes[0].set(title="Wall-clock time", xlabel="Epoch", ylabel="Seconds")
    axes[0].legend()
    for key in ["rss_mb", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb"]:
        axes[1].plot(epochs, [h[key] for h in history], "o-", label=key)
    axes[1].set(title="Memory (CPU RSS is end-of-epoch)", xlabel="Epoch", ylabel="MiB")
    axes[1].legend(fontsize=7)
    axes[2].plot(epochs, [h["learning_rate"] for h in history], "o-")
    axes[2].set(title="Learning rate", xlabel="Epoch")
    fig.savefig(output / "resources.png")
    plt.close(fig)


def metric_plots(metrics, classes, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.75), 4), layout="constrained")
    x = np.arange(len(classes))
    for shift, key, color in [(-0.25, "iou", "#126782"), (0, "dice", "#52b69a"), (0.25, "biou", "#d97706")]:
        values = [np.nan if v is None else v for v in metrics[f"per_class_{key}"]]
        ax.bar(x + shift, values, 0.24, label=key.upper(), color=color)
    ax.set(xticks=x, xticklabels=classes, ylim=(0, 1), ylabel="Score", title="Per-class segmentation")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.savefig(output / "per_class.png")
    plt.close(fig)
    cm = np.array(metrics["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(classes)), max(5, len(classes) * 0.7)), layout="constrained")
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(classes) + 1), xticklabels=classes + ["Unknown"],
           yticks=range(len(classes)), yticklabels=classes, xlabel="Prediction", ylabel="Ground truth",
           title="Confusion matrix | normalized by true class")
    ax.tick_params(axis="x", rotation=35)
    for y in range(len(classes)):
        for x in range(len(classes) + 1):
            ax.text(x, y, f"{norm[y, x]:.2f}", ha="center", va="center", color="white" if norm[y, x] > .5 else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(output / "confusion_matrix.png")
    plt.close(fig)


def visual(image, target, prediction, classes, path, title):
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(classes)))[:, :3]
    def colorize(mask):
        result = np.full((*mask.shape, 3), 0.35)
        for i, color in enumerate(colors):
            result[mask == i] = color
        return result
    truth_rgb, pred_rgb = colorize(target), colorize(prediction)
    error = (target >= 0) & (prediction != target)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), layout="constrained")
    for ax, a, label in zip(axes, [image, truth_rgb, pred_rgb, .55 * image + .45 * pred_rgb, error],
                            ["Image", "Ground truth", "Prediction", "Overlay", "Error pixels"]):
        ax.imshow(a, cmap="Reds" if label == "Error pixels" else None, vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    handles = [Patch(color=color, label=name) for color, name in zip(colors, classes)]
    handles.append(Patch(color="0.35", label="Ignored / unknown"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(6, len(handles)), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def continual_plot(rows, tasks, output):
    matrix = np.full((len(rows), len(tasks)), np.nan)
    for i, row in enumerate(rows):
        for j, task in enumerate(tasks):
            if task in row:
                matrix[i, j] = row[task]["miou"]
    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 1.5), 5), layout="constrained")
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set(xticks=range(len(tasks)), xticklabels=tasks, yticks=range(len(rows)),
           yticklabels=[f"Stage {i + 1}" for i in range(len(rows))], title="Continual test mIoU", xlabel="Task")
    for i, j in np.ndindex(matrix.shape):
        if np.isfinite(matrix[i, j]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.savefig(Path(output) / "continual_miou.png")
    plt.close(fig)


def router_plots(artifact, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot([h["epoch"] for h in artifact["history"]], [h["loss"] for h in artifact["history"]], "o-")
    axes[0].set(title="Current-task VAE training", xlabel="Epoch", ylabel="Reconstruction + beta KL")
    hist = artifact["calibration_histogram"]
    axes[1].stairs(hist["counts"], hist["edges"], fill=True, alpha=.5)
    axes[1].axvline(artifact["threshold"], color="#d97706", label="Acceptance threshold")
    axes[1].set(title="Validation score distribution", xlabel="Deterministic routing score", ylabel="Patches")
    axes[1].legend()
    fig.savefig(Path(output) / "router_calibration.png")
    plt.close(fig)


def scene_distribution(rows, output):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.hist([r["miou"] for r in rows], bins=np.linspace(0, 1, 11), color="#126782", edgecolor="white")
    ax.set(title="Per-scene test mIoU", xlabel="mIoU", ylabel="Scenes", xlim=(0, 1))
    fig.savefig(Path(output) / "scene_score_distribution.png")
    plt.close(fig)


def routing_plot(results, output):
    tasks = list(results)
    if not tasks or "routed" not in results[tasks[0]]:
        return
    counts = np.array([results[t]["routed"]["route_counts"] for t in tasks], dtype=float)
    ratios = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(tasks)*1.5), 5), layout="constrained")
    im = ax.imshow(ratios, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(tasks)+1), xticklabels=tasks+["Unknown"], yticks=range(len(tasks)),
           yticklabels=tasks, title="Task routing | patch-level fractions", xlabel="Selected expert", ylabel="True task")
    for i, j in np.ndindex(ratios.shape):
        ax.text(j, i, f"{ratios[i,j]:.2f}", ha="center", va="center", color="white" if ratios[i,j] > .5 else "black")
    fig.colorbar(im, ax=ax)
    fig.savefig(Path(output) / "routing_confusion.png")
    plt.close(fig)
