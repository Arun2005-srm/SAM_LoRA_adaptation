from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from segmentation.config import DEFAULTS, merge, load_config, validate_dataset
from segmentation.data import discover, split_records, SegmentationDataset, decode_mask, offsets
from segmentation.models import Segmenter, LoRAQKV, segmentation_loss
from segmentation.metrics import Metrics
from segmentation.router import TaskVAE, route
from smoke import make_data

torch.set_num_threads(2)


@pytest.fixture
def tiny_cfg():
    return merge(DEFAULTS["model"], {"backend": "tiny", "input_size": 32, "rank": 2, "last_blocks": 2})


def test_split_reproducible_70_15_15_and_group_disjoint():
    records = [{"id": f"{i}_{j}", "group": str(i)} for i in range(20) for j in range(2)]
    a = split_records(records, {})
    assert a == split_records(records, {})
    assert [len(a[s]) for s in ["train", "val", "test"]] == [28, 6, 6]
    sets = [{r["group"] for r in a[s]} for s in a]
    assert not sets[0] & sets[1] and not sets[0] & sets[2]


def test_explicit_split_leakage_rejected():
    records = [{"id": str(i), "group": "same", "split": split} for i, split in enumerate(["train", "val", "test"])]
    with pytest.raises(ValueError, match="leakage"):
        split_records(records, {})


def test_palette_and_index_masks_preserve_classes_and_ignore():
    raw = np.array([[0, 7, 255]], dtype=np.uint8)[..., None]
    out = decode_mask(raw, {"label_map": {0: 0, 7: 1}, "ignore_values": [255]})
    assert out.tolist() == [[0, 1, -100]]
    rgb = np.array([[[0, 0, 0], [255, 0, 0]]])
    assert decode_mask(rgb, {"color_map": {"0,0,0": 0, "255,0,0": 1}}).tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="unmapped"):
        decode_mask(raw, {"label_map": {0: 0}})


def test_dataset_contract_and_padding(tmp_path):
    cfg = make_data(tmp_path, tasks=1)[0]
    cfg["tile_size"] = 40
    data = SegmentationDataset(discover(cfg), cfg)
    sample = data[0]
    assert sample["image"].shape == (3, 40, 40)
    assert sample["target"].dtype == torch.int64
    assert (sample["target"][32:] == -100).all()
    assert set(sample["target"].unique().tolist()) == {-100, 0, 1}


def test_unpaired_file_rejected(tmp_path):
    cfg = make_data(tmp_path, tasks=1)[0]
    from pathlib import Path
    (Path(cfg["root"]) / "masks" / "scene_000.png").unlink()
    with pytest.raises(ValueError, match="Missing mask"):
        discover(cfg)


def test_windows_cover_edges():
    for length in [12, 32, 33, 90]:
        covered = np.zeros(length, bool)
        for x in offsets(length, 32, 19):
            covered[x:x+32] = True
        assert covered.all()


def test_lora_zero_initialization_and_freezing():
    base = nn.Linear(8, 24)
    lora = LoRAQKV(base, 2, 2)
    x = torch.randn(2, 4, 8)
    assert torch.equal(lora(x), base(x))
    lora(x).sum().backward()
    assert base.weight.grad is None
    assert lora.q_b.weight.grad.abs().sum() > 0


def test_model_shape_gradients_and_base_preservation(tiny_cfg):
    model = Segmenter(tiny_cfg, 3)
    x = torch.rand(2, 3, 32, 32)
    y = torch.randint(3, (2, 32, 32))
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)
    segmentation_loss(out, y).backward()
    optimizer.step()
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "q_b" in n)


def test_frozen_router_features_invariant_to_adapter_updates(tiny_cfg):
    model = Segmenter(tiny_cfg, 2)
    x = torch.rand(1, 3, 32, 32)
    before = model.features(x)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRAQKV):
                m.q_b.weight.fill_(3)
                m.v_b.weight.fill_(2)
    assert torch.equal(before, model.features(x))


def test_adapter_roundtrip_and_task_isolation(tiny_cfg):
    torch.manual_seed(1)
    model = Segmenter(tiny_cfg, 2).eval()
    x = torch.rand(1, 3, 32, 32)
    state = model.adaptation_state()
    expected = model(x).detach().clone()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1)
    model.load_adaptation(state)
    assert torch.equal(expected, model(x))
    assert not any("base" in key or "patch" in key for key in state)


def test_metric_known_confusion_and_unknown_penalty():
    m = Metrics(2)
    m.update(torch.tensor([[[0, 1], [1, 2]]]), torch.tensor([[[0, 0], [1, 1]]]))
    r = m.compute()
    assert r["accuracy"] == .5
    # Sum TP=2; known-class unions are 2 and 3. Unknown contributes FN.
    assert r["iou"] == pytest.approx(2/5)
    assert r["miou"] == pytest.approx((.5 + 1/3)/2)
    assert r["confusion"] == [[1, 1, 0], [0, 1, 1]]


def test_perfect_boundary_and_ignored_pixels():
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[:, 10:20, 10:20] = 1
    y[:, :2] = -100
    m = Metrics(2)
    m.update(y.clamp_min(0), y)
    r = m.compute()
    assert r["miou"] == r["biou"] == r["accuracy"] == 1
    assert r["valid_pixels"] == 30 * 32


def test_all_ignored_loss_is_finite(tiny_cfg):
    model = Segmenter(tiny_cfg, 2)
    loss = segmentation_loss(model(torch.rand(1, 3, 32, 32)), torch.full((1, 32, 32), -100))
    assert loss.item() == 0
    loss.backward()


def test_vae_deterministic_scores_and_ood_rejection():
    vae = TaskVAE(8, 8, 2).eval()
    x = torch.rand(3, 8)
    assert torch.equal(vae.score(x, 1), vae.score(x, 1))
    selected, _ = route(x, [vae], [{"threshold": -1, "config": {"beta": 1}}])
    assert (selected == -1).all()


def test_overfit_tiny_batch(tiny_cfg):
    torch.manual_seed(42)
    model = Segmenter(tiny_cfg, 2)
    x = torch.zeros(2, 3, 16, 16)
    x[:, 0, :, 8:] = 1
    y = torch.zeros(2, 16, 16, dtype=torch.long)
    y[:, :, 8:] = 1
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.02)
    start = segmentation_loss(model(x), y).item()
    for _ in range(60):
        loss = segmentation_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert segmentation_loss(model(x), y).item() < start * .25
    assert (model(x).argmax(1) == y).float().mean() > .95


def test_official_sam_encoder_integration(tmp_path, monkeypatch):
    from segment_anything.modeling import ImageEncoderViT
    import segment_anything
    encoder = ImageEncoderViT(img_size=32, patch_size=8, embed_dim=32, depth=2,
                             num_heads=4, out_chans=256, window_size=0)
    monkeypatch.setitem(segment_anything.sam_model_registry, "test", lambda checkpoint: SimpleNamespace(image_encoder=encoder))
    path = tmp_path / "placeholder.pt"
    path.touch()
    cfg = {"backend": "sam", "variant": "test", "checkpoint": str(path),
           "rank": 2, "alpha": 2, "last_blocks": 1, "input_size": 32}
    model = Segmenter(cfg, 3)
    out = model(torch.rand(1, 3, 24, 32))
    assert out.shape == (1, 3, 24, 32)
    out.mean().backward()
    assert model.encoder.blocks[-1].attn.qkv.q_b.weight.grad is not None
    assert model.features(torch.rand(1, 3, 24, 32)).shape == (1, 256)


def test_config_rejects_typo_and_bad_class_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("trainig: {}")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    with pytest.raises(ValueError, match="index"):
        validate_dataset({"name": "x", "classes": ["a", "b"], "label_map": {0: 9}})


def test_raster_reader_grid_and_band_preservation(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from segmentation.data import read
    path = tmp_path / "x.tif"
    a = np.stack([np.full((8, 8), v, dtype=np.uint16) for v in [100, 200, 300, 400]])
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=4,
                       dtype="uint16", crs="EPSG:32632", transform=from_origin(0, 100, 10, 10)) as dst:
        dst.write(a)
    image, valid = read(path, [4, 2, 1], (1, 2, 3, 4))
    assert image.shape == (4, 3, 3)
    assert image[0, 0].tolist() == [400, 200, 100]
    assert valid.all()
