import os
import pytest
from PIL import Image

def test_tsv_parsing():
    from src.data_loader import MINI_DIR
    tsv = os.path.join(MINI_DIR, "train.txt")
    assert os.path.exists(tsv), f"Missing {tsv}"
    with open(tsv, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert 50 <= len(lines) <= 500 or len(lines)>0
    for line in lines[:3]:
        assert "\t" in line
        img_path, latex = line.split("\t", 1)
        assert os.path.exists(img_path)
        assert latex.strip() != ""

def test_images_load_rgb():
    from src.data_loader import MINI_DIR
    tsv = os.path.join(MINI_DIR, "train.txt")
    with open(tsv, encoding="utf-8") as f:
        img_path = f.readline().split("\t", 1)[0]
    img = Image.open(img_path).convert("RGB")
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0

def test_batch_shape():
    from src.data_loader import get_dataloaders
    train_loader, _, _ = get_dataloaders(batch_size=1, num_workers=0, pin_memory=False)
    batch = next(iter(train_loader))
    assert batch["pixel_values"].shape == (1, 3, 384, 384), f"Got {batch['pixel_values'].shape}"
    assert batch["labels"].dim() == 2
    assert batch["labels"].shape[0] == 1

def test_missing_file_handling():
    from src.data_loader import Im2LatexDataset
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("/nonexistent/path.png\t\\frac{a}{b}\n")
        f.write(os.path.join(r"E:\pix2tex_project\data\mini\images", "0.png") + "\tvalid latex\n")
        tmp = f.name
    # should skip nonexistent and keep 1 valid
    ds = Im2LatexDataset(tmp)
    assert len(ds) == 1
    os.unlink(tmp)
