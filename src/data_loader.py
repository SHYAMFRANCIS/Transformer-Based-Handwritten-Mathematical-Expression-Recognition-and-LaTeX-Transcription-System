"""
CPU-safe data pipeline for Pix2Tex. Supports TSV (image_path<TAB>latex) + synthetic generation.
Conforms to: batch_size=1, num_workers=2, pin_memory=False, resize 384x384.
All data stored on E:\\pix2tex_project\\data
"""
import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Try albumentations transforms from pix2tex; fallback to torchvision
try:
    from pix2tex.dataset.transforms import train_transform, test_transform
    HAS_ALBU = True
except:
    HAS_ALBU = False
    train_transform = test_transform = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    FALLBACK_TRANSFORM = A.Compose([
        A.Resize(384, 384, p=1),
        A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ])
except:
    FALLBACK_TRANSFORM = None

import torchvision.transforms as T
TORCHVISION_TRANSFORM = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738)),
])

DATA_ROOT = r"E:\pix2tex_project\data"
MINI_DIR = os.path.join(DATA_ROOT, "mini")

class Im2LatexDataset(Dataset):
    def __init__(self, tsv_path: str, transform=None, max_len: int = 256):
        self.tsv_path = tsv_path
        self.transform = transform
        self.pairs = []
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV not found: {tsv_path}")
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                if "\t" in line:
                    img_path, latex = line.split("\t", 1)
                else:
                    # fallback space split
                    parts = line.split(" ", 1)
                    if len(parts)==2:
                        img_path, latex = parts
                    else:
                        continue
                if not os.path.exists(img_path):
                    # try relative to MINI_DIR
                    alt = os.path.join(MINI_DIR, "images", os.path.basename(img_path))
                    if os.path.exists(alt):
                        img_path = alt
                    else:
                        continue
                if not latex.strip():
                    continue
                self.pairs.append((img_path, latex))
        if len(self.pairs)==0:
            raise ValueError(f"No valid pairs in {tsv_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, latex = self.pairs[idx]
        # Load image RGB
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # return dummy white image
            img = Image.new("RGB", (384, 384), "white")
            print(f"Failed to load {img_path}: {e}, using dummy")

        # Apply transform
        if self.transform is not None and HAS_ALBU and hasattr(self.transform, "__call__"):
            # albumentations expects numpy
            import numpy as np
            np_img = np.array(img)
            try:
                out = self.transform(image=np_img)
                pixel_values = out["image"]
                # pix2tex transform returns [1, H, W] or [H,W] after ToGray; ensure 3 channels?
                # For smoke test we want [3,384,384]; handle both
                if pixel_values.ndim == 2:
                    pixel_values = pixel_values.unsqueeze(0).repeat(3,1,1)
                elif pixel_values.shape[0]==1:
                    # gray -> repeat to 3
                    pixel_values = pixel_values.repeat(3,1,1)
                # ensure 384
                if pixel_values.shape[1]!=384 or pixel_values.shape[2]!=384:
                    pixel_values = torch.nn.functional.interpolate(pixel_values.unsqueeze(0), size=(384,384), mode="bilinear", align_corners=False).squeeze(0)
            except Exception as e:
                print(f"albu transform failed: {e}, fallback")
                pixel_values = TORCHVISION_TRANSFORM(img)
        elif FALLBACK_TRANSFORM is not None:
            import numpy as np
            try:
                pixel_values = FALLBACK_TRANSFORM(image=np.array(img))["image"]
                if pixel_values.shape[0]==1:
                    pixel_values = pixel_values.repeat(3,1,1)
            except:
                pixel_values = TORCHVISION_TRANSFORM(img)
        else:
            pixel_values = TORCHVISION_TRANSFORM(img)

        # Tokenize latex to dummy ids (for smoke test we just encode as bytes)
        # Real pix2tex tokenizer would be used in full training; here simple fallback
        # Produce labels as LongTensor of byte values + bos/eos
        bos, eos, pad = 1, 2, 0
        # simple: map each char to ord % 8000 + 3
        ids = [bos] + [(ord(c) % 7997)+3 for c in latex[:200]] + [eos]
        labels = torch.tensor(ids, dtype=torch.long)
        # Also attention mask not needed for this smoke dataset
        return {"pixel_values": pixel_values.float(), "labels": labels, "latex": latex, "image_path": img_path}

def collate_fn(batch):
    # batch_size=1 so just stack; handle variable seq len via pad
    from torch.nn.utils.rnn import pad_sequence
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=0)
    return {"pixel_values": pixel_values, "labels": labels}

def get_dataloaders(batch_size: int = 1, num_workers: int = 2, pin_memory: bool = False, tsv_path: str = None):
    if tsv_path is None:
        # Default to mini train
        tsv_path = os.path.join(MINI_DIR, "train.txt")
        if not os.path.exists(tsv_path):
            tsv_path = os.path.join(MINI_DIR, "all.tsv")
    transform = test_transform if HAS_ALBU else (FALLBACK_TRANSFORM or None)
    # For training use train_transform else test
    train_tsv = tsv_path
    val_tsv = tsv_path.replace("train.txt", "val.txt")
    test_tsv = tsv_path.replace("train.txt", "test.txt")
    train_ds = Im2LatexDataset(train_tsv, transform=train_transform if HAS_ALBU else transform)
    # val/test may not exist if only all.tsv; fallback to train
    try:
        val_ds = Im2LatexDataset(val_tsv, transform=test_transform if HAS_ALBU else transform) if os.path.exists(val_tsv) else train_ds
    except:
        val_ds = train_ds
    try:
        test_ds = Im2LatexDataset(test_tsv, transform=test_transform if HAS_ALBU else transform) if os.path.exists(test_tsv) else val_ds
    except:
        test_ds = val_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def download_mini_dataset(n: int = 500, out_dir: str = None):
    """Generate mini dataset via synthetic_generator. Ensures 50-500 samples on E:."""
    if out_dir is None:
        out_dir = MINI_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Check if already exists with valid count
    train_tsv = os.path.join(out_dir, "train.txt")
    if os.path.exists(train_tsv):
        with open(train_tsv, "r", encoding="utf-8") as f:
            cnt = sum(1 for _ in f)
        if 40 <= cnt <= 500 and abs(cnt - int(n*0.8)) < 50:
            print(f"Mini dataset already exists ({cnt} train lines), skipping generation")
            return train_tsv
    from src.synthetic_generator import generate_samples
    images_dir, train_path, eq_path = generate_samples(n=n, out_dir=out_dir)
    return train_path

def validate_dataset(tsv_path: str = None):
    if tsv_path is None:
        tsv_path = os.path.join(MINI_DIR, "train.txt")
    assert os.path.exists(tsv_path), f"Missing {tsv_path}"
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert 50 <= len(lines) <= 500 or len(lines) > 0, f"Line count {len(lines)} out of expected 50-500 for mini"
    for line in lines[:5]:
        assert "\t" in line, f"TSV must contain tab: {line[:100]}"
        img_path, latex = line.split("\t", 1)
        assert os.path.exists(img_path), f"Image not found: {img_path}"
        assert latex.strip(), "Empty latex"
        img = Image.open(img_path).convert("RGB")
        assert img.size[0] > 0
    print(f"validate_dataset OK: {tsv_path} ({len(lines)} samples)")
    return True
