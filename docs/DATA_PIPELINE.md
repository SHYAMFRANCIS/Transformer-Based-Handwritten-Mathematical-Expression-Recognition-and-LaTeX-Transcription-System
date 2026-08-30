# Data Pipeline — Pix2Tex TSV + Synthetic KaTeX

## Source Datasets (Combination)

| Dataset | Size | Format | Location |
|---|---|---|---|
| IM2LaTeX-100K | 100k | PNG + latex file | HF `lukas-blecher/im2latex-100k` → `E:\pix2tex_project\data\raw\im2latex` (Colab) |
| CROHME 2019 | 8k | InkML→PNG | `data/raw/crohme` |
| Synthetic KaTeX | 100–500 | Generated PNG | `data/mini/images/*.png` |

## TSV Spec (Pix2Tex-compatible)

Per `pix2tex/dataset/dataset.py:60`, pix2tex expects `equations.txt` indexed by `0.png` basename + `images/*.png`. For smoke we use simpler **TSV**:

```
E:\pix2tex_project\data\mini\images\0.png<TAB>x^2 + y^2 = z^2
E:\pix2tex_project\data\mini\images\1.png<TAB>\frac{a}{b}
...
```

Loader `src/data_loader.py:Im2LatexDataset` handles both: TSV primary, falls back to `equations.txt` indexed.

## Generation

```powershell
# Synthetic fallback (offline, always works, PIL text if matplotlib missing)
uv run python -c "from src.synthetic_generator import generate_samples; generate_samples(100)"
# → data/mini/images/0..99.png + equations.txt + all.tsv + train.txt(80)/val.txt(10)/test.txt(10) 80/10/10
```

`synthetic_generator.py:render_latex_to_png` chain: `matplotlib mathtext ($latex$)` → fallback `PIL Image.new+draw.text`. Uses 20 `LATEX_SAMPLES` with `\frac`, `\sum`, `\int`, `\sqrt` etc.

## Transforms

- **Train:** `pix2tex.dataset.transforms.train_transform` (ShiftScaleRotate, GridDistortion, RGBShift, GaussNoise, Normalize `(0.7931,0.1738)`, ToTensorV2) `p=...` — wrapped to handle `[:1]` grayscale.
- **Fallback:** `albumentations Resize(384,384)+Normalize+ToTensorV2` or `torchvision Resize(384,384)+ToTensor+Normalize`.
- All return `pixel_values: FloatTensor [3,384,384]` (gray repeated to 3).

## Split

- 80/10/10 via `random.shuffle` seed 42 in `generate_samples`.
- `download_mini_dataset(n=500)` ensures 50–500 train lines for smoke.

## Validation

```python
from src.data_loader import validate_dataset
validate_dataset() # checks tab, file exists, RGB, non-empty latex, 50-500 lines
```

Run `uv run python -c "from src.data_loader import get_dataloaders; dl,_,_=get_dataloaders(); print(next(iter(dl))['pixel_values'].shape)"` → `torch.Size([1,3,384,384])`.

## Full Pipeline (Colab)

Colab cell 3: `snapshot_download("lukas-blecher/im2latex-100k")` + Drive mirror `/content/drive/MyDrive/pix2tex_data` fallback synthetic; `Im2LatexDataset` with `batchsize=8`.
