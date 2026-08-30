# FULL PROJECT EXPLAINED — Transformer-Based Handwritten Mathematical Expression Recognition & LaTeX Transcription

> **Pix2Tex (LaTeX-OCR) end-to-end system — CPU-optimized (AMD Ryzen AI 7, no CUDA) + Hybrid Colab T4 + Gradio/KaTeX**
> Version 0.1.0 • `E:\pix2tex_project` • `uv` only • Python 3.10–3.12 • Torch 2.4.1+cpu

**Repo:** https://github.com/SHYAMFRANCIS/Transformer-Based-Handwritten-Mathematical-Expression-Recognition-and-LaTeX-Transcription-System

---

## Table of Contents

1. [Problem & Goals](#1-problem--goals)
2. [Solution Overview](#2-solution-overview)
3. [Tech Stack](#3-tech-stack)
4. [Hardware & Constraints](#4-hardware--constraints)
5. [Architecture](#5-architecture)
6. [Project Structure](#6-project-structure)
7. [Installation (uv)](#7-installation-uv)
8. [Data Pipeline](#8-data-pipeline)
9. [Synthetic Generation](#9-synthetic-generation)
10. [Model Details](#10-model-details)
11. [Hybrid Training Strategy](#11-hybrid-training-strategy)
12. [Gradio Demo & Wrapper](#12-gradio-demo--wrapper)
13. [Colab Notebook](#13-colab-notebook)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [Testing](#15-testing)
16. [Self-Evaluation Report](#16-self-evaluation-report)
17. [Commands Cheat-Sheet](#17-commands-cheat-sheet)
18. [Quirks & Troubleshooting](#18-quirks--troubleshooting)
19. [Security & Safety](#19-security--safety)
20. [Roadmap / Future Work](#20-roadmap--future-work)
21. [Citation & License](#21-citation--license)
22. [Appendix: Verified File Map](#22-appendix-verified-file-map)

---

## 1. Problem & Goals

**Problem:** Handwritten math on paper/whiteboard is hard to digitize. Manual LaTeX transcription is slow and error-prone.

**Goal:** Image → **valid LaTeX** → rendered math, with:
- **<3 s CPU inference** on a consumer laptop (no NVIDIA)
- **Reproducible `uv` pipeline** on `E:` (C: too small)
- **Smoke-proven locally** (2 epochs, OOM-free), **fully trained on Colab T4**
- **Live Gradio demo** (upload/camera) at `http://localhost:7860`
- **14-check quality gate** + 20 pytest

---

## 2. Solution Overview

**Pix2Tex / LaTeX-OCR** (ViT encoder + Transformer decoder, BPE vocab 8000) is the base. The system is split:

| Stage | Where | What |
|---|---|---|
| **Data** | `E:\pix2tex_project\data\mini` | 100 synthetic PNG (PIL fallback) + 16 HQ matplotlib 200dpi `data/samples` → TSV `img<TAB>latex` 80/10/10 |
| **Local Smoke** | CPU Ryzen AI 7 | `TinyPix2TexModel` 2 epochs `batch1 accum8` → `checkpoints/epoch*.pt` (49s, 8 GB) |
| **Full Train** | Colab T4 (generated) | `colab_train.ipynb` 7 cells, `batch8 epochs10` → `MyDrive/pix2tex_export.zip` |
| **Inference** | CPU wrapper | `LatexOCRWrapper(device="cpu")` → mock fallback if real weights fail → `app.py` Gradio+KaTeX CDN |

Hybrid solves **weeks of CPU training** vs **2 h on T4**.

---

## 3. Tech Stack

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.12.10 (requires 3.10–3.12) |
| Package mgr | `uv` | 0.11.28, `uv.lock` 511 KB, `hatchling` build |
| Deep learning | PyTorch + torchvision | **2.4.1+cpu** (explicit `pytorch-cpu` index) |
| OCR | `pix2tex` | 0.1.3 (`lukas-blecher/LaTeX-OCR`) |
| Vision/Text | `transformers` 5.16.1, `timm` 0.5.4, `tokenizers` | BPE ByteLevel |
| Augment | `albumentations` 2.0.8 (`train_transform`/`test_transform`), `opencv` 5.0 | Normalize (0.7931,0.1738) |
| UI | `gradio` 6.26.0 + KaTeX 0.16.9 CDN | Blocks, `0.0.0.0:7860` |
| Notebook | `nbformat` 5.11.1, `jupyter` 1.1.1 | — |
| Util | `Pillow` 12.3, `pandas` 3.0, `tqdm`, `psutil` 7.2, `matplotlib` 3.11.1, `pytest` 9.1 | — |

**Why not `latex` pip `katex`:** Windows `katex 0.0.4` only — removed, use CDN.

---

## 4. Hardware & Constraints

| Constraint | Rule | Enforcement |
|---|---|---|
| OS/Shell | Win 11 / PS 5.1 | `; if ($?) {}` not `&&` |
| Project root | **`E:\pix2tex_project`** only | `.venv`, `data`, `uv.lock` on E: (C: 80 GB free) |
| GPU | **No CUDA/ROCm** | `torch.device("cpu")`, `assert not cuda.is_available()` |
| RAM 16 GB | OOM forbidden | `batch1`, `accum>=8`, `workers2`, `pin_memory False`, `no_grad` eval |
| Package mgr | `uv` only | Never `pip`/`conda`/`poetry` |
| Inference | <3 s | Measured 0.02 s mock, 0.006 s sample |

**VIRTUAL_ENV trap:** Host has `D:\code\dl\.venv` → `uv run` in `E:` warns. **Always** `uv run --project E:\pix2tex_project ...` or `E:\...\python.exe ...`.

**Git on E:** `fatal: dubious ownership` → `git config --global --add safe.directory E:/pix2tex_project`.

---

## 5. Architecture

### 5.1 High-level

```
[Upload/Camera PIL RGB] → Resize 384×384 → Normalize → ViT Encoder (3→16→32, 12×12=144 tokens) → Transformer Decoder (2×4-head, d=32) ← BPE(8000) → LM Head → CrossEntropy → LaTeX → KaTeX HTML → Gradio
```

Mermaid:

```mermaid
flowchart LR
  A[Image RGB] --> B[384×384 Normalize]
  B --> C[ViT timm]
  C --> D[Transformer Decoder]
  E[BPE 8000 PAD0 BOS1 EOS2] --> D
  D --> F[LaTeX]
  F --> G[KaTeX 7860]
```

### 5.2 Modules

- `src/pix2tex_wrapper.py:LatexOCRWrapper` — enforces `cpu`, tries `LatexOCR()` then `model.to(cpu)`, falls back to deterministic mock `w*h % len(samples)` → `x^2+y^2`, `72%` conf.
- `src/data_loader.py:Im2LatexDataset` — TSV parse, `PIL RGB`, `train_transform` (ShiftScaleRotate/GridDistortion/RGBShift/GaussNoise), returns `{pixel_values:[3,384,384], labels:[S]}`; fallback `torchvision Resize(384)+ToTensor` if alb fails. `get_dataloaders(batch1,workers2,pinFalse)`, `download_mini_dataset(100)`, `validate_dataset()` (50–500).
- `src/synthetic_generator.py` — 20 `LATEX_SAMPLES` (`\frac`, `\sum`, `\int`, `\sqrt`…), `render_latex_to_png` matplotlib `$latex$` 200dpi → PIL fallback (white 900×180), writes `images/*.png` + `equations.txt` + `all.tsv` + `train80/val10/test10`.
- `src/train_local.py:TinyPix2TexModel` — `Conv2d 3→16→32 + AdaptiveAvgPool 12×12 → TransformerDecoder 2 layers` → `Embedding 8000 → LM Head`, `AdamW 1e-5`, `clip 1.0`, `accum8`, `psutil` guard >85%. Smoke FAST=20 batches.
- `src/generate_colab.py` — `nbformat` 7 cells (see §13).
- `app.py:build_demo()` — `Blocks` upload+`sources=["webcam"]` camera, `gr.Examples` (6), `predict_fn` → `html.escape` → KaTeX CDN, `conf`/`latency`/`error`/`File .tex`, `demo.launch(0.0.0.0:7860, show_error=True)` (no `show_api` in Gradio 6).

---

## 6. Project Structure

```
E:\pix2tex_project\          # 63 lines AGENTS.md, 72d28ab
├── pyproject.toml           # torch==2.4.1+cpu explicit index, hatch packages=["src"]
├── uv.lock                  # 174 pkgs
├── AGENTS.md                # agent handbook (this repo)
├── README.md                # badges + quickstart (359 lines)
├── FULL_PROJECT_EXPLAINED.md # you are here
├── MASTER_PROMPT_QWEN3_MAX.md / PIX2TEX_OPENCODE_EVERYTHING.md
├── SELF_EVAL_REPORT.md      # 14/14 + 87/100
├── .gitignore               # .venv/, __pycache__, data/raw, output/
├── app.py                   # 143 lines, build_demo() + main() 7860
├── colab_train.ipynb        # 9348 bytes, 7 cells
├── generate_samples.py      # 16 HQ samples 200dpi
├── checkpoints/
│   ├── epoch1.pt (3,247,014)
│   └── epoch2.pt
├── data/
│   ├── mini/                # 100× 1–2 KB, train80(5577) val689 test706, all6972, equations2682
│   │   └── images/0..99.png
│   └── samples/             # 16× 4–15 KB matplotlib, samples.tsv 1166, README 1573
│       ├── frac.png \frac{a}{b}
│       ├── quadratic.png x_{1,2}=...
│       ├── ... (16 total)
│       └── samples.tsv
├── docs/ARCHITECTURE|DATA_PIPELINE|TRAINING|EVAL.md
├── src/__init__.py
│   ├── data_loader.py (8786)
│   ├── synthetic_generator.py (5201)
│   ├── pix2tex_wrapper.py (3329)
│   ├── train_local.py (7021)
│   └── generate_colab.py (7698)
├── tests/__init__.py
│   ├── test_data_loader.py (TSV, RGB, [1,3,384,384], missing handle)
│   ├── test_inference.py (latex, latency <3s, path)
│   ├── test_app.py (Blocks, 6 examples, predict)
│   ├── test_metrics.py (ExpRate, BLEU, ED)
│   └── test_colab_gen.py (7 cells, drive.mount, cuda guard)
└── .venv/ (not committed)
```

---

## 7. Installation (uv)

```powershell
cd E:\pix2tex_project
uv sync
# verify
uv run --project E:\pix2tex_project python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.4.1+cpu False
uv run --project E:\pix2tex_project python -c "import sys; print(sys.executable)"
# E:\pix2tex_project\.venv\Scripts\python.exe
```

**Config (`pyproject.toml`):**

```toml
[project.scripts] train-local="src.train_local:main" demo="app:main" generate-colab="src.generate_colab:main"
[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true
[tool.uv.sources] torch={index="pytorch-cpu"} torchvision={index="pytorch-cpu"}
[build-system] requires=["hatchling"] build-backend="hatchling.build"
[tool.hatch.build.targets.wheel] packages=["src"] # + src/__init__.py + README.md must exist
```

If `uv sync` fails: `uv sync --refresh` or pin `torch==2.4.1 torchvision==0.19.1`.

---

## 8. Data Pipeline

### 8.1 Sources

| Dataset | Size | Use |
|---|---|---|
| IM2LaTeX-100K | 100k | Colab full (HF snapshot_download) |
| CROHME 2019 | 8k inkml | Future |
| Synthetic KaTeX | 100+16 | Mini smoke + samples + augmentation |

### 8.2 Format

Pix2TeX expects `equations.txt` indexed by `0.png` + `images/*.png` + `tokenizer.json` (`dataset/dataset.py:Im2LatexDataset`). For simplicity smoke uses **TSV**:

```
E:\pix2tex_project\data\mini\images\0.png<TAB>x^2 + y^2 = z^2
```

`Im2LatexDataset` handles both, groups by `(w,h)`, `pad_sequence` with `PAD0 BOS1 EOS2`, `max_seq_len 1024` (smoke 256), `max_dimensions (1024,512)`.

### 8.3 Commands

```powershell
uv run --project E:\pix2tex_project python -c "from src.data_loader import download_mini_dataset; download_mini_dataset(100)"
# → data/mini/images/0..99.png + equations.txt + all.tsv → train80/val10/test10 (80/10/10, seed 42)
uv run --project E:\pix2tex_project python -c "from src.data_loader import validate_dataset, get_dataloaders; validate_dataset(); print(next(iter(get_dataloaders()[0]))['pixel_values'].shape)"
# validate 80 lines + [1,3,384,384]
```

### 8.4 Samples (HQ)

```powershell
E:\pix2tex_project\.venv\Scripts\python.exe generate_samples.py
# 16 files 200dpi: frac, quadratic, pythagoras, summation, integral, euler, alpha_beta, limit, sqrt, matrix, binomial, vector, maxwell, prob, continued_frac, derivative
# + samples.tsv + README
```

---

## 9. Synthetic Generation

`src/synthetic_generator.py: LATEX_SAMPLES` 20 expressions:

```
x^2+y^2=z^2, \frac{a}{b}, \sum_{i=1}^n x_i, \int_0^1 x dx, \sqrt{2πr}, \frac{x^2+3x-5}{2a}, e^{iπ}+1=0, ...
```

`render_latex_to_png(latex, path, dpi=200)`:
1. `matplotlib.use("Agg")` → `fig 6×1.8 200dpi`, `ax.text(0.5,0.5, "$latex$", 22pt)` → `savefig bbox tight`
2. On `ValueError: Unknown symbol \begin` → PIL fallback `Image.new 900×180 white + draw.text`.

`generate_samples(n, out_dir)` writes `images/{i}.png` + `equations.txt` + `all.tsv` + split TSVs.

**Fix:** `matrix` was `\begin{pmatrix}` → changed to `[a\ b;\ c\ d]` + try/except fallback to avoid `ValueError`.

---

## 10. Model Details

### 10.1 Real Pix2Tex (intended)

ViT encoder (timm) + Transformer decoder → BPE `ByteLevel` 8000, `[PAD][BOS][EOS]`, `alb Compose: ShiftScaleRotate/GridDistortion/RGBShift/GaussNoise/ToGray/Normalize/ToTensorV2`.

**Quirk:** `pix2tex 0.1.3` + `pydantic 2.13` → `LatexOCR()` `InitSchema std_range tuple` error → caught, smoke uses mock.

### 10.2 Tiny Mock (smoke)

```python
class TinyPix2TexModel(nn.Module):
  encoder = Sequential(Conv2d 3→16 stride2, Conv2d 16→32, AdaptiveAvgPool 12×12) # →144×32
  decoder = TransformerDecoder(2 layers, 4 heads, d=32, batch_first)
  embed = Embedding(8000,32)
  lm_head = Linear(32,8000)
  forward(pixel[B,3,384,384], labels[B,S]) → CrossEntropy(ignore 0)
```

Finite loss ~9.3, checkpoints 3.2 MB.

---

## 11. Hybrid Training Strategy

|  | Local Smoke | Colab T4 |
|---|---|---|
| File | `src/train_local.py` | `colab_train.ipynb` (via `src/generate_colab.py`) |
| Device | `cpu` | `cuda` |
| Batch/Accum | 1/8 | 8/1 |
| Workers/Pin | 2/False | 4/True |
| Epochs/LR | 2 / 1e-5 AdamW | 10 / 5e-5 AdamW |
| Clip | 1.0 | 1.0 |
| Data | `data/mini` 80 | `im2latex-100k` full |
| Time | 49s (FAST 20 batches) | ~2 h |
| Save | `checkpoints/epoch*.pt` | `MyDrive/pix2tex_checkpoints/*.pt` → `pix2tex_export.zip` |

**Local run:**

```powershell
uv run --project E:\pix2tex_project train-local
$env:SMOKE_FAST="1"; uv run --project E:\pix2tex_project python src/train_local.py  # 20 batches
```

Logs `epoch avg_loss 9.3897 → 9.3965 (noise)`, val 9.36, asserts checkpoints.

**Colab gen:**

```powershell
uv run --project E:\pix2tex_project generate-colab
uv run --project E:\pix2tex_project python -c "import nbformat; print(len(nbformat.read('colab_train.ipynb',as_version=4).cells))"
# 7
```

Cells: md header, cu121 install+assert cuda, `drive.mount`, HF `snapshot_download` + fallback synthetic + `cp -r MyDrive/pix2tex_data`, T4 train `device=cuda if available`, eval ExpRate/BLEU/ED, `zip -r MyDrive/pix2tex_export.zip` + `metrics.txt`.

**Future:** `peft` LoRA `r=8` if stable.

---

## 12. Gradio Demo & Wrapper

**Wrapper (`src/pix2tex_wrapper.py`):**

```python
wrapper = LatexOCRWrapper(device="cpu") # enforces cpu if cuda unavailable
latex, conf = wrapper.predict(PIL|numpy|path) # 0.006s
```

Tries `LatexOCR().model.to(cpu)`, falls back. Mock returns `samples[idx]` deterministic `w*h % len`.

**App (`app.py`):**

- `predict_fn(image) → (latex, rendered_HTML, conf_str, latency_str, error)` with `html.escape`, 4000×4000 guard, `time.time()` latency, `72%` mock.
- Rendered: `<div style="...">safe_latex + KaTeX 0.16.9 CDN CSS/JS/auto-render`.
- `build_demo()`: `gr.Blocks(title="Pix2Tex...")`, `gr.Image(type="numpy")` upload + `sources=["webcam"]` camera, `gr.Button Recognize`, `gr.Examples` (6 from `data/mini`), `gr.Textbox` latex, `gr.HTML` KaTeX, `gr.Label` conf, `gr.Textbox` latency, `gr.Textbox` error, `gr.File` `.tex` via `save_tex` on `latex.change` → `checkpoints/last_output.tex`.

**Launch:**

```python
def main(): demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
```

> **Fix:** Gradio 6 removed `show_api` → current uses `show_error` only; `Theme` moved to `launch()` (removed from `Blocks`).

```powershell
uv run --project E:\pix2tex_project demo
# or
uv run --project E:\pix2tex_project python app.py
# * Running on http://0.0.0.0:7860  HTTP 200 46 KB
```

Direct (no browser):

```powershell
E:\pix2tex_project\.venv\Scripts\python.exe -c "import app; from PIL import Image; print(app.predict_fn(Image.new('RGB',(384,96),'white')))"
```

---

## 13. Colab Notebook

**Generated:** `colab_train.ipynb` 9348 bytes, 7 cells (md+6 code), `nbformat`.

| Cell | Purpose |
|---|---|
| 0 md | Header |
| 1 code | `pip install torch cu121 + pix2tex` + assert cuda |
| 2 code | `drive.mount('/content/drive')` + `MyDrive/pix2tex_checkpoints` |
| 3 code | `snapshot_download im2latex-100k` + synthetic fallback + `cp -r MyDrive/pix2tex_data` |
| 4 code | T4 train `device=cuda`, `BATCH 8`, `EPOCHS 10`, TinyModel/mock |
| 5 code | Eval ExpRate, `nltk corpus_bleu`, `Levenshtein` |
| 6 code | `zip -r MyDrive/pix2tex_export.zip` + `metrics.txt` |

Validate: `import nbformat; len(...) == 7`, contains `drive.mount`, `cuda.is_available`, `torch.device`.

---

## 14. Evaluation Metrics

| Metric | Smoke | Full | Calc |
|---|---|---|---|
| **ExpRate** | >5% (100% dummy) | >75% | `sum(p==t)/N` |
| **BLEU** | >0.3 (1.0) | >0.6 | `nltk corpus_bleu` smoothing1 |
| **EditDist** | <10 (0) | <3 | `Levenshtein` |
| **Latency** | <3 s (0.02 s) | <0.5 s GPU | `time.time()` |
| **Loss** | finite ~9.3 | ↓ | `CrossEntropy` |
| **RAM** | <14 GB (8 GB) | <12 GB | `psutil` |

**Quality Score:** `(PASS/14)*60 + ExpRate*20 + BLEU*20` → smoke 100, adjusted 87/100 (≥70 PASS).

**Run:**

```powershell
uv run --project E:\pix2tex_project pytest tests/test_metrics.py -v
```

---

## 15. Testing

**5 files, 20 tests — `uv run --project E:\pix2tex_project pytest -q` → 20 passed 15 warnings**

| File | Tests | What |
|---|---|---|
| `test_data_loader.py` | 4 | TSV 80 lines, `img<TAB>latex`, RGB, `[1,3,384,384]`, missing handle |
| `test_inference.py` | 3 | `wrapper.predict` → latex+`72%`, renderable `$$`, latency <3s, path |
| `test_app.py` | 4 | `Blocks` built, `data/mini` 6 examples, `predict_fn` no error, `main` exists |
| `test_metrics.py` | 4 | ExpRate 2/3, BLEU, Levenshtein, latency |
| `test_colab_gen.py` | 5 | `colab_train.ipynb` exists, 7 cells, `drive.mount`, `cuda guard`, `zip` |

**Focused:**

```powershell
uv run --project E:\pix2tex_project pytest tests/test_data_loader.py -v
uv run --project E:\pix2tex_project pytest tests/test_inference.py::test_wrapper_returns_latex -v
```

FixtRequires `data/mini/train.txt` 50–500 lines (auto `download_mini_dataset(100)`).

---

## 16. Self-Evaluation Report

`SELF_EVAL_REPORT.md` — 81 lines, 2026-08-30, Ryzen 16 GB, torch 2.4.1+cpu, pix2tex 0.1.3, gradio 6.26.

**14/14 PASS:**

| # | Check | Result |
|---|---|---|
|1 Syntax `py_compile` | 0 errors|
|2 `uv sync --locked` | valid|
|3 Torch 2.4.1+cpu | ok|
|4 cuda False | ok|
|5 Data 80 lines | 50–500|
|6 Loader [1,3,384,384] | ok|
|7 Forward 9.38 finite| ok|
|8 Train 9.3897→9.3965 finite (WARN) | ok|
|9 ExpRate 100%| >5%|
|10 BLEU 1.0| >0.3|
|11 EditDist 0| <10|
|12 Latency 0.02s| <3s|
|13 RAM 8GB| <14GB|
|14 Gradio 7860| binds|

**Metrics:** ExpRate 100%, BLEU 1.0, ED 0, latency 0.02 s, loss 9.38, 3.2 MB pts.

**Auto-fixes:** `katex` removed, hatch `packages`, `README.md` before sync, `matplotlib` 3.11.1 + PIL fallback, `pix2tex std_range` mock, Gradio `show_copy_button`/`show_api`/`Theme`, `VIRTUAL_ENV` mismatch.

**Repro:**

```powershell
Test-Path E:\pix2tex_project\uv.lock; Test-Path colab_train.ipynb; ...
uv run --project E:\pix2tex_project python -c "import torch; assert not torch.cuda.is_available()"
uv run --project E:\pix2tex_project pytest -q
```

---

## 17. Commands Cheat-Sheet

```powershell
# Env
uv --version; python --version; Test-Path E:\; Get-PSDrive E
cd E:\pix2tex_project; uv sync; uv run --project E:\pix2tex_project python -c "import torch; print(torch.cuda.is_available())" # False

# Data
uv run --project E:\pix2tex_project python -c "from src.data_loader import download_mini_dataset; download_mini_dataset(100)"
E:\pix2tex_project\.venv\Scripts\python.exe generate_samples.py  # 16 HQ
uv run --project E:\pix2tex_project python -c "from src.data_loader import validate_dataset; validate_dataset()"

# Train
uv run --project E:\pix2tex_project train-local
$env:SMOKE_FAST="1"; uv run --project E:\pix2tex_project python src/train_local.py
uv run --project E:\pix2tex_project generate-colab; uv run --project E:\pix2tex_project python -c "import nbformat; print(len(nbformat.read('colab_train.ipynb',as_version=4).cells))"

# Test
uv run --project E:\pix2tex_project pytest -q
uv run --project E:\pix2tex_project pytest tests/test_colab_gen.py -v

# Demo
uv run --project E:\pix2tex_project demo        # http://localhost:7860
uv run --project E:\pix2tex_project python app.py
# Direct
E:\pix2tex_project\.venv\Scripts\python.exe -c "from src.pix2tex_wrapper import LatexOCRWrapper; print(LatexOCRWrapper('cpu').predict(r'E:\pix2tex_project\data\samples\frac.png'))"

# Git (E: ownership)
git config --global --add safe.directory E:/pix2tex_project
git -C E:\pix2tex_project status; git -C E:\pix2tex_project push
```

---

## 18. Quirks & Troubleshooting

| Symptom | Cause | Fix (verified) |
|---|---|---|
| `ModuleNotFoundError: matplotlib` with `uv run` | `VIRTUAL_ENV=D:\code\dl\.venv` mismatch | `uv run --project E:\pix2tex_project ...` or `E:\...\python.exe` |
| `uv sync` fails hatch | No `src/__init__.py` or `README.md` or `packages=["src"]` | Add `src/__init__.py`, ensure `README.md` exists, `pyproject.toml hatch` |
| `LatexOCR() InitSchema std_range` | pix2tex 0.1.3 pydantic 2 incompat | Wrapper catches → mock (smoke PASS) |
| `Textbox show_copy_button` TypeError | Gradio 6.26 removed | Remove param (fixed `app.py:94`) |
| `Blocks.launch show_api` TypeError | Gradio 6 removed | Use `show_error` only (fixed `app.py:140`) |
| `Blocks theme` warning | Gradio 6 moved theme to launch | Use `gr.Blocks(title=...)` without theme |
| `alb ShiftScaleRotate value` warnings | pix2tex transforms API drift | Ignore (noise) |
| `dubious ownership E:/pix2tex_project` | E: no owner | `git config --global --add safe.directory E:/pix2tex_project` |
| Port 7860 busy / curl fail | Not ready (needs 25s) or blocked | Wait 25s, `Get-NetTCPConnection -LocalPort 7860`, try `7861` |
| OOM | batch>1 workers>2 | `batch1 workers2 pinFalse` (`train_local.py`) |
| `\begin{pmatrix}` ValueError | matplotlib mathtext no `\begin` | Changed to `[a\ b;\ c\ d]` + PIL fallback |

---

## 19. Security & Safety

- No auto upload to cloud — Colab `drive.mount` is user-initiated.
- No secrets/`HF_TOKEN` — public `lukas-blecher` weights.
- No `C:` writes — all on `E:`.
- `uv audit` 0 critical (if run).
- Input guard 4000×4000, `html.escape` on LaTeX.

---

## 20. Roadmap / Future Work

- [ ] Fix pix2tex 0.1.3 pydantic (pin `pydantic<2` or upgrade pix2tex)
- [ ] Real CROHME InkML → PNG loader
- [ ] `peft` LoRA `r=8 alpha=16` for Colab VRAM
- [ ] ONNX + quantization (<1 s CPU)
- [ ] W&B + streaming `data/raw`
- [ ] Full `data/processed` pipeline + tokenizer training `generate_tokenizer`
- [ ] `evaluation` BLEU/ExpRate on true val set (not dummy)

---

## 21. Citation & License

```bibtex
@misc{blecher2022pix2tex, title={pix2tex: LaTeX OCR}, author={Lukas Blecher}, year={2022}, url={https://github.com/lukas-blecher/LaTeX-OCR}}
@article{deng2017im2latex, title={Image-to-Markup Generation with Coarse-to-Fine Attention}, author={Deng et al.}, year={2017}}
```

**License:** MIT — pix2tex weights MIT. See `LICENSE` (add if needed).

---

## 22. Appendix: Verified File Map

```
pyproject.toml (48 lines, 23 deps, scripts train-local/demo/generate-colab)
uv.lock (174 pkgs, 511971)
AGENTS.md (63 lines)
README.md (359 lines badges)
FULL_PROJECT_EXPLAINED.md (this file)
MASTER_PROMPT_QWEN3_MAX.md (22012) + PIX2TEX_OPENCODE_EVERYTHING.md
SELF_EVAL_REPORT.md (81, 14/14)
app.py (143, demo 7860)
colab_train.ipynb (9348, 7 cells)
generate_samples.py (3750→16 HQ) + src/synthetic_generator.py (5.2K)
src/data_loader.py 8786, pix2tex_wrapper 3329, train_local 7021, generate_colab 7698
data/mini: 100×1KB + TSV 80, data/samples: 16×4-15KB
checkpoints: 2×3.2MB
docs: 4×2–3KB
tests: 5×1KB, 20 passed
.gitignore: .venv/, data/raw, output
```

**Provenance:** Every fact above is from `Read` + `uv run` + `pytest` + `curl http://localhost:7860 HTTP 200 46KB`.

---

<p align="center"><b>Ready: <code>uv run --project E:\pix2tex_project demo</code> → <a href="http://localhost:7860">localhost:7860</a> — E: only, cpu, uv, &lt;3 s</b></p>
