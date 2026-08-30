# Pix2Tex Handwritten Mathematical Expression Recognition — CPU Optimized

> **Hardware:** AMD Ryzen AI 7 + Radeon 860M (NO CUDA/ROCm) • 16GB RAM • Windows 11 • E: drive only  
> **Package Manager:** `uv` ONLY (no pip/conda/poetry)  
> **Strategy:** Local CPU smoke test (2 epochs, <30 min) + Colab T4 full training + Gradio demo

## Quickstart (PowerShell 5.1)

```powershell
# 1. Verify environment
uv --version
python --version
Test-Path E:\

# 2. Install (CPU-only PyTorch via explicit index)
cd E:\pix2tex_project
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1 False

# 3. Smoke data + loader test
uv run python -c "from src.data_loader import download_mini_dataset; download_mini_dataset(100)"
uv run python -c "from src.data_loader import get_dataloaders; dl,_,_=get_dataloaders(); print(next(iter(dl))['pixel_values'].shape)"
# Expected: torch.Size([1,3,384,384])

# 4. Local CPU smoke training (2 epochs, batch=1, accum=8)
uv run train-local
# or
uv run python src/train_local.py

# 5. Generate Colab T4 notebook
uv run generate-colab
uv run python -c "import nbformat; print(len(nbformat.read('colab_train.ipynb', as_version=4).cells))"

# 6. Run tests
uv run pytest -q

# 7. Launch Gradio demo (CPU <3s)
uv run demo
# or
uv run python app.py
# Open http://localhost:7860
```

## Colab Full Training

1. Upload `colab_train.ipynb` to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all 6 cells (Drive mount → dataset → train `cuda` → eval ExpRate/BLEU → export `pix2tex_export.zip` to Drive)
4. Download `pix2tex_export.zip` → extract to `E:\pix2tex_project\checkpoints\`

## Troubleshooting

| Issue | Fix |
|---|---|
| `uv sync` fails | `uv sync --refresh` — check `[[tool.uv.index]] pytorch-cpu` in `pyproject.toml` |
| Torch installs CUDA | `uv remove torch torchvision; uv add torch==2.4.1 torchvision==0.19.1 --index pytorch-cpu; uv sync` |
| OOM | `batch_size=1, num_workers=0, accum=16` in `src/train_local.py` |
| Gradio port blocked | Change `server_port=7861` in `app.py` |
| Dataset download fails | `uv run python -c "from src.synthetic_generator import generate_samples; generate_samples(100)"` |
| Invalid `colab_train.ipynb` | `uv run generate-colab` |

## Project Tree

```
E:\pix2tex_project\
 ├── pyproject.toml / uv.lock
 ├── data/{raw,processed,mini}
 ├── src/{data_loader,train_local,generate_colab,synthetic_generator,pix2tex_wrapper}
 ├── checkpoints/
 ├── colab_train.ipynb
 ├── app.py
 └── tests/
```

## Inference Target

CPU inference <3s/image on AMD Ryzen AI 7 via `torch.device("cpu")`.

## License

MIT — Uses `lukas-blecher/LaTeX-OCR` (pix2tex) weights.
