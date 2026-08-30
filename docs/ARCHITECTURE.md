# Architecture — Pix2Tex Handwritten MER (CPU + Colab Hybrid)

## High-Level Flow

```
[Image Upload / Camera] 
        |
        v
   PIL RGB  (384x384)  --> albumentations Normalize (0.7931, 0.1738)
        |
        v
  ViT Encoder (timm/torch, tiny mock for smoke) 
  3 -> 16 -> 32 channels, AdaptiveAvgPool 12x12  -> 144 tokens x 32 dim
        |
        v
 Transformer Decoder (2 layers, 4 heads, d=32)  <-- Label Embeddings (bos/eos)
        |
        v
   LM Head (vocab 8000) -> CrossEntropy (ignore PAD=0)
        |
        v
  Latex String  -> KaTeX HTML (CDN) -> Gradio Blocks (port 7860)
```

Real `lukas-blecher/LaTeX-OCR` uses same ViT + Transformer decoder + BPE tokenizer (vocab 8000, `[PAD]=0 [BOS]=1 [EOS]=2`), but smoke mock keeps identical signature `model(pixel_values, labels).loss`.

## Component Map

- `src/pix2tex_wrapper.py:LatexOCRWrapper` — CPU-enforced `torch.device("cpu")`, tries `pix2tex.cli.LatexOCR`, falls back to mock heuristic. `predict(PIL) -> (latex, conf)`.
- `src/data_loader.py:Im2LatexDataset` — TSV `image_path<TAB>latex`, RGB, 384x384, returns `{pixel_values:[3,384,384], labels:[S]}`. `get_dataloaders(batch=1, workers=2, pin_memory=False)`.
- `src/synthetic_generator.py` — KaTeX→PNG via matplotlib (fallback PIL). Writes `images/*.png`, `equations.txt`, `train.txt/val.txt/test.txt` 80/10/10.
- `src/train_local.py:TinyPix2TexModel` — CPU smoke, `AdamW lr=1e-5`, `accum=8`, `clip=1.0`, 2 epochs, checkpoints `E:\pix2tex_project\checkpoints\epoch*.pt`.
- `src/generate_colab.py` — nbformat generates `colab_train.ipynb` 7 cells (GPU `cu121`, Drive mount, IM2LaTeX-100K, T4 `batch8 epochs10`, ExpRate/BLEU, zip export).
- `app.py:build_demo()` — `gr.Blocks` with Upload+Camera, Textbox LaTeX, HTML KaTeX, Label Confidence, Number Latency, Error, Examples(6), File download, `server_name 0.0.0.0:7860`.

## E: Drive Layout (Single Source of Truth)

```
E:\pix2tex_project\
 ├── pyproject.toml (torch==2.4.1+cpu, torchvision==0.19.1+cpu, pix2tex>=0.0.44, gradio>=4.44.0)
 ├── uv.lock (strict)
 ├── data/{raw,processed,mini/{images/*.png, train.txt(80), val.txt(10), test.txt(10), equations.txt, all.tsv}}
 ├── src/{data_loader.py:42, synthetic_generator.py:38, pix2tex_wrapper.py:28, train_local.py:65, generate_colab.py:58}
 ├── checkpoints/{epoch1.pt(3.2MB), epoch2.pt, last_output.tex}
 ├── colab_train.ipynb (7 cells, drive.mount, cuda guard)
 ├── app.py (127 lines, Gradio)
 ├── tests/{5 files, 20 tests}
 └── docs/{ARCHITECTURE, DATA_PIPELINE, TRAINING, EVAL}.md
```

## Invariants

- `torch.cuda.is_available() == False` locally; all `to(DEVICE)` where `DEVICE=cpu`.
- No `C:` writes; `E:` only.
- `uv` only; `[[tool.uv.index]] pytorch-cpu` explicit.

## API

- `GET /` → Gradio
- `POST /api/predict` (Gradio auto) + documented `POST /api/latex` shim.
