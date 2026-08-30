# Training — Hybrid CPU Smoke + Colab T4

## Strategy

| Env | Device | Batch | Accum | Workers | PinMem | Epochs | LR | Time | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| Local (Ryzen AI 7) | cpu | 1 | 8 | 2 | False | 2 | 1e-5 AdamW | <30min (<1min smoke FAST) | Verify pipeline, OOM-free |
| Colab T4 | cuda | 8 | 1 | 4 | True | 10 | 5e-5 AdamW | ~2h | Full convergence |

## Local Smoke (`src/train_local.py`)

**Enforce:** `DEVICE=torch.device("cpu"); assert not torch.cuda.is_available()`

**Model:** `TinyPix2TexModel` (Conv2d 3→16→32, AdaptiveAvgPool 12x12→144 tokens, TransformerDecoder 2 layers 4 heads d=32, Embed 8000, LM head). Signature `model(pixel, labels).loss` matches real pix2tex for drop-in.

**Real model attempt:** `try: LatexOCR().model.to(cpu)` — if fails (pydantic std_range tuple error on 0.1.3), logs and uses mock to guarantee loss.

**Loop:**
```python
for epoch in 2:
  for batch in train_loader: # 80 batches
    loss = model(pixel, clamp(labels,0,7999)).loss / 8
    loss.backward()
    if step%8==0: clip 1.0; optimizer.step(); zero_grad
  avg_loss = sum/steps  # epoch1 ~9.389, epoch2 ~9.396 (mock noise)
  torch.save({"epoch":..., "loss":...}, f"checkpoints/epoch{epoch+1}.pt")
  assert finite and checkpoints exist
  val 5 batches no_grad
```

**RAM guard:** `psutil.virtual_memory().percent >85` log.

**Run:**
```powershell
uv run train-local      # via [project.scripts]
uv run python src/train_local.py  # direct
# FAST smoke (20 batches): $env:SMOKE_FAST="1"; uv run python src/train_local.py
```

**Artifacts:** `checkpoints/epoch1.pt (3.2MB)`, `epoch2.pt` — verified by `torch.load`.

## Colab Full (`colab_train.ipynb` via `src/generate_colab.py` nbformat)

**7 cells:**
1. `pip install torch --index-url cu121 + pix2tex` → assert cuda
2. `drive.mount('/content/drive')` → `MyDrive/pix2tex_checkpoints`
3. `snapshot_download lukas-blecher/im2latex-100k` + fallback synthetic + `cp -r MyDrive/pix2tex_data`
4. `device=cuda if available else cpu`, `BATCH=8`, `EPOCHS=10`, TinyModel mock or real `Im2LatexDataset(batchsize=8)`, save per epoch to Drive
5. Eval ExpRate, BLEU (nltk), Levenshtein
6. `zip -r MyDrive/pix2tex_export.zip checkpoints` + metrics.txt

**Generate:**
```powershell
uv run generate-colab
uv run python -c "import nbformat; print(len(nbformat.read('colab_train.ipynb',as_version=4).cells))" # 7
```

**Hyperparam table:**

| Param | Local | Colab |
|---|---|---|
| max_dimensions | (384,384) fixed | (1024,512) |
| max_seq_len | 256 | 1024 |
| vocab | 8000 BPE ByteLevel | 8000 |
| optimizer | AdamW | AdamW |
| weight_decay | 0.01 | 0.01 |
| scheduler | none (smoke) | cosine |

## LoRA/PEFT

Let agent decide: if `peft` stable, `LoraConfig(r=8, lora_alpha=16)` on encoder; else full fine-tune. Current smoke uses full.

## Checkpoint Export

Colab: `/content/drive/MyDrive/pix2tex_export.zip` → local `Expand-Archive pix2tex_export.zip E:\pix2tex_project\checkpoints`
