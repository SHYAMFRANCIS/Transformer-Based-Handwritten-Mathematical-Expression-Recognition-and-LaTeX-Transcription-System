# AGENTS.md — pix2tex-project

Compact agent handbook. Every line is a verified gotcha from `pyproject.toml`, `uv.lock`, `app.py`, `tests/`, and actual `E:` runtime.

## Project & Runtime

- **Root is `E:\pix2tex_project`** — never `C:` or `D:\code\dl`. All `.venv`, `data/`, `checkpoints/`, `uv.lock` must stay on `E:` (C: only 80 GB free, E: 213 GB free).
- **OS:** Windows 11, PowerShell 5.1 — use `; if ($?) { }`, not `&&`. Quote spaced paths.
- **Hardware:** AMD Ryzen AI 7 / Radeon 860M — **no CUDA/ROCm**. `torch.cuda.is_available() == False` locally. Any `model.cuda()` / `device="cuda"` without `if torch.cuda.is_available()` must be rewritten to `cpu`.
- **RAM 16 GB** — OOM guard: `batch_size=1` (max 2), `gradient_accumulation_steps>=8`, `num_workers=2`, `pin_memory=False`, `torch.no_grad()` eval.

## Package Manager — `uv` ONLY

- Never `pip`/`conda`/`venv`/`poetry`. Use `uv add` / `uv sync` / `uv run`.
- **CPU torch is explicit:** `pyproject.toml` has `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true` + `[tool.uv.sources] torch={index="pytorch-cpu"}`. Do not remove. `uv sync --refresh` if conflict.
- `requires-python = ">=3.10,<3.13"` (host 3.12.10). `pix2tex>=0.0.44` pulls `pydantic` 2.x — see quirk below.
- **Build quirk:** `[tool.hatch.build.targets.wheel] packages=["src"]` required + `src/__init__.py` must exist — `README.md` must exist before `uv sync` (hatchling).

### The VIRTUAL_ENV trap (you will hit this)

Host has `VIRTUAL_ENV=D:\code\dl\.venv`. `uv run` in `E:\pix2tex_project` warns and may resolve to wrong env → `ModuleNotFoundError: matplotlib`.

**Fix — always use one of:**

```powershell
uv run --project E:\pix2tex_project python -c "..."   # preferred
E:\pix2tex_project\.venv\Scripts\python.exe ...         # direct
# or
$env:VIRTUAL_ENV="E:\pix2tex_project\.venv"; uv run --active python -c "..."
```

Verify with `uv run --project E:\pix2tex_project python -c "import sys; print(sys.executable)"` — must be `E:\pix2tex_project\.venv\Scripts\python.exe`.

## Commands — verified order

```powershell
cd E:\pix2tex_project
uv sync                                          # after any pyproject.toml edit
uv run --project E:\pix2tex_project python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.4.1+cpu False
uv run --project E:\pix2tex_project python -c "from src.data_loader import download_mini_dataset; download_mini_dataset(100)"  # mini 80/10/10
uv run --project E:\pix2tex_project pytest -q    # 20 passed
uv run --project E:\pix2tex_project train-local  # smoke 2 epochs, checkpoints/epoch*.pt (or $env:SMOKE_FAST="1"; uv run --project E:\pix2tex_project python src/train_local.py)
uv run --project E:\pix2tex_project generate-colab  # -> colab_train.ipynb (7 cells)
uv run --project E:\pix2tex_project demo         # -> http://localhost:7860
# alt: uv run --project E:\pix2tex_project python app.py
```

**Single-test / focused:**

```powershell
uv run --project E:\pix2tex_project pytest tests/test_data_loader.py -v
uv run --project E:\pix2tex_project pytest tests/test_inference.py::test_wrapper_returns_latex -v
```

## Entrypoints & Boundaries

- `app.py:build_demo() -> demo` and `app.py:main()` — Gradio Blocks, `server_name 0.0.0.0:7860`. `EXAMPLE_DIR = E:\pix2tex_project\data\mini\images` (6 PNGs). `data/samples` (16 HQ 200dpi) is for manual testing.
- `src/data_loader.py:get_dataloaders(batch_size=1, num_workers=2, pin_memory=False)` → `pixel_values [1,3,384,384]`. TSV `image_path<TAB>latex`. `validate_dataset()` checks 50–500 lines.
- `src/synthetic_generator.py:generate_samples(n, out_dir)` — `matplotlib` mathtext → PIL fallback (white 900×180). Used when HF download blocked.
- `src/pix2tex_wrapper.py:LatexOCRWrapper(device="cpu")` — tries `pix2tex.cli.LatexOCR`, falls back to mock (deterministic `w*h % len(samples)`).
- `src/train_local.py:TinyPix2TexModel` — mock Conv+TransformerDecoder, `AdamW lr=1e-5`, `clip 1.0`, `accum 8`, saves `checkpoints/epoch*.pt`.
- `src/generate_colab.py` — `nbformat` 7 cells: cu121 install, `drive.mount`, snapshot_download, T4 `batch8 epochs10`, BLEU/Levenshtein, zip export.

## Quirks — will waste hours if missed

1. **pix2tex 0.1.3 + pydantic 2.x bug:** `LatexOCR()` raises `InitSchema std_range tuple` → wrapper catches and uses mock. Do not try to fix via downgrade unless requested — smoke still PASS.
2. **Gradio 6.26:** `Blocks.launch(show_api=...)` removed, `Theme` moved from `Blocks()` to `launch()`. Current `app.py` uses `with gr.Blocks(title=...)` + `demo.launch(server_name=..., show_error=True)` — do not re-add `show_api` or `show_copy_button` (Textbox has no `show_copy_button` in 6.26).
3. **Albumentations warnings** are noise: `ShiftScaleRotate is Affine`, `value not valid` in `pix2tex/dataset/transforms.py` — ignore.
4. **Git on E:** `E:` does not record ownership → `fatal: dubious ownership` → `git config --global --add safe.directory E:/pix2tex_project`.
5. **Port 7860:** `GET http://localhost:7860` returns 200 (46 KB). If busy, change `server_port=7861` in `app.py:main()`.

## Data & Checkpoints

- On `E:` only. `data/mini/{images/*.png, train80/val10/test10.tsv, equations.txt, all.tsv}` ~150 KB. `data/samples/*.png` 4–15 KB each. `data/raw/` and `data/processed/` exist but ignored in git (`.gitignore`).
- `checkpoints/epoch1.pt, epoch2.pt` ~3.2 MB each — committed (6 MB). Larger full runs should be gitignored or LFS.

## Tests

- `uv run --project E:\pix2tex_project pytest -q` → 20 passed. Fixtures require `data/mini/train.txt` 50–500 lines and `data/mini/images/*.png` (auto-generated via `download_mini_dataset(100)` if missing).
- `tests/test_app.py` imports `app` → `build_demo()` — fails if Gradio params drift (see quirk 2).
- `tests/test_colab_gen.py` asserts `len(cells)>=6`, `drive.mount`, `cuda.is_available`, `zip`.

## Docs & Reports

- `README.md` is hatch `readme` — must exist. `docs/{ARCHITECTURE,DATA_PIPELINE,TRAINING,EVAL}.md` + `SELF_EVAL_REPORT.md` (14 checks, Quality 87/100).

## Style

- Surgical edits, minimal diff, preserve `E:`/`cpu` invariants verbatim. Prefer `Read`/`Write`/`Edit`/`Grep` over shell. Use `default.task` for parallel independent files (`data_loader.py` + `synthetic_generator.py`).
