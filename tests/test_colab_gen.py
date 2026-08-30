import os
import nbformat

NB_PATH = r"E:\pix2tex_project\colab_train.ipynb"

def test_notebook_exists():
    assert os.path.exists(NB_PATH), f"Missing {NB_PATH}"

def test_notebook_cells():
    nb = nbformat.read(NB_PATH, as_version=4)
    assert len(nb.cells) >= 6, f"Need 6+ cells, got {len(nb.cells)}"
    # at least 1 markdown + 5 code
    code_cells = [c for c in nb.cells if c.cell_type=="code"]
    assert len(code_cells) >= 5

def test_has_drive_mount():
    nb = nbformat.read(NB_PATH, as_version=4)
    src = "\n".join(c.source for c in nb.cells if c.cell_type=="code")
    assert "drive.mount" in src, "Missing drive.mount"
    assert "MyDrive" in src

def test_has_cuda_guard():
    nb = nbformat.read(NB_PATH, as_version=4)
    src = "\n".join(c.source for c in nb.cells if c.cell_type=="code")
    assert "cuda.is_available" in src
    assert "torch.device" in src
    # check both cpu and cuda strings
    assert "cuda" in src and "cpu" in src

def test_has_training_and_export():
    nb = nbformat.read(NB_PATH, as_version=4)
    src = "\n".join(c.source for c in nb.cells if c.cell_type=="code")
    assert "pix2tex" in src.lower() or "TinyModel" in src
    assert "zip" in src.lower() or "export" in src.lower()
    assert "checkpoints" in src.lower()
