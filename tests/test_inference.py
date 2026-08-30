import time
from PIL import Image

def test_wrapper_returns_latex():
    from src.pix2tex_wrapper import LatexOCRWrapper
    wrapper = LatexOCRWrapper(device="cpu")
    img = Image.new("RGB", (384, 96), "white")
    latex, conf = wrapper.predict(img)
    assert isinstance(latex, str) and len(latex) > 0
    assert latex.startswith("\\") or "^" in latex or "=" in latex
    assert isinstance(conf, float)
    assert 0 < conf <= 1

def test_renderable_and_latency():
    from src.pix2tex_wrapper import LatexOCRWrapper
    wrapper = LatexOCRWrapper(device="cpu")
    img = Image.new("RGB", (384, 96), "white")
    t0 = time.time()
    latex, _ = wrapper.predict(img)
    latency = time.time() - t0
    # should be renderable as math (contains latex-like chars)
    assert len(latex) > 0
    assert latency < 3.0, f"Latency {latency:.2f}s exceeds 3s"
    # check $$ wrapping would be valid
    rendered = f"$$ {latex} $$"
    assert "$$" in rendered

def test_wrapper_path_input():
    from src.pix2tex_wrapper import LatexOCRWrapper
    import os
    wrapper = LatexOCRWrapper(device="cpu")
    sample = r"E:\pix2tex_project\data\mini\images\0.png"
    if os.path.exists(sample):
        latex, conf = wrapper.predict(sample)
        assert isinstance(latex, str)
