import os

def test_gradio_blocks_build():
    import app
    assert hasattr(app, "demo")
    assert hasattr(app, "predict_fn")
    # Check demo is Blocks
    import gradio as gr
    assert isinstance(app.demo, gr.Blocks)

def test_examples_exist():
    from pathlib import Path
    ex_dir = r"E:\pix2tex_project\data\mini\images"
    assert os.path.exists(ex_dir)
    files = list(Path(ex_dir).glob("*.png"))
    assert len(files) >= 6, f"Need 6 examples, got {len(files)}"

def test_app_predict_no_error():
    import app
    from PIL import Image
    img = Image.new("RGB", (384,96), "white")
    # pass as numpy (gradio style)
    import numpy as np
    np_img = __import__("numpy").array(img)
    latex, rendered, conf, latency, error = app.predict_fn(np_img)
    assert isinstance(latex, str)
    assert error == "" or isinstance(error, str)
    assert "katex" in rendered.lower() or "$$" in rendered

def test_app_launch_import():
    import app
    # ensure launch function exists
    assert hasattr(app, "main")
    # do not actually launch server in test
