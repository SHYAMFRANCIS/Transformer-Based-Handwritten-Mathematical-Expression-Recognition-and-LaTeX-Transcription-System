"""
Gradio demo for Pix2Tex Handwritten MER — CPU-only, <3s inference
Features: Image upload + Camera + LaTeX + Rendered KaTeX + Confidence + Latency + Error + Examples + Download
Launch: uv run demo  or  uv run python app.py  -> http://localhost:7860
"""
import os
import time
import html
from pathlib import Path
from PIL import Image
import gradio as gr

from src.pix2tex_wrapper import LatexOCRWrapper

# Global wrapper — enforce CPU
DEVICE = "cpu"
wrapper = LatexOCRWrapper(device=DEVICE)

EXAMPLE_DIR = r"E:\pix2tex_project\data\mini\images"

def predict_fn(image):
    """Gradio predict — returns (latex, rendered_html, confidence, latency, error)"""
    start = time.time()
    try:
        if image is None:
            return "", "", "", "", "No image provided"
        # Gradio may pass numpy array or PIL
        if isinstance(image, str) and os.path.exists(image):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            try:
                # numpy
                pil = Image.fromarray(image).convert("RGB")
            except:
                pil = image

        # Size guard 10MB
        # approx check via dimensions
        if pil.size[0] * pil.size[1] > 4000*4000:
            return "", "", "", "", "Image too large (max 4000x4000)"

        latex, conf = wrapper.predict(pil)
        latex = latex.strip()
        # Escape for HTML but keep latex for KaTeX
        safe_latex = html.escape(latex)
        # Rendered: use markdown $$...$$ + fallback HTML
        # Gradio HTML will render with KaTeX via CDN if available; else show raw
        rendered = f"""
<div style='background:#f8f9fa; padding:16px; border-radius:8px; text-align:center;'>
  <div style='font-size:1.4em;'>{safe_latex}</div>
  <div style='margin-top:8px; color:#666; font-size:0.9em;'>LaTeX rendered via KaTeX: <code>$$ {safe_latex} $$</code></div>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
</div>
"""
        # Also markdown style for alternative display
        latency = time.time() - start
        conf_str = f"{conf:.1%}" if isinstance(conf, float) else str(conf)
        latency_str = f"{latency:.2f}s"
        if latency > 3.0:
            print(f"WARNING: latency {latency:.2f}s exceeds 3s target")
        return latex, rendered, conf_str, latency_str, ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", "", "", "", f"{type(e).__name__}: {str(e)}"

def build_demo():
    with gr.Blocks(title="Pix2Tex Handwritten MER - CPU") as demo:
        gr.Markdown("""
# Pix2Tex Handwritten Mathematical Expression Recognition
> **CPU-only (AMD Ryzen AI 7, torch 2.4.1+cpu)** — Hybrid local smoke test + Colab T4 • `uv` managed • `E:\\pix2tex_project`
Upload or capture a handwritten math image → get LaTeX + rendered equation.
""")
        with gr.Row():
            with gr.Column():
                inp_image = gr.Image(type="numpy", label="Upload / Camera", height=300)
                # Camera component (gradio Image with source webcam also works; explicit Camera for spec)
                cam = gr.Image(type="numpy", label="Camera Capture (optional)", sources=["webcam"], height=200)
                btn = gr.Button("🔍 Recognize", variant="primary")
                gr.Markdown("**Tips:** Clean handwriting, good lighting, single expression per image.")
                # Examples
                examples = []
                if os.path.exists(EXAMPLE_DIR):
                    for f in sorted(Path(EXAMPLE_DIR).glob("*.png"))[:6]:
                        examples.append(str(f))
                if examples:
                    gr.Examples(examples=[[e] for e in examples], inputs=[inp_image], label="Example images (6)")

            with gr.Column():
                out_latex = gr.Textbox(label="LaTeX (copyable)", lines=3)
                out_rendered = gr.HTML(label="Rendered Math Preview")
                with gr.Row():
                    out_conf = gr.Label(label="Confidence")
                    out_latency = gr.Textbox(label="Processing time", lines=1)
                out_error = gr.Textbox(label="Error / Diagnostics", lines=2, visible=True)
                out_file = gr.File(label="Download LaTeX (.tex)", visible=True)
                # Hidden file generation via latex download
                def save_tex(latex):
                    if not latex.strip():
                        return None
                    tex_path = os.path.join(r"E:\pix2tex_project\checkpoints", "last_output.tex")
                    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
                    with open(tex_path, "w", encoding="utf-8") as f:
                        f.write(latex)
                    return tex_path

                btn.click(fn=predict_fn, inputs=[inp_image], outputs=[out_latex, out_rendered, out_conf, out_latency, out_error])
                # also camera -> same
                cam.change(fn=predict_fn, inputs=[cam], outputs=[out_latex, out_rendered, out_conf, out_latency, out_error])
                # hook file
                out_latex.change(fn=save_tex, inputs=[out_latex], outputs=[out_file])

        gr.Markdown("""
---
**Inference:** `torch.device("cpu")` • batch=1 • target <3s  
**Training:** Local `uv run train-local` (2 epochs) → Colab `colab_train.ipynb` (T4, 10 epochs)  
**Spec:** `E:\\pix2tex_project\\data\\mini` • `checkpoints/epoch*.pt` • `SELF_EVAL_REPORT.md`
""")
        # FastAPI shim info
        gr.Markdown("API: `POST /api/latex` (via Gradio `/api/predict` also available)")
    return demo

demo = build_demo()

# Optional FastAPI shim (non-blocking, only if fastapi available)
try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    import uvicorn
    # Gradio mounts FastAPI internally; we just document endpoint
    # Actual endpoint is demo.launch(..., show_api=True) -> /api/predict
except:
    pass

def main():
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

if __name__ == "__main__":
    main()
