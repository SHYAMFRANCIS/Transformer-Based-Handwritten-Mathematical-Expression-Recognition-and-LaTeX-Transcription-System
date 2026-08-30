"""Generate curated sample images for manual / Gradio testing - uses matplotlib mathtext for high-quality rendering."""
import os
import pathlib
SAMPLES = [
    ("frac", r"\frac{a}{b}"),
    ("quadratic", r"x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2a}"),
    ("pythagoras", r"x^2 + y^2 = z^2"),
    ("summation", r"\sum_{i=1}^{n} x_i = S"),
    ("integral", r"\int_0^1 x^2 \, dx = \frac{1}{3}"),
    ("euler", r"e^{i\pi} + 1 = 0"),
    ("alpha_beta", r"\alpha + \beta = \gamma"),
    ("limit", r"\lim_{x \to \infty} \frac{1}{x} = 0"),
    ("sqrt", r"\sqrt{2\pi r}"),
    ("matrix", r"[a\ b;\ c\ d]"),
    ("binomial", r"\binom{n}{k} = \frac{n!}{k!(n-k)!}"),
    ("vector", r"\vec{F} = m \vec{a}"),
    ("maxwell", r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}"),
    ("prob", r"P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}"),
    ("continued_frac", r"\frac{1}{1 + \frac{1}{2}}"),
    ("derivative", r"\frac{d}{dx} x^n = n x^{n-1}"),
]

OUT_DIR = r"E:\pix2tex_project\data\samples"

def render(latex, path, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 1.8), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    txt = f"${latex}$"
    try:
        ax.text(0.5, 0.5, txt, fontsize=22, ha="center", va="center", color="black")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white", dpi=dpi)
        plt.close(fig)
        return path
    except Exception as e:
        plt.close(fig)
        # Fallback to PIL
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (900, 180), "white")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        d.text((20, 70), latex, fill="black", font=font)
        img.save(path, "PNG", dpi=(dpi, dpi))
        print(f"  fallback PIL for {latex[:40]}: {e}")
        return path

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mapping = []
    for name, latex in SAMPLES:
        fname = f"{name}.png"
        fpath = os.path.join(OUT_DIR, fname)
        render(latex, fpath)
        # verify exists
        size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        print(f"{fname:20} -> {latex:45} [{size} bytes] {'OK' if size>2000 else 'SMALL'}")
        mapping.append((fname, latex))
    # Write TSV and markdown manifest
    tsv = os.path.join(OUT_DIR, "samples.tsv")
    md = os.path.join(OUT_DIR, "README.md")
    with open(tsv, "w", encoding="utf-8") as f:
        for fname, latex in mapping:
            f.write(f"{os.path.join(OUT_DIR, fname)}\t{latex}\n")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Sample Images for Testing\n\n")
        f.write("Curated 16 expressions rendered via matplotlib mathtext (dpi 200).\n\n")
        f.write("| File | LaTeX | Preview |\n|---|---|---|\n")
        for fname, latex in mapping:
            f.write(f"| `{fname}` | `{latex}` | ![]({fname}) |\n")
        f.write("\n## Usage\n\n")
        f.write("```powershell\n")
        f.write("uv run python app.py  # then drag any PNG into Gradio\n")
        f.write("uv run python -c \"from src.pix2tex_wrapper import LatexOCRWrapper; w=LatexOCRWrapper('cpu'); print(w.predict(r'E:\\pix2tex_project\\data\\samples\\frac.png'))\"\n")
        f.write("```\n")
        f.write(f"\nTotal: {len(mapping)} samples in `{OUT_DIR}`\n")
    print(f"\nWrote {tsv} + {md}")
    # Also copy to examples dir for Gradio if needed
    import shutil, pathlib
    print(f"\nAll samples: {OUT_DIR}")
    for p in pathlib.Path(OUT_DIR).glob("*.png"):
        print(p.name, p.stat().st_size)

if __name__ == "__main__":
    main()
