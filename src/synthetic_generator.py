"""
Synthetic KaTeX -> PNG generator for Pix2Tex mini-dataset.
Fallback chain: matplotlib mathtext -> PIL text.
All outputs saved to E:\\pix2tex_project\\data\\mini (or specified dir).
"""
import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont

LATEX_SAMPLES = [
    r"x^2 + y^2 = z^2",
    r"\frac{a}{b}",
    r"\sum_{i=1}^{n} x_i",
    r"\int_0^1 x \, dx",
    r"\sqrt{2\pi r}",
    r"\frac{x^2 + 3x - 5}{2a}",
    r"e^{i\pi} + 1 = 0",
    r"\alpha + \beta = \gamma",
    r"\lim_{x \to \infty} \frac{1}{x} = 0",
    r"\prod_{k=1}^{n} k = n!",
    r"\vec{F} = m\vec{a}",
    r"\mathcal{L} = \frac{1}{2}mv^2",
    r"x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
    r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    r"\sin^2 \theta + \cos^2 \theta = 1",
    r"\log_{2} 8 = 3",
    r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
    r"\forall x \in \mathbb{R}",
    r"\exists y : y > x",
    r"\partial_t u + \nabla \cdot (u \mathbf{v}) = 0",
]

def _render_matplotlib(latex: str, out_path: str, dpi: int = 150):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 1.2), dpi=dpi)
        fig.patch.set_facecolor("white")
        # Use mathtext; wrap in $...$
        txt = f"${latex}$"
        plt.text(0.5, 0.5, txt, fontsize=14, ha="center", va="center", color="black")
        plt.axis("off")
        plt.tight_layout(pad_inches=0.1)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"matplotlib render failed for {latex}: {e}")
        return False

def _render_pil_fallback(latex: str, out_path: str, size=(384, 96)):
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    try:
        # try default font, else load
        font = ImageFont.load_default()
    except:
        font = None
    # Center text
    text = latex  # raw latex as fallback visible
    # simple text wrapping
    draw.text((10, size[1]//2 - 10), text, fill="black", font=font)
    img.save(out_path, "PNG")
    return True

def render_latex_to_png(latex: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = _render_matplotlib(latex, out_path)
    if not ok or not os.path.exists(out_path):
        _render_pil_fallback(latex, out_path)
    # Ensure output exists and is not zero bytes
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _render_pil_fallback(latex, out_path)
    # Normalize to 384 width via resize if needed (kept by loader)
    return out_path

def generate_samples(n: int = 100, out_dir: str = None, seed: int = 42):
    """Generate n synthetic PNGs + TSV + pix2tex-compatible files on E: drive.
    Returns (images_dir, tsv_path, equations_path)
    """
    if out_dir is None:
        out_dir = r"E:\pix2tex_project\data\mini"
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    random.seed(seed)

    tsv_paths = {
        "all": os.path.join(out_dir, "all.tsv"),
        "train": os.path.join(out_dir, "train.txt"),
        "val": os.path.join(out_dir, "val.txt"),
        "test": os.path.join(out_dir, "test.txt"),
    }
    # pix2tex compatible: equations.txt + images/*.png indexed
    equations_path = os.path.join(out_dir, "equations.txt")

    pairs = []
    equations = []
    for i in range(n):
        latex = random.choice(LATEX_SAMPLES)
        # Add slight variation index to avoid duplicate filenames issues
        # keep latex exact for determinism of tests
        fname = f"{i}.png"
        fpath = os.path.join(images_dir, fname)
        render_latex_to_png(latex, fpath)
        pairs.append((fpath, latex))
        equations.append(latex)

    # Write equations.txt indexed by filename integer
    with open(equations_path, "w", encoding="utf-8") as f:
        for eq in equations:
            f.write(eq + "\n")

    # Write all.tsv (image_path<TAB>latex)
    with open(tsv_paths["all"], "w", encoding="utf-8") as f:
        for p, l in pairs:
            f.write(f"{p}\t{l}\n")

    # 80/10/10 split
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(tsv_paths[name], "w", encoding="utf-8") as f:
            for i in idx:
                p, l = pairs[i]
                f.write(f"{p}\t{l}\n")

    print(f"Generated {n} samples -> {images_dir} | TSV: {tsv_paths['train']} ({len(train_idx)}), val {len(val_idx)}, test {len(test_idx)}")
    return images_dir, tsv_paths["train"], equations_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default=r"E:\pix2tex_project\data\mini")
    args = parser.parse_args()
    generate_samples(n=args.n, out_dir=args.out)

if __name__ == "__main__":
    main()
