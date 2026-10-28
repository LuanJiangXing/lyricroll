#!/usr/bin/env python3
"""生成样片用的占位素材（不是 AIGC，只是抽象色块占位）。
产出：assets/demo-a.jpg / demo-b.jpg / demo-c.jpg
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
W, H = 1400, 1350

PALETTES = [
    # top, bottom, blob colors
    ((9, 24, 30), (44, 80, 86), [(232, 176, 96), (56, 118, 124), (150, 200, 190)]),
    ((26, 17, 13), (92, 60, 36), [(236, 168, 92), (138, 88, 58), (250, 216, 164)]),
    ((22, 17, 28), (82, 58, 80), [(206, 140, 150), (118, 88, 138), (240, 202, 192)]),
]
NAMES = ["demo-a.jpg", "demo-b.jpg", "demo-c.jpg"]


def make_image(seed, top, bottom, blobs):
    rng = np.random.default_rng(seed)
    y = (np.linspace(0, 1, H) ** 0.85)[:, None]
    base = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        base[:, :, c] = top[c] + (bottom[c] - top[c]) * y
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i in range(6):
        cx = rng.uniform(-W * 0.2, W * 1.2)
        cy = rng.uniform(-H * 0.1, H * 1.1)
        r = rng.uniform(W * 0.22, W * 0.62)
        col = blobs[i % len(blobs)]
        a = int(rng.uniform(30, 95))
        d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=col + (a,))
    layer = layer.filter(ImageFilter.GaussianBlur(120))
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    rad = np.sqrt(nx * nx + ny * ny)
    vig = np.clip(1.0 - 0.42 * np.clip(rad - 0.45, 0, None) ** 1.5, 0.35, 1.0)[..., None]
    arr = np.asarray(img, np.float32) * vig

    # grain
    arr += rng.normal(0, 3.2, (H, W, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, name in enumerate(NAMES):
        top, bottom, blobs = PALETTES[i]
        im = make_image(11 + i * 7, top, bottom, blobs)
        path = os.path.join(OUT, name)
        im.save(path, quality=92)
        print("wrote", path, im.size)


if __name__ == "__main__":
    main()
