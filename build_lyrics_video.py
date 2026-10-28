#!/usr/bin/env python3
"""
整行滚动歌词 MV —— 左图右词（v2）
--------------------------------
v2 改动：同时只显示三句 / 换衬线字体 / 左图缩小居中不占满 / 右栏只剩文字

时间轴走外部 JSON，改词、改秒数不用动这个脚本。

用法:
  python3 build_lyrics_video.py --timeline demo_timeline.json --audio assets/demo-pad.m4a --out out/x.mp4
  python3 build_lyrics_video.py ... --start 0 --end 8 --out out/preview.mp4

时间轴格式:
{
  "meta":     {"title": "…"},
  "duration": 43.6,
  "images":   [{"t": 0.0, "file": "assets/a.jpg"}, …],   # t = 切换点
  "lines":    [{"t": 3.0, "text": "……"}, …]              # t = 这句开始唱的时刻
}
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

# ---------- 版式（要改都在这） ----------
W, H = 1920, 1080
FPS = 30

BG = (13, 14, 16)

SPLIT = 1120                       # 左右分界
IMG_BOX_W, IMG_BOX_H = 780, 880    # 左图最大尺寸，按原始比例塞进去
IMG_ZOOM = 1.05                    # 每段内的缓慢推近
IMG_FADE = 0.70                    # 换图交叉淡入

TEXT_X = 1192                      # 歌词左边线
TEXT_RIGHT = 1860
LINE_H = 112                       # 行距
CENTER_Y = 540

FONT_ACTIVE = 52
FONT_IDLE = 36
TEXT_ACTIVE = (246, 244, 239)
TEXT_IDLE = (128, 130, 136)

TITLE_SIZE = 66                    # 前奏标题字号
CREDIT_SIZE = 24                   # 词曲字样号

VISIBLE = 1.66                     # 超过 ±1.66 行就完全不显示 → 同时三句

# 衬线中文字体，按平台找。找不到就用环境变量指定：
#   MV_FONT_REG / MV_FONT_BOLD / MV_FONT_INDEX
FONT_SEARCH = [
    # (regular, bold, ttc 里的序号)
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
     "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 2),      # Linux
    ("/System/Library/Fonts/Supplemental/Songti.ttc",
     "/System/Library/Fonts/Supplemental/Songti.ttc", 0),             # macOS
    ("C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simsun.ttc", 0),  # Windows 宋体
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", 0),    # Windows 雅黑
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
     "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
]


def resolve_fonts():
    reg = os.environ.get("MV_FONT_REG")
    bold = os.environ.get("MV_FONT_BOLD")
    idx = int(os.environ.get("MV_FONT_INDEX", "0"))
    if reg:
        return reg, bold or reg, idx
    for r, b, i in FONT_SEARCH:
        if not (os.path.exists(r) and os.path.exists(b)):
            continue
        try:
            ImageFont.truetype(r, 20, index=i)
            ImageFont.truetype(b, 20, index=i)
            return r, b, i
        except OSError:
            continue
    sys.exit("找不到中文字体，用 MV_FONT_REG / MV_FONT_BOLD / MV_FONT_INDEX 指定")


FONT_REG, FONT_BOLD, FONT_INDEX_SC = resolve_fonts()
_font_cache = {}


def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(
            FONT_BOLD if bold else FONT_REG, size, index=FONT_INDEX_SC
        )
    return _font_cache[key]


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def mix(c1, c2, t):
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


# ---------- 左图 ----------
def fit_size(w, h):
    s = min(IMG_BOX_W / w, IMG_BOX_H / h)
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def prep_image(path):
    """按 fit 尺寸铺满后多留 20% 余量，供推近裁切。"""
    src = Image.open(path).convert("RGB")
    sw, sh = src.size
    dw, dh = fit_size(sw, sh)
    tw, th = int(dw * 1.2), int(dh * 1.2)
    scale = max(tw / sw, th / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    src = src.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - tw) // 2, (nh - th) // 2
    return src.crop((l, t, l + tw, t + th)), (dw, dh)


class LeftPane:
    def __init__(self, segments, end_of_timeline):
        self.segs = sorted(segments, key=lambda s: s["t"])
        self.imgs, self.sizes = [], []
        for s in self.segs:
            im, size = prep_image(s["file"])
            self.imgs.append(im)
            self.sizes.append(size)
        self.end = end_of_timeline

    @staticmethod
    def _one(prepped, size, zoom):
        sw, sh = prepped.size
        cw, ch = sw / zoom, sh / zoom
        box = ((sw - cw) / 2, (sh - ch) / 2, (sw + cw) / 2, (sh + ch) / 2)
        return prepped.resize(size, Image.BILINEAR, box=box)

    def frame(self, t):
        segs = self.segs
        idx = 0
        for i, s in enumerate(segs):
            if t >= s["t"]:
                idx = i
        start = segs[idx]["t"]
        stop = segs[idx + 1]["t"] if idx + 1 < len(segs) else self.end
        span = max(stop - start, 0.1)
        z = 1.0 + (IMG_ZOOM - 1.0) * clamp((t - start) / span)
        cur = self._one(self.imgs[idx], self.sizes[idx], z)

        if idx > 0 and t < start + IMG_FADE / 2:
            pstart = segs[idx - 1]["t"]
            pspan = max(start - pstart, 0.1)
            pz = 1.0 + (IMG_ZOOM - 1.0) * clamp((t - pstart) / pspan)
            prev = self._one(self.imgs[idx - 1], self.sizes[idx - 1], pz)
            if prev.size != cur.size:
                prev = prev.resize(cur.size, Image.BILINEAR)
            u = smoothstep((t - (start - IMG_FADE / 2)) / IMG_FADE)
            return Image.blend(prev, cur, u)
        return cur

    def paste_at(self, frame, im):
        dw, dh = im.size
        frame.paste(im, ((SPLIT - dw) // 2, (H - dh) // 2))


# ---------- 歌词 ----------
class LyricScroll:
    def __init__(self, lines):
        self.lines = sorted(lines, key=lambda l: l["t"])
        self.starts = [l["t"] for l in self.lines]

    def pos(self, t):
        starts = self.starts
        if not starts:
            return 0.0
        if t < starts[0]:
            return 0.0
        k = 0
        for i, s in enumerate(starts):
            if t >= s:
                k = i
        if k >= len(starts) - 1:
            return float(k)
        gap = starts[k + 1] - starts[k]
        if gap <= 0.05:
            return float(k)
        move = min(0.75, gap * 0.55)
        frac = (t - starts[k]) / gap
        u0 = 1.0 - move / gap
        if frac <= u0:
            return float(k)
        return k + smoothstep((frac - u0) / (move / gap))


def draw_intro(d, intro, alpha):
    """前奏的标题 / 词曲，alpha 归零后自然让位给第一句。"""
    if alpha <= 0.004:
        return
    title = intro.get("title", "")
    if title:
        d.text((0, CENTER_Y - 52), title, font=get_font(TITLE_SIZE, True),
               fill=TEXT_ACTIVE + (int(255 * alpha),), anchor="lm")
    for i, c in enumerate(intro.get("credits", [])):
        d.text((0, CENTER_Y + 46 + i * 34), c, font=get_font(CREDIT_SIZE),
               fill=TEXT_IDLE + (int(240 * alpha),), anchor="lm")


def draw_lyrics(frame, scroll, t, intro=None):
    overlay = Image.new("RGBA", (W - TEXT_X, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay, "RGBA")
    avail = TEXT_RIGHT - TEXT_X

    first = scroll.starts[0] if scroll.starts else 1e9
    if intro:
        lyric_in = smoothstep((t - (first - 1.4)) / 1.2)
        title_a = (1.0 - smoothstep((t - (first - 2.4)) / 1.8)) * smoothstep(t / 1.2)
        draw_intro(d, intro, title_a)
    else:
        lyric_in = 1.0
    if lyric_in <= 0.004:
        frame.paste(overlay, (TEXT_X, 0), overlay)
        return

    pos = scroll.pos(t)
    for i, line in enumerate(scroll.lines):
        d_off = i - pos
        ad = abs(d_off)
        if ad > VISIBLE:
            continue
        act = clamp(1.0 - ad / 0.72)
        alpha = clamp((VISIBLE - ad) / 0.62) * lyric_in
        if alpha <= 0.005:
            continue

        size = int(round(FONT_IDLE + (FONT_ACTIVE - FONT_IDLE) * act))
        text = line["text"]
        f = get_font(size, bold=True)
        while f.getlength(text) > avail and size > 20:
            size -= 2
            f = get_font(size, bold=True)

        y = CENTER_Y + d_off * LINE_H
        col = mix(TEXT_IDLE, TEXT_ACTIVE, act)
        d.text((0, y), text, font=f, fill=col + (int(255 * alpha),), anchor="lm")

    frame.paste(overlay, (TEXT_X, 0), overlay)


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    args = ap.parse_args()

    with open(args.timeline, encoding="utf-8") as fh:
        tl = json.load(fh)
    duration = float(tl["duration"])

    t0 = 0.0 if args.start is None else max(0.0, args.start)
    t1 = duration if args.end is None else min(duration, args.end)
    nframes = int(round((t1 - t0) * args.fps))
    if nframes <= 0:
        sys.exit("时间范围为 0，检查 --start/--end")

    left = LeftPane(tl["images"], duration)
    scroll = LyricScroll(tl["lines"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "%dx%d" % (W, H), "-r", str(args.fps), "-i", "-",
    ]
    if args.audio:
        cmd += ["-ss", str(t0), "-i", args.audio]
    cmd += [
        "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if args.audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += [args.out]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for n in range(nframes):
            t = t0 + n / args.fps
            frame = Image.new("RGB", (W, H), BG)
            left.paste_at(frame, left.frame(t))
            draw_lyrics(frame, scroll, t, tl.get("intro"))
            proc.stdin.write(frame.tobytes())
            if n % 200 == 0:
                print("  %d/%d  %.1fs" % (n, nframes, t), flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        sys.exit("ffmpeg 退出码 %d" % rc)
    print("done:", args.out, "%.1fs / %d 帧" % (t1 - t0, nframes))


if __name__ == "__main__":
    main()
