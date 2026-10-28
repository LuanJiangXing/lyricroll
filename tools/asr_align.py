#!/usr/bin/env python3
"""
音频 → 歌词时间轴（不需要人工打点）
=====================================
思路：不指望模型认对字，只要它给出"这句大概在第几秒"，
再用你自己的歌词文本去套，错字当噪声吸收掉。

依赖（一次性）:
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple funasr-onnx

模型（首次自动下载，约 228MB，来自 ModelScope）:
    iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx

用法:
    python3 tools/asr_align.py \
        --audio assets/song.m4a --lyrics lyrics.txt \
        --out demo_timeline.json --shift 0.5

--lyrics 的 1-3 行如果是「标题 / 词： / 曲：」会被自动识别为片头信息。
--shift 是全局偏移（秒），正值＝整体往后推。
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
FILES = ["model_quant.onnx", "config.yaml", "am.mvn", "tokens.json"]
BASE = "https://modelscope.cn/models/%s/resolve/master" % MODEL_ID


def norm(s):
    return re.sub(r"[^一-鿿]", "", s)


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def cover(a, b):
    """a 里有多少字按顺序出现在 b 里。"""
    if not a or not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(x.size for x in sm.get_matching_blocks()) / len(a)


# ---------- 模型 ----------
def ensure_model(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for f in FILES:
        p = os.path.join(model_dir, f)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        print("下载", f, flush=True)
        subprocess.run(["curl", "-sSL", "-o", p, "%s/%s" % (BASE, f)], check=True)


# ---------- 音频 ----------
def to_wav16k(audio, path):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", audio, "-ac", "1", "-ar", "16000", path], check=True)
    import soundfile as sf
    return sf.read(path, dtype="float32")[0]


def windows(wav, sr, win, hop, dirpath):
    import soundfile as sf
    os.makedirs(dirpath, exist_ok=True)
    dur = len(wav) / sr
    paths, starts, t = [], [], 0.0
    while t + min(1.5, win) < dur:
        p = os.path.join(dirpath, "%05d.wav" % len(paths))
        sf.write(p, wav[int(t * sr):int(min(t + win, dur) * sr)], sr)
        paths.append(p)
        starts.append(round(t, 2))
        t += hop
    return paths, starts


def run_asr(model, paths):
    out = []
    res = model(paths)
    for r in res:
        p = r.get("preds")
        txt = p[0] if isinstance(p, (list, tuple)) else p
        out.append(norm(txt or ""))
    return out


# ---------- 对齐 ----------
def coarse(lines, starts, texts):
    """单调 DP：把所有句按顺序铺到时间轴上，最大化整句覆盖率。"""
    L = [norm(x) for x in lines]
    n, m = len(L), len(starts)
    S = [[cover(L[i], texts[j]) for j in range(m)] for i in range(n)]
    NEG = -1e9
    dp = [[NEG] * m for _ in range(n)]
    bk = [[-1] * m for _ in range(n)]
    for j in range(m):
        dp[0][j] = S[0][j]
    for i in range(1, n):
        best, bi = NEG, -1
        for j in range(m):
            if j > 0 and dp[i - 1][j - 1] > best:
                best, bi = dp[i - 1][j - 1], j - 1
            if best > NEG / 2:
                dp[i][j] = best + S[i][j]
                bk[i][j] = bi
    j = max(range(m), key=lambda k: dp[n - 1][k])
    path = [j]
    for i in range(n - 1, 0, -1):
        path.append(bk[i][path[-1]])
    path.reverse()
    return [(starts[j], S[i][j]) for i, j in enumerate(path)]


def fine(lines, est, wav, sr, model, tmpdir):
    """在粗定位附近 ±1.5/+1.0 秒内细扫，找句子开头第一次出现的位置。"""
    paths, starts, idx = [], [], []
    for i, (t, _) in enumerate(est):
        for k in range(-6, 5):
            b = max(0.0, round(t + k * 0.25, 2))
            p = os.path.join(tmpdir, "L%02d_%05d.wav" % (i, int(b * 100)))
            if not os.path.exists(p):
                import soundfile as sf
                sf.write(p, wav[int(b * sr):int(min(b + 2.2, len(wav) / sr) * sr)], sr)
            paths.append(p)
            starts.append(b)
            idx.append(i)
    texts = run_asr(model, paths)
    bucket = {}
    for i, b, tx in zip(idx, starts, texts):
        bucket.setdefault(i, []).append((b, tx))

    out, prev = [], -99.0
    for i, (t0, _) in enumerate(est):
        ln = norm(lines[i])
        head = ln[:3]
        cand = sorted(bucket.get(i, []))
        pick = None
        for b, tx in cand:
            if ratio(head, tx[:4]) >= 0.67 and ratio(head, tx[:8]) >= 0.45:
                pick = b
                break
        if pick is None:
            pick = max(cand, key=lambda x: ratio(ln, x[1]))[0] if cand else t0
        pick = max(pick, prev + 1.2)
        prev = pick
        out.append(round(pick, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=os.path.join(tempfile.gettempdir(), "asr", "paraformer"))
    ap.add_argument("--shift", type=float, default=0.0)
    args = ap.parse_args()

    from funasr_onnx import Paraformer

    raw = [l.strip() for l in open(args.lyrics, encoding="utf-8") if l.strip()]
    intro = None
    if raw and ("：" in raw[1] or ":" in raw[1]):
        intro = {"title": raw[0], "credits": raw[1:3]}
        raw = raw[3:]
    lines = raw

    ensure_model(args.model_dir)
    tmp = tempfile.mkdtemp(prefix="asr_")
    wav = to_wav16k(args.audio, os.path.join(tmp, "16k.wav"))
    sr = 16000
    model = Paraformer(args.model_dir, quantize=True, intra_op_num_threads=2, batch_size=16)

    print("粗扫…", flush=True)
    p1, s1 = windows(wav, sr, 3.0, 0.5, os.path.join(tmp, "w1"))
    t1 = run_asr(model, p1)
    est = coarse(lines, s1, t1)
    for i, (t, sc) in enumerate(est):
        print("  %2d %6.1f  %.2f  %s" % (i + 1, t, sc, lines[i]), flush=True)

    print("细对…", flush=True)
    times = fine(lines, est, wav, sr, model, os.path.join(tmp, "w2"))

    times = [round(t + args.shift, 2) for t in times]
    out = {"meta": {"title": intro["title"] if intro else ""},
           "duration": round(len(wav) / sr, 2),
           "lines": [{"t": t, "text": s} for t, s in zip(times, lines)]}
    if intro:
        out["intro"] = intro
    # 图片表给个占位，渲染前自己改
    out["images"] = [{"t": 0.0, "file": "assets/cover.jpg"}]
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("写出", args.out)
    for i, t in enumerate(times):
        print("%2d %7.2f  %s" % (i + 1, t, lines[i]))


if __name__ == "__main__":
    main()
