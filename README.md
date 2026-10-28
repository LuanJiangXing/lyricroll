# 滚动歌词 MV

把一首歌做成一左图右词、整行滚动的视频。不用 AIGC，画面只来自你给的图。

歌词时间轴不用手打——语音模型只负责给时间，不采信它认的字，所以同音字错得再多也不影响结果。

## 需要

- Python 3.10+，`ffmpeg` 在 PATH
- 一条音轨、一个歌词文本（前 3 行可放标题和「词：」「曲：」）、若干张图
- 首次运行会从 ModelScope 自动下 228MB 的 Paraformer 模型

```bash
pip install -r requirements.txt
# 国内装不动就加镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple
```

纯 onnxruntime，不需要 PyTorch。

字体按平台自动找（Linux 思源宋体 / mac 宋体 / Windows 宋体或雅黑）。要指定自己的：

```bash
MV_FONT_REG=/path/to/font.ttf MV_FONT_BOLD=/path/to/font-bold.ttf MV_FONT_INDEX=0
```

## 跑

```bash
# 1. 对轴：音频 + 歌词 → 时间轴
python3 tools/asr_align.py --audio assets/song.m4a --lyrics lyrics.txt \
        --out timeline.json --shift 0.5

# 2. 把 timeline.json 的 images 改成本地图，然后渲染
python3 build_lyrics_video.py --timeline timeline.json \
        --audio assets/song.m4a --out out/song.mp4
```

`--shift` 是全局偏移，正值＝歌词整体往后推。渲染约 2.5 倍速。

`timeline.example.json` 是时间轴长什么样的最小示例，可以照着改。

## 文件

| 文件 | 作用 |
|---|---|
| `build_lyrics_video.py` | 渲染器。顶部常量区是全部版式参数 |
| `tools/asr_align.py` | 音频 → 时间轴 |
| `timeline.example.json` | 时间轴 + 图片表 + 片头信息的格式示例 |
| `WORKFLOW.md` | 原理、改动位置、已知边界 |
| `make_demo_assets.py` | 生成占位图，没有素材时用来试版式 |

`assets/` 和 `out/` 在 `.gitignore` 里，是个人素材和成片，不进仓库。

## 说明

- 时间是算出来的，不是听出来的，成品需要人耳过一遍
- 只做整行滚动，不做逐字卡拉 OK
- 换一首歌只需要重跑对轴脚本，版式参数一个都不用动
