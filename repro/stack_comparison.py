#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repro/stack_comparison.py
=========================
把 snapshot_denoise.py --both-modes 产出的 default + kvcache trajectory 拼成一张大图:

每个 case 一行,布局:
    ┌──────────────────────────────────────┐ ← prompt
    │                                      │
    ├──────┬───────────────────────────────┤
    │      │  DEFAULT 模式(右上)             │
    │ cond │  noise→step4→...→step28         │
    │(左,  ├───────────────────────────────┤
    │跨两行)│  KVCACHE 模式(右下)             │
    │      │  noise→step4→...→step28         │
    └──────┴───────────────────────────────┘

依赖:
    snapshot_denoise.py --all --every 4 --both-modes  ✓ 已跑过
    产物:
        <snapshots>/{case}_cond.png
        <snapshots>/{case}_default_trajectory.png
        <snapshots>/{case}_kvcache_trajectory.png

用法:
    python repro/stack_comparison.py
    python repro/stack_comparison.py --out-dir repro/quality_compare_256/snapshots \
                                     --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


# ── case 列表 + prompt(与 snapshot_denoise.py / stage4 notebook 完全一致) ──
CASES = [
    ("vase",    "A beautiful ceramic vase with flowers on a wooden table, soft light."),
    ("room",    "A cozy room corner with an armchair and warm afternoon lighting."),
    ("oranges", "Fresh ripe oranges on a plate, studio product photography."),
    ("clock",   "A vintage alarm clock on a desk, detailed, photorealistic."),
    ("rc_car",  "A red remote control toy car on the ground, sharp focus."),
]


def find_repo_root(start: Path) -> Path:
    """与 snapshot_denoise.py / visualize_quality_compare.py 同款约定。"""
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").is_file() and (p / ".git").exists():
            return p
    raise FileNotFoundError(f"找不到仓库根:{start}")


def load_font(size: int):
    """优先 DejaVuSans(远程机器有),失败 fallback 默认字体。"""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render(out_dir: Path, out_path: Path, traj_scale: float = 0.6) -> Path:
    """主拼图函数。

    Args:
        traj_scale: trajectory 条带的缩放比例(默认 0.6)。
            - 1.0 = 2076×312 大小,5 case 拼起来 canvas ≈ 2300×3300
            - 0.6 = 1246×187,canvas ≈ 1640×2230(更易看)
            - 0.5 = 更小更紧凑
    """
    # ── 加载所有素材 + 检查完整性 ──
    errors: List[str] = []
    cond_imgs, default_trajs, kvcache_trajs, prompts = {}, {}, {}, {}
    for case, prompt in CASES:
        prompts[case] = prompt
        for label, store, fname in [
            ("cond", cond_imgs, f"{case}_cond.png"),
            ("default_traj", default_trajs, f"{case}_default_trajectory.png"),
            ("kvcache_traj", kvcache_trajs, f"{case}_kvcache_trajectory.png"),
        ]:
            p = out_dir / fname
            if not p.is_file():
                errors.append(str(p))
                continue
            store[case] = Image.open(p)

    if errors:
        miss = "\n  - ".join(errors)
        raise FileNotFoundError(
            f"缺少以下文件(先跑 snapshot_denoise.py --both-modes 生成):\n  - {miss}")

    # ── 几何参数 ──
    sample_traj = next(iter(default_trajs.values()))
    traj_w_raw, traj_h_raw = sample_traj.size
    # trajectory 缩放后尺寸(保持比例)
    new_traj_w = int(traj_w_raw * traj_scale)
    new_traj_h = int(traj_h_raw * traj_scale)

    # prompt 行高
    prompt_h = 40

    # 右侧两个 trajectory 行 + 行间 gap
    inter_gap = 6
    right_col_h = 2 * new_traj_h + inter_gap

    # cond 跨两行,1:1 aspect(条件图本来就是方形 canny)
    cond_h = right_col_h
    # cond 取原始 aspect(可能不是正方形,虽然 canny 是):用 Image.thumbnail 等比
    cond_raw = next(iter(cond_imgs.values()))
    cond_w = cond_h  # 把 cond_w 强制等于 cond_h(方形),与 right_col 等高

    # ── canvas 总尺寸 ──
    margin = 24
    label_band = 28  # 左边写 "DEFAULT 模式"/"KVCACHE 模式" 的窄竖条
    case_v_gap = 18  # case 之间竖向间距

    content_w = label_band + cond_w + 6 + new_traj_w
    canvas_w = content_w + 2 * margin

    per_case_h = prompt_h + right_col_h  # 每行 case 的高度
    canvas_h = margin + 5 * per_case_h + 4 * case_v_gap + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # ── 字体 ──
    font_prompt = load_font(14)
    font_label = load_font(14)
    font_section = load_font(13)

    # ── 顶层 section header(写在 canvas 顶部,标注整张图的语义) ──
    section_text = ("Stage4 quality diff: kv-cache vs default (28 steps, every-4 snapshots, "
                    "left/right top = default, left/right bottom = kv-cache)")
    try:
        sb = draw.textbbox((0, 0), section_text, font=font_section)
        sec_w = sb[2] - sb[0]
        draw.text(((canvas_w - sec_w) // 2, 6), section_text, fill="#666666", font=font_section)
        # 给 section header 让出 26px 顶部空间
        top_pad = 30
    except Exception:
        top_pad = 0

    # ── 主循环:逐 case 拼图 ──
    y = margin + top_pad
    for case_idx, (case, prompt) in enumerate(CASES):
        x0 = margin

        # ── prompt 行 ──
        bbox = draw.textbbox((0, 0), prompt, font=font_prompt)
        pw = bbox[2] - bbox[0]
        # prompt 在 cond + traj 中点上方(垂直居中于 prompt_h)
        text_x = x0 + (content_w - pw) // 2
        text_y = y + (prompt_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((text_x, text_y), prompt, fill="#222222", font=font_prompt)

        # ── cond 列(左,跨两行) ──
        cond_resized = cond_imgs[case].resize((cond_w, cond_h), Image.LANCZOS)
        canvas.paste(cond_resized, (x0, y + prompt_h))

        # 在 cond 列左侧画个窄条说明
        # 第一行(对应 default):标 "DEFAULT"
        # 第二行(对应 kvcache):标 "KVCACHE"

        # ── 行 1: DEFAULT 模式(右上的 trajectory) ──
        row1_y = y + prompt_h
        default_resized = default_trajs[case].resize((new_traj_w, new_traj_h), Image.LANCZOS)
        canvas.paste(default_resized,
                     (x0 + label_band + cond_w + 6, row1_y))

        # ── 行 2: KVCACHE 模式(右下) ──
        row2_y = row1_y + new_traj_h + inter_gap
        kv_resized = kvcache_trajs[case].resize((new_traj_w, new_traj_h), Image.LANCZOS)
        canvas.paste(kv_resized,
                     (x0 + label_band + cond_w + 6, row2_y))

        # ── 左侧 "DEFAULT"/"KVCACHE" 两条窄竖列标签 ──
        # WHY 写在 cond 左侧的窄条:同一行 visual group,不用额外 header
        # 用浅灰背景 + 黑字区分两行
        # 行 1 背景
        draw.rectangle(
            [(x0, row1_y),
             (x0 + label_band, row1_y + new_traj_h)],
            fill="#f0f0f0")
        draw.text(
            (x0 + 4, row1_y + new_traj_h // 2 - 8),
            "DEFAULT", fill="#222222", font=font_label)
        # 行 2 背景
        draw.rectangle(
            [(x0, row2_y),
             (x0 + label_band, row2_y + new_traj_h)],
            fill="#fff5e0")  # 暖底色突出 kvcache 行
        draw.text(
            (x0 + 4, row2_y + new_traj_h // 2 - 8),
            "KVCACHE", fill="#a04a00", font=font_label)

        # 推进 y
        y += per_case_h + case_v_gap

    canvas.save(out_path, optimize=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--snapshots-dir", type=Path, default=None,
                    help="snapshot_denoise.py 的 --out-dir(默认 <repo>/repro/quality_compare_256/snapshots)")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 png 路径(默认 <snapshots-dir>/comparison_grid.png)")
    ap.add_argument("--traj-scale", type=float, default=0.6,
                    help="trajectory 条带缩放比例(默认 0.6)")
    args = ap.parse_args()

    repo_root = args.repo_root or find_repo_root(Path(__file__).resolve().parent)
    snap_dir = args.snapshots_dir or repo_root / "repro" / "quality_compare_256" / "snapshots"
    if not snap_dir.is_dir():
        print(f"[ERROR] {snap_dir} 不存在,先跑 snapshot_denoise.py --both-modes",
              file=sys.stderr)
        return 1

    out_path = args.out or snap_dir / "comparison_grid.png"
    print(f"[INFO] snapshots dir : {snap_dir}")
    print(f"[INFO] output         : {out_path}")
    print(f"[INFO] traj scale     : {args.traj_scale}")

    saved = render(snap_dir, out_path, traj_scale=args.traj_scale)
    sz = saved.stat().st_size // 1024
    print(f"[INFO] 生成 {saved.name} ({sz} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
