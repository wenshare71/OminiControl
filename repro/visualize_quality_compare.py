#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repro/visualize_quality_compare.py
==================================
Stage4 质量对比可视化 —— 把 records.json 里的 5 case × 2 seed × 2 模式渲成
[prompt | 条件图 | 默认 | KV-Cache] 网格,落盘 PNG,命令行里直接看结果。

用法:
    python repro/visualize_quality_compare.py
    python repro/visualize_quality_compare.py --records path/to/records.json --out my_grid.png

设计要点:
- 自洽:不依赖 notebook kernel 状态、OUT_DIR 变量。自动定位仓库根(向上找 .git/CLAUDE.md),
  任何 cwd 下跑都找得到 records.json。
- 离线:matplotlib 走 'Agg' backend,无 DISPLAY 也行(远程机器不需要 X server)。
- prompt 在左:用 axes[0] 的负坐标 text 画在条件图左侧空白处,加灰色圆角底视觉分组。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 离线 backend,无 DISPLAY 也能画
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


# ── 定位仓库根 ─────────────────────────────────────────────────────────────
# WHY: 脚本可能在任何 cwd 下被调用(crontab / 其它 notebook / 命令行),不能假设 cwd。
# 向上找 CLAUDE.md 或 .git 目录 —— 项目唯一的"我是仓库根"信号。
def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").is_file() and (p / ".git").exists():
            return p
    raise FileNotFoundError(
        f"从 {start} 向上找不到仓库根(需同时存在 CLAUDE.md 和 .git)。"
        "请在仓库内运行,或用 --repo-root 显式指定。")


# ── 加载 records ──────────────────────────────────────────────────────────
def load_records(records_path: Path) -> list[dict]:
    with open(records_path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{records_path} 是空的 —— 需要先跑 cell 5 生成数据。")
    # 兼容:records 里的图片路径可能是相对仓库根的(在 cell 5 里写出去的),用 repo_root 解析
    return records


# ── 画图 ──────────────────────────────────────────────────────────────────
def render(records: list[dict], out_dir: Path) -> Path:
    """每行 layout: [prompt(单行,跨 3 列) | 条件图 | 默认 | KV-Cache]

    设计:
        - 文字头只放 prompt,不加 case/seed/speedup(那些信息已在 ax.set_title 里)
        - wspace/hspace 压到 0.02,三张图紧贴、行与行紧贴
        - 文字行 height_ratio 0.12,只够放一行小字,不占视觉权重
    """
    n = len(records)
    fig = plt.figure(figsize=(12, 3.6 * n))
    height_ratios = []
    for _ in range(n):
        height_ratios.extend([0.12, 1.0])  # [text_row, image_row]
    gs = GridSpec(2 * n, 3, height_ratios=height_ratios,
                  hspace=0.05, wspace=0.02,
                  left=0.02, right=0.98, top=0.98, bottom=0.02)

    for i, r in enumerate(records):
        # ── 文字头:只放 prompt,单行,跨 3 列 ──
        ax_text = fig.add_subplot(gs[2 * i, :])
        ax_text.axis("off")
        ax_text.text(0.5, 0.5, r["prompt"],
                     ha="center", va="center", fontsize=8, wrap=True,
                     color="#333333")
        # ── 图像行:3 列 ──
        for j, (title, fp) in enumerate([
            (f"cond", out_dir / f"{r['case']}_cond.png"),
            (f"default · {r['default_sec']}s", out_dir / Path(r["default"]).name),
            (f"kv_cache · {r['kvcache_sec']}s", out_dir / Path(r["kvcache"]).name),
        ]):
            ax = fig.add_subplot(gs[2 * i + 1, j])
            if not fp.is_file():
                ax.text(0.5, 0.5, f"missing:\n{fp.name}",
                        ha="center", va="center", fontsize=7, color="red")
                ax.set_facecolor("#fff0f0")
            else:
                ax.imshow(Image.open(fp))
            ax.axis("off")
            ax.set_title(f"{r['case']} s{r['seed']} · {title}", fontsize=8)

    out_png = out_dir / "comparison_grid.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_png


# ── 打印人类可读摘要 ───────────────────────────────────────────────────────
def print_summary(records: list[dict]) -> None:
    print(f"\n{'='*70}\n对比摘要  ({len(records)} 组, 同 LoRA 同 seed)\n{'='*70}")
    for r in records:
        speedup = r["default_sec"] / r["kvcache_sec"]
        print(f"  {r['case']:8s} s{r['seed']:4d}  default={r['default_sec']:5.1f}s  "
              f"kvcache={r['kvcache_sec']:5.1f}s  speedup={speedup:.2f}x")
    avg = sum(r["default_sec"] / r["kvcache_sec"] for r in records) / len(records)
    print(f"  -> 平均 speedup: {avg:.2f}x")


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="仓库根目录(默认自动向上找)")
    ap.add_argument("--records", type=Path, default=None,
                    help="records.json 路径(默认 <repo>/repro/quality_compare/records.json)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="图片输出目录(默认与 records 同目录)")
    args = ap.parse_args()

    repo_root = args.repo_root or find_repo_root(Path(__file__).resolve().parent)
    records_path = args.records or repo_root / "repro" / "quality_compare" / "records.json"
    out_dir = args.out_dir or records_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not records_path.is_file():
        print(f"[ERROR] {records_path} 不存在 —— 先跑 cell 5 生成 records", file=sys.stderr)
        return 1

    print(f"[INFO] repo_root  : {repo_root}")
    print(f"[INFO] records    : {records_path}")
    print(f"[INFO] out_dir    : {out_dir}")

    records = load_records(records_path)
    out_png = render(records, out_dir)
    print(f"[INFO] 已生成     : {out_png}  ({out_png.stat().st_size // 1024} KB)")
    print_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
