#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repro/snapshot_denoise.py
=========================
对每张目标图跑一次去噪,每 N 步解码 latent 成 PNG,横向拼成 噪声 → 图像 的进度图。

与 stage4_quality_compare.ipynb 的差异:
    - 单 case 单模式 (--case / --mode),或一轮全部 (--all)
    - 在去噪循环里挂 callback 抓 latent,VAE decode → PNG
    - 末尾用 PIL 把所有快照横拼成 noise → clean 大图(左侧标签写步数)

设计要点:
    - 自洽:不依赖 notebook kernel;跟 visualize_quality_compare.py 一样自动定位
      仓库根(向上找 .git + CLAUDE.md),任何 cwd 下跑都 OK。
    - 完全离线:HF_HUB_OFFLINE=1 + 直接传本地 snapshot 路径,绕开 gated repo auth。
    - 复用 stage4 notebook 的所有预设(CASES、LoRA、condition 转换、同 seed
      同噪声),保证与 stage4 records.json 同口径、可视化结果直接对接。

用法:
    # 单 case + default 模式(最常用) + 每 4 步一张
    python repro/snapshot_denoise.py --case vase --mode default --every 4

    # 单 case + KV-Cache
    python repro/snapshot_denoise.py --case vase --mode kvcache --every 4

    # 一轮全部 5 case × 2 模式 = 10 张轨迹图
    python repro/snapshot_denoise.py --all --every 4

    # 自定义分辨率(与 stage4 的 TARGET_SIZE / COND_SIZE 对齐)
    python repro/snapshot_denoise.py --case vase --target-size 512 --cond-size 256 --every 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # 离线 backend,无 DISPLAY 也能画(只用于保存 PIL/拼图不需要这个,但保险)
import torch

# ─── cudnn workaround(与 stage4_quality_compare.ipynb 完全一致) ──────────────
# WHY:torch 2.8 + diffusers 0.38 + bf16 在某些 GPU 上跑 VAE conv2d 时
# 报 CUDNN_STATUS_NOT_INITIALIZED,关掉 cudnn 走原生 eager 路径即可。
# 必须在首次 cuda op 之前设置。
torch.backends.cudnn.enabled = False

from PIL import Image


# ─── CASES(与 stage4_quality_compare.ipynb cell 4 完全一致) ─────────────────
# WHY 复用同一组:本脚本产物可与 stage4 的 records.json 在视觉上直接对齐,
# 比如同一 case s42 看 default vs kvcache 的轨迹差异,无需重选 case。
CASES = [
    ("vase",    "assets/vase_hq.jpg",     "A beautiful ceramic vase with flowers on a wooden table, soft light."),
    ("room",    "assets/room_corner.jpg", "A cozy room corner with an armchair and warm afternoon lighting."),
    ("oranges", "assets/oranges.jpg",     "Fresh ripe oranges on a plate, studio product photography."),
    ("clock",   "assets/clock.jpg",       "A vintage alarm clock on a desk, detailed, photorealistic."),
    ("rc_car",  "assets/rc_car.jpg",      "A red remote control toy car on the ground, sharp focus."),
]


# ── 仓库根定位 ──────────────────────────────────────────────────────────────
# WHY 复用而非新写:visualize_quality_compare.py 已证明这个方法在各种 cwd 下都能找到。
def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").is_file() and (p / ".git").exists():
            return p
    raise FileNotFoundError(
        f"从 {start} 向上找不到仓库根(需同时存在 CLAUDE.md 和 .git)。"
        "请在仓库内运行,或用 --repo-root 显式指定。")


# ── 默认 FLUX 路径(沿用 stage4 notebook 的本地 snapshot) ────────────────────
DEFAULT_LOCAL_FLUX_PATH = (
    "/home/wuwenxuan03/.cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.1-dev/"
    "snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)


# ── 找最新 ckpt(沿用 stage4 的兜底 glob 逻辑) ──────────────────────────────
def find_latest_ckpt(repo_root: Path) -> str:
    """取 runs/*/ckpt/*/default.safetensors 中 mtime 最新那个的 ckpt 目录。"""
    candidates = sorted(
        glob.glob(str(repo_root / "runs" / "*" / "ckpt" / "*" / "default.safetensors")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "找不到 runs/*/ckpt/*/default.safetensors。可用 --ckpt-dir 显式指定,"
            "或设环境变量 OMINI_CKPT_DIR。")
    return os.path.dirname(candidates[-1])


# ── VAE 解码(完全照搬 generate() 行 832-836) ────────────────────────────────
# WHY 自己写一份而不调 generate() 的最后阶段:我们要在循环中段抓 latent,
# 不能等循环结束。逻辑跟 generate() 末尾一行不差,保证出图与 stage4 主图严格一致。
#
# 关键:整个函数必须包在 torch.no_grad() 里!
# WHY:peft 在 set_adapters 后会让相关模块的输出 requires_grad=True,即便它的
# params 不是可训练。原因可能是某个 hooks 的实现细节。这对我们解码中间 latent
# 没影响 —— 中间 latent 本来就不用反传 —— 但 image_processor.postprocess() 内部
# 调 .cpu().numpy() 时会因 requires_grad=True 而炸 RuntimeError。
# stage4 notebook 看起来没踩这个坑是因为它(1)只 decode 最终 latent,(2)generate()
# 末尾的 vae.decode 是在 torch.is_grad_enabled()==False 的隐式 context 里跑的。
# 我们的循环 mid-step decode 没有这个保护,所以显式加。
@torch.no_grad()
def decode_latents(pipe, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(latents, return_dict=False)[0]
    pil = pipe.image_processor.postprocess(img, output_type="pil")
    return pil[0]


# ── VAE 在 cuda:0,scheduler/latents 也在 cuda:0(主场),但 transformer 在 cuda:1 ──
# decode_latents 拿到的是循环里的 latents(已经在 cuda:0),无需搬卡。
# 如果未来部署改了 dispatch,需要在这里加 _move(latents, "cuda:0")。


# ── 主流程:跑一个 case,产 8 张快照 + 1 张轨迹拼图 ──────────────────────────
def run_one(
    pipe, *, case_name: str, case_path: str, prompt: str,
    height: int, width: int, cond_size: int, pos_scale: float,
    steps: int, every: int, seed: int, mode: str,
    out_dir: Path, tag: str = "",
) -> None:
    """单 case 跑一次去噪,产 per-step PNG + 横向 noise→clean 拼图。

    Args:
        mode: "default" 或 "kvcache"
        tag:  可选后缀(如 "vase_e4" 用于区分不同 every),会拼进文件名。
    """
    from omini.pipeline.flux_omini import Condition, convert_to_condition, generate

    out_dir.mkdir(parents=True, exist_ok=True)
    is_kv = (mode == "kvcache")
    suffix = f"_{tag}" if tag else ""
    print(f"\n[INFO] case={case_name} mode={mode} steps={steps} "
          f"every={every} seed={seed} target={height}x{width} cond={cond_size}")

    # ── 构造条件(与 stage4 notebook cell 4 build_condition 一致) ──
    img = Image.open(case_path).convert("RGB").resize((cond_size, cond_size))
    cond_vis = convert_to_condition("canny", img)          # 纯展示用(PIL)
    condition = Condition(cond_vis, "feature_reuse",
                          position_scale=pos_scale)        # 注入 LoRA 名字 + 位置缩放
    cond_vis.save(out_dir / f"{case_name}{suffix}_cond.png")

    # ── 初始化 generator + 准备初始 latent(用来 decode step 0) ──
    # WHY 自己 prepare:循环外的 latent 就是"未做任何去噪的纯噪声"快照。
    # 之后把同一 tensor 喂给 generate(),保证两路从完全一致的起点出发,
    # 跟 stage4 notebook 的"同 seed 同噪声"约定一致。
    g = torch.Generator(device="cuda:0").manual_seed(seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    initial_latents, _ = pipe.prepare_latents(
        1, num_channels_latents, height, width,
        torch.bfloat16, "cuda:0", g, None,
    )

    # ── snapshot 列表:(label, PIL.Image) ──
    snapshots: List[Tuple[str, Image.Image]] = []

    # 先把 step 0 (noise) 拍下来
    step0_img = decode_latents(pipe, initial_latents, height, width)
    snapshots.append((f"step 0/{steps}\n(noise)", step0_img))

    # ── 定义每步 callback ──
    # 注意:generate() 内部在 scheduler.step 之后调 callback,此时 latents 已
    # 被推到下一个 sigma。所以"done = i + 1"对应"已完成 done 步去噪"。
    def callback(p, i, t, cb_kwargs):
        lat = cb_kwargs["latents"]
        done = i + 1
        # 边界:步数刚好是 every 的倍数,或者最后一步
        if done % every == 0 or i == steps - 1:
            img = decode_latents(pipe, lat, height, width)
            snapshots.append((f"step {done}/{steps}", img))
            print(f"  [snap] step {done}/{steps} captured")
        return cb_kwargs

    # ── 跑 generate(用同一份 initial_latents,seed 不再重要,因为 latent 已给定) ──
    g_unused = torch.Generator(device="cuda:0").manual_seed(seed)  # 占位,实际被 latents= 覆盖
    t0 = time.perf_counter()
    out = generate(
        pipe,
        prompt=prompt,
        conditions=[condition],
        height=height, width=width,
        num_inference_steps=steps,
        guidance_scale=3.5,
        generator=g_unused,
        latents=initial_latents,
        kv_cache=is_kv,
        callback_on_step_end=callback,
    )
    dt = time.perf_counter() - t0

    # ── 持久化单帧 PNG ──
    # WHY 单帧 + 拼图都给:拼图方便一眼看轨迹;单帧方便后续做插值/fade 视频。
    for label, im in snapshots:
        # 把 label 里 "/ \n" 改成 "_" 用作文件名
        safe = label.replace("/", "of").replace("\n", "_").replace("(", "").replace(")", "").replace(" ", "_")
        im.save(out_dir / f"{case_name}{suffix}_{safe}.png")

    # ── 末尾 PIL 横向拼接 ──
    # 设计:每张缩到 256 高(保持宽高比),间距 0,顶端留 32px 写标签。
    # WHY PIL 而非 matplotlib:PIL 更轻,不依赖字体;拼图我自己也写过几版,
    # matplotlib 在边距/对齐上不如 PIL 可控。
    pad = 4
    label_h = 56                      # 够两行 14pt + 上下余量,避免 "step 0/28\n(noise)" 溢出
    target_h = 256
    tiles = []
    for label, im in snapshots:
        scale = target_h / im.height
        new_w = int(im.width * scale)
        tiles.append((label, im.resize((new_w, target_h), Image.LANCZOS)))

    total_w = sum(t.width for _, t in tiles) + pad * (len(tiles) - 1)
    canvas = Image.new("RGB", (total_w, target_h + label_h), "white")

    # 写标签(用 PIL.ImageDraw,默认字体即可,无中文需求)
    # 用 multiline textbbox 算高度 → 顶部留 6px 上 padding,底部再留一些,
    # 保证两行标签不会贴到图像边缘。
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    x = 0
    for label, im in tiles:
        # 标签多行居中:先 bbox 算宽高(无 anchor 时 xy 必须是左上角)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        # 水平:让文字中心落在 tile 中心 → x = x + (im.width - tw) // 2
        # 垂直:让文字块在 label_h 范围内垂直居中 → y = (label_h - th) // 2 - bbox[1]
        # (bbox[1] 通常是负数,代表字体的 ascender,减去它避免字符顶部被裁)
        draw.multiline_text(
            (x + (im.width - tw) // 2 - bbox[0],
             (label_h - th) // 2 - bbox[1]),
            label, fill="black", font=font, align="center",
        )
        canvas.paste(im, (x, label_h))
        x += im.width + pad

    strip_path = out_dir / f"{case_name}{suffix}_trajectory.png"
    canvas.save(strip_path)

    # ── 落 records.jsonl 便于与 stage4 records 对齐 ──
    rec = {
        "case": case_name,
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "every": every,
        "n_snapshots": len(snapshots),
        "wall_s": round(dt, 2),
        "strip_png": str(strip_path.relative_to(out_dir)),
        "snapshot_labels": [label for label, _ in snapshots],
    }
    with open(out_dir / f"{case_name}{suffix}_record.json", "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 生成 {len(snapshots)} 张快照, 拼图 {strip_path.name}  "
          f"({strip_path.stat().st_size // 1024} KB), 用时 {dt:.2f}s")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--flux-path", default=DEFAULT_LOCAL_FLUX_PATH,
                    help="FLUX.1-dev snapshot 路径(默认本地,绕开 gated repo)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="LoRA ckpt 目录(含 default.safetensors),默认取 runs/ 下最新")
    ap.add_argument("--adapter-name", default="feature_reuse")

    # ── case 选择:二选一 ──
    ap.add_argument("--case", choices=[c[0] for c in CASES], default=None,
                    help="单个 case;与 --all 互斥")
    ap.add_argument("--all", action="store_true",
                    help="跑全部 5 个 case(每个 case 都产图)")

    # ── 模式 ──
    ap.add_argument("--mode", choices=["default", "kvcache"], default="default")
    ap.add_argument("--both-modes", action="store_true",
                    help="default 和 kvcache 各跑一遍(用于直接对比)")

    # ── 分辨率 / 步数 / 间隔(与 stage4 cell 1 对齐) ──
    ap.add_argument("--target-size", type=int, default=256)
    ap.add_argument("--cond-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--every", type=int, default=4,
                    help="每隔 N 步保存一张快照(默认 4 → 28/4+1=8 张)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录(默认 <repo>/repro/quality_compare_256/snapshots)")
    args = ap.parse_args()

    # ── sanity check ──
    if not args.case and not args.all:
        ap.error("必须指定 --case <name> 或 --all")
    if args.case and args.all:
        ap.error("--case 与 --all 互斥")

    # ── 离线 + 仓库路径 ──
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    repo_root = args.repo_root or find_repo_root(Path(__file__).resolve().parent)
    sys.path.insert(0, str(repo_root))

    # 校验 FLUX snapshot 存在(脚本期望离线,提前失败好过运行时困惑)
    if not Path(args.flux_path).joinpath("model_index.json").is_file():
        print(f"[ERROR] --flux-path 指向的 {args.flux_path} 不含 model_index.json。",
              file=sys.stderr)
        return 1

    out_dir = args.out_dir or repo_root / "repro" / "quality_compare_256" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] repo_root : {repo_root}")
    print(f"[INFO] out_dir   : {out_dir}")
    print(f"[INFO] flux_path : {args.flux_path}")

    # ── 加载 pipeline + LoRA(与 stage4 notebook cell 3 一致) ──
    from kvcache_benchmark import load_pipeline, attach_lora
    pipe = load_pipeline(args.flux_path, dispatch="auto")

    ckpt_dir = args.ckpt_dir or os.environ.get("OMINI_CKPT_DIR") or find_latest_ckpt(repo_root)
    print(f"[INFO] ckpt_dir  : {ckpt_dir}")
    attach_lora(pipe, ckpt_dir, "default.safetensors", args.adapter_name)

    # ── 选 case 子集 ──
    if args.case:
        selected = [c for c in CASES if c[0] == args.case]
    else:
        selected = list(CASES)

    # ── 选模式子集 ──
    if args.both_modes:
        modes = ["default", "kvcache"]
    else:
        modes = [args.mode]

    pos_scale = args.target_size / args.cond_size

    # ── 主循环 ──
    for case_name, case_path, prompt in selected:
        case_full_path = repo_root / case_path
        if not case_full_path.is_file():
            print(f"[WARN] case 图缺失,跳过 {case_name}: {case_full_path}", file=sys.stderr)
            continue
        for mode in modes:
            # 文件名后缀:仅当 both_modes 时把 mode 拼进去(单模式时避免文件名冗长)
            tag = mode if args.both_modes else ""
            run_one(
                pipe,
                case_name=case_name,
                case_path=str(case_full_path),
                prompt=prompt,
                height=args.target_size, width=args.target_size,
                cond_size=args.cond_size,
                pos_scale=pos_scale,
                steps=args.steps,
                every=args.every,
                seed=args.seed,
                mode=mode,
                out_dir=out_dir,
                tag=tag,
            )

    print(f"\n[INFO] 全部完成。产物目录:{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
