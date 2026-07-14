#!/usr/bin/env python3
"""
探测:kv_cache=True 下最多能放几张条件图不 OOM(方案一:256 条件 + position_scale=2)。

思路:
  条件数从 1 开始递增,每档只跑一次短程生成(默认 3 步 —— 步数不影响峰值显存,
  KV 写在第 1 步、读在第 2 步就已把缓存和注意力峰值都踩到了),OOM 即停,
  报告最后一个成功档位和各档的峰值显存。

用法(仓库根目录、双卡拆分环境,与 stage3 相同):
  python repro/probe_max_conditions.py
  OMINI_COND_SIZE=512 OMINI_MAX_TRY=4 python repro/probe_max_conditions.py   # 复核 512 条件的上限

可调环境变量:
  OMINI_CKPT_DIR / OMINI_LORA_WEIGHT  LoRA 位置(默认自动取 runs/ 下最新 ckpt)
  OMINI_TARGET_SIZE   生成尺寸(默认 512)
  OMINI_COND_SIZE     条件图尺寸(默认 256;=512 时 position_scale 自动为 1)
  OMINI_MAX_TRY       最多试到几张(默认 8)
  OMINI_STEPS         每档步数(默认 3,>=2 才会触发缓存读路径)
"""
import glob
import logging
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "repro"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s][probe] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("probe")

import torch  # noqa: E402
# 与 stage3 相同的 cuDNN 兜底(torch 2.8 + diffusers 0.38 的 VAE conv_in 问题)
torch.backends.cudnn.enabled = False

TARGET_SIZE = int(os.environ.get("OMINI_TARGET_SIZE", "512"))
COND_SIZE = int(os.environ.get("OMINI_COND_SIZE", "256"))
MAX_TRY = int(os.environ.get("OMINI_MAX_TRY", "8"))
STEPS = int(os.environ.get("OMINI_STEPS", "3"))
# position_scale 必须把条件 latent 网格拉伸到与目标一致,否则位置编码错位
POS_SCALE = TARGET_SIZE / COND_SIZE
IMAGE = "assets/vase_hq.jpg"
PROMPT = "A beautiful vase on a wooden table."


def _find_ckpt():
    d = os.environ.get("OMINI_CKPT_DIR")
    if d:
        return d, os.environ.get("OMINI_LORA_WEIGHT", "default.safetensors")
    hits = sorted(glob.glob("runs/*/ckpt/*/default.safetensors"), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError("runs/ 下找不到 ckpt,请设 OMINI_CKPT_DIR。")
    return os.path.dirname(hits[-1]), "default.safetensors"


def build_conditions(n_cond):
    from PIL import Image
    from omini.pipeline.flux_omini import Condition, convert_to_condition
    img = Image.open(IMAGE).convert("RGB").resize((COND_SIZE, COND_SIZE))
    cond_img = convert_to_condition("canny", img)
    # 复用同一张图 n 份:探测的是显存规模,与条件内容无关(同 kvcache_benchmark 的理由)
    return [Condition(cond_img, "feature_reuse", position_scale=POS_SCALE)
            for _ in range(n_cond)]


def main():
    from kvcache_benchmark import load_pipeline, attach_lora
    from omini.pipeline.flux_omini import generate

    ckpt_dir, weight = _find_ckpt()
    log.info("ckpt=%s cond=%dpx(pos_scale=%.1f) target=%dpx steps=%d 最多试 %d 张",
             ckpt_dir, COND_SIZE, POS_SCALE, TARGET_SIZE, STEPS, MAX_TRY)

    pipe = load_pipeline("black-forest-labs/FLUX.1-dev", dispatch="auto")
    attach_lora(pipe, ckpt_dir, weight, "feature_reuse")

    max_ok, results = 0, []
    for n in range(1, MAX_TRY + 1):
        conditions = build_conditions(n)
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.perf_counter()
        try:
            g = torch.Generator(device="cuda:0").manual_seed(42)
            generate(pipe, prompt=PROMPT, conditions=conditions,
                     height=TARGET_SIZE, width=TARGET_SIZE,
                     num_inference_steps=STEPS, guidance_scale=3.5,
                     generator=g, kv_cache=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("n_cond=%d  OOM ✗ —— 上限确定", n)
            break
        peaks = [torch.cuda.max_memory_allocated(d) / 1024**3
                 for d in range(torch.cuda.device_count())]
        max_ok = n
        results.append((n, peaks))
        log.info("n_cond=%d  OK ✓  %.1fs  峰值 %s",
                 n, time.perf_counter() - t0,
                 " ".join(f"cuda:{i}={p:.1f}G" for i, p in enumerate(peaks)))

    print("\n===== 结论 =====")
    print(f"条件图 {COND_SIZE}px + kv_cache=True:最多 {max_ok} 张不 OOM"
          f"(试到 {min(MAX_TRY, max_ok + 1)} 张为止)")
    for n, peaks in results:
        print(f"  n_cond={n}: 峰值 " +
              " ".join(f"cuda:{i}={p:.1f}G" for i, p in enumerate(peaks)))
    if max_ok == MAX_TRY:
        print(f"  注意:到 {MAX_TRY} 张仍未 OOM,可调大 OMINI_MAX_TRY 继续探。")


if __name__ == "__main__":
    main()
