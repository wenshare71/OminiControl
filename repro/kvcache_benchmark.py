#!/usr/bin/env python3
"""
OminiControl2 · Feature Reuse (KV-Cache) 复现基准脚本
=====================================================

目标:在同一台机器、同一 seed / prompt / 条件图下,只切换 `kv_cache` 开关,
测量 KV-Cache 带来的【墙钟加速】【峰值显存】变化,并检查【生成质量是否保持】。

论文口径提醒(务必先读):
  - 仓库 README 自测:单条件 8 步 ≈ 1.5x 端到端加速。
  - 论文标题 5.9x / >90%:是【多条件】场景、且专指【条件分支处理开销】的相对降低,
    不是单条件端到端。所以本脚本【必须扫 num_conditions】才能看到接近论文的曲线——
    KV-Cache 省掉的计算量 ∝ (条件分支数 × (总步数 - 1)),条件越多、步数越多,收益越大。

前置条件:
  1. 在仓库根目录运行(保证 `import omini` 可用),或用 --repo-root 指定。
  2. LoRA 必须用 `independent_condition: true` 训练(train/script/train_feature_reuse.sh)。
     用未独立训练的权重也能跑、速度数字有效,但质量对比无意义(会掉),脚本会告警。

用法示例:
  python kvcache_benchmark.py \
      --lora-repo runs/feature_reuse_canny/ckpt --lora-weight pytorch_lora_weights.safetensors \
      --adapter-name canny --condition-type canny \
      --steps 8,20,28 --conditions 1,2,3 --repeats 3 --image assets/vase_hq.jpg

  # 冒烟测试(拿现成 v1 权重先验证管线,质量不作数):
  python kvcache_benchmark.py --lora-repo Yuanshi/OminiControl \
      --lora-weight experimental/canny.safetensors --adapter-name canny \
      --condition-type canny --steps 8 --conditions 1 --repeats 2 --no-independent-check
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

# ---- 日志:带时间戳 + 级别,排查远程机器问题时能对上 wandb / nvidia-smi 时间线 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kvbench")


@dataclass
class BenchResult:
    steps: int
    n_cond: int
    kv_cache: bool
    wall_s: float          # 中位墙钟(秒/张)
    wall_std: float
    peak_mem_gb: float     # 峰值显存
    samples: List = field(default_factory=list)  # 生成图,用于质量对比


def parse_args():
    p = argparse.ArgumentParser(description="OminiControl2 KV-Cache 复现基准")
    p.add_argument("--repo-root", default=".", help="OminiControl 仓库根目录")
    p.add_argument("--flux-path", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--lora-repo", required=True, help="LoRA 的 HF repo id 或本地目录")
    p.add_argument("--lora-weight", required=True, help="safetensors 文件名")
    p.add_argument("--adapter-name", default="canny")
    p.add_argument("--condition-type", default="canny",
                   choices=["canny", "depth", "coloring", "deblurring"])
    p.add_argument("--image", default="assets/vase_hq.jpg", help="条件源图")
    p.add_argument("--prompt", default="A beautiful vase on a wooden table.")
    p.add_argument("--steps", default="8,20,28", help="逗号分隔的推理步数列表")
    p.add_argument("--conditions", default="1,2,3", help="逗号分隔的条件分支数列表")
    p.add_argument("--repeats", type=int, default=3, help="每个配置重复次数(取中位)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=3.5)  # dev 蒸馏值,勿动
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no-independent-check", action="store_true",
                   help="跳过 independent_condition 权重检查(冒烟测试用)")
    p.add_argument("--out", default="kvcache_results", help="输出目录")
    return p.parse_args()


def _sync():
    """CUDA 是异步的:计时前后必须 synchronize,否则量到的是"提交时间"而非"执行时间"。"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_one(gen_fn, repeats: int):
    """跑 1 次 warmup(触发 kernel 编译/autotune)+ repeats 次计时,返回中位数与标准差。"""
    import statistics
    import torch

    # warmup —— 第一次调用包含 CUDA 图/kernel 编译开销,计进去会严重高估
    _ = gen_fn()
    _sync()

    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        out = gen_fn()
        _sync()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    med = statistics.median(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return med, std, peak, out


def build_conditions(n_cond, args, Condition, convert_to_condition):
    """
    构造 n_cond 路条件分支。

    WHY 复用同一张 canny 图 + 同一 adapter:本脚本量的是【KV-Cache 省下的计算量】,
    这是纯粹的时间/显存指标,与条件图具体内容无关。因此用同一条件复制 n 份即可,
    既能真实反映多分支的计算规模,又避免为多路准备多套 LoRA。
    若要做多条件【质量】复现,应换成不同任务的条件图 + 各自的 adapter(见 train_multi_condition)。
    """
    from PIL import Image
    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    cond_img = convert_to_condition(args.condition_type, img)
    # position_delta=(0,0):spatial 对齐任务让条件与生成图共享坐标系(见 train/README)
    return [Condition(cond_img, args.adapter_name) for _ in range(n_cond)]


def run():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.repo_root))

    try:
        import torch
        from diffusers import FluxPipeline
        from omini.pipeline.flux_omini import Condition, generate, convert_to_condition
    except ImportError as e:
        log.error("导入失败,确认在仓库根目录且已 pip install -r requirements.txt:%s", e)
        raise

    if not torch.cuda.is_available():
        log.error("未检测到 CUDA GPU;KV-Cache 基准必须在 GPU 上跑。")
        sys.exit(1)

    log.info("GPU: %s | 显存 %.1f GB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))

    # ---- 加载 pipeline + LoRA ----
    # FLUX.1-dev bf16 整体约 36 GB(text_encoder_2 ~10 + transformer ~24 + text_encoder ~2 + vae 0.3)，
    # 单张 24 GB 4090 装不下。多种 offload/device_map 都踩过坑:
    #   - .to("cuda"): OOM
    #   - model_cpu_offload: VAE slow_conv2d_forward 设备不一致
    #   - sequential_cpu_offload: diffusers 0.38 + accelerate 1.14 bug,把全部 module 移到了 meta 设备
    #   - device_map=balanced: accelerate 拆 transformer 跨卡,内部 matmul 跨卡炸
    # 修法:手工 dispatch——
    #   * text_encoder (~1.7) + text_encoder_2 (~9.5) + vae (~0.3) → cuda:0 (~11.5 GB)
    #   * transformer (~24 GB) 整块不拆 → cuda:1
    # 关键:transformer 必须整块放一卡;accelerate 的 balanced 会按权重切 submodule 切坏。
    n_gpu = torch.cuda.device_count()
    log.info("加载 FLUX pipeline: %s (bf16) + 手工 dispatch(0: encoders+vae, 1: transformer)", args.flux_path)
    pipe = FluxPipeline.from_pretrained(
        args.flux_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if n_gpu < 2:
        log.error("本策略需要至少 2 张 GPU,当前只有 %d 张", n_gpu)
        sys.exit(1)
    pipe.text_encoder.to("cuda:0")
    pipe.text_encoder_2.to("cuda:0")
    pipe.vae.to("cuda:0")
    pipe.transformer.to("cuda:1")
    # pipe.device 保持默认(cuda:0,跟 text_encoder 一致):让 token 输入 .to(pipe.device) 去
    # encoders 不撞 device。
    # 跨卡的搬运统一在 omini 的 transformer_forward 入口处理:把所有 list of tensor
    # 一次性搬到 transformer.device(cuda:1)。这覆盖 latents / text_features / img_ids /
    # txt_ids / pooled_projections / timesteps / guidances 全部。
    log.info("dispatch 完成: cuda:0 encoders+vae, cuda:1 transformer")

    from omini.pipeline.flux_omini import transformer_forward as _omni_tx_fwd

    def _tx_fwd_cuda1_wrapper(*args, **kwargs):
        tx = kwargs.get("transformer") or args[0]
        tx_dev = tx.device
        for k in ("image_features", "text_features", "img_ids", "txt_ids",
                 "pooled_projections", "timesteps", "guidances"):
            v = kwargs.get(k)
            if v is None and k == "image_features":
                v = args[1]
            if v is not None and isinstance(v, list):
                moved = []
                for t in v:
                    if isinstance(t, torch.Tensor) and t.device != tx_dev:
                        log.info("  move %s %s %s -> %s", k, tuple(t.shape), t.device, tx_dev)
                        t = t.to(tx_dev)
                    moved.append(t)
                kwargs[k] = moved
        return _omni_tx_fwd(*args, **kwargs)

    # 把 omini 模块级函数换成我们的 wrapper,所有 import 此函数的 omini 路径都会跟着变
    import omini.pipeline.flux_omini as _omni_mod
    _omni_mod.transformer_forward = _tx_fwd_cuda1_wrapper
    # 同时在 omini 内部的 generate() 也会用 module-level transformer_forward
    log.info("已 patch omini.transformer_forward: 所有输入 tensor 搬到 transformer.device")

    # ---- 跨卡补丁:text encoder / vae 输出搬到 transformer 所在卡 ----
    # 原因:encoders 在 cuda:0、transformer 在 cuda:1 时,encoder 的 hidden_states 是 cuda:0,
    # 进了 transformer 的 addmm 算子就跟 cuda:1 的 W 撞设备。
    # monkey-patch 三个 forward,把它们的输出统一搬到 pipe.transformer.device。
    # 只发生在 forward 调用时(每张图一次),代价是一次 12 GB 的 H2D/D2D 拷贝,可接受。
    _tx_dev = pipe.transformer.device

    def _patch_module_output_to_tx(mod, original_forward):
        def wrapped(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            return _move_to(out, _tx_dev)
        return wrapped

    def _move_to(obj, dev):
        # transformers.ModelOutput(如 BaseModelOutputWithPooling)和
        # diffusers.utils.BaseOutput(如 AutoencoderKLOutput、DecoderOutput)都是
        # OrderedDict 子类,没有 .to() 方法,isinstance(obj, dict) 会命中。
        # 这种情况必须重建同类型实例,否则会丢 .last_hidden_state / .latent_dist 等属性。
        from transformers.utils import ModelOutput
        from diffusers.utils.outputs import BaseOutput
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        if isinstance(obj, (ModelOutput, BaseOutput)):
            return obj.__class__(**{k: _move_to(v, dev) for k, v in obj.items()})
        if isinstance(obj, tuple):
            return tuple(_move_to(x, dev) for x in obj)
        if isinstance(obj, list):
            return [_move_to(x, dev) for x in obj]
        if isinstance(obj, dict):
            return {k: _move_to(v, dev) for k, v in obj.items()}
        return obj

    pipe.text_encoder.forward = _patch_module_output_to_tx(pipe.text_encoder, pipe.text_encoder.forward)
    pipe.text_encoder_2.forward = _patch_module_output_to_tx(pipe.text_encoder_2, pipe.text_encoder_2.forward)
    pipe.vae.encode = _patch_module_output_to_tx(pipe.vae, pipe.vae.encode)
    pipe.vae.decode = _patch_module_output_to_tx(pipe.vae, pipe.vae.decode)
    log.info("已 patch encoders/vae forward: 输出搬到 %s", _tx_dev)

    log.info("加载 LoRA: %s :: %s", args.lora_repo, args.lora_weight)
    pipe.load_lora_weights(args.lora_repo, weight_name=args.lora_weight,
                           adapter_name=args.adapter_name)
    pipe.set_adapters([args.adapter_name])

    # ---- 搬运安全网:PEFT 的 load_lora_weights 有时把新加的 LoRA buffer 留在 cuda:0 ----
    # 跨卡场景下需要强行把 transformer 的所有 param + buffer 全部搬到 transformer.device。
    moved = 0
    for module in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        target_dev = module.device
        for p in module.parameters():
            if p.device != target_dev:
                p.data = p.data.to(target_dev); moved += 1
        for b in module.buffers():
            if b.device != target_dev:
                b.data = b.data.to(target_dev); moved += 1
    log.info("device 安全网:搬了 %d 个 param/buffer 到目标 device", moved)

    if not args.no_independent_check:
        log.warning(
            "请确认该 LoRA 用 independent_condition=true 训练。否则 kv_cache=True 生成质量会掉,"
            "速度数字仍有效但质量对比无意义。加 --no-independent-check 可静默此提示。")

    steps_list = [int(x) for x in args.steps.split(",")]
    cond_list = [int(x) for x in args.conditions.split(",")]

    results: List[BenchResult] = []
    for n_cond in cond_list:
        conditions = build_conditions(n_cond, args, Condition, convert_to_condition)
        for steps in steps_list:
            for kv in (False, True):
                def gen_fn(kv=kv, steps=steps, conditions=conditions):
                    g = torch.Generator(device="cuda").manual_seed(args.seed)
                    return generate(
                        pipe,
                        prompt=args.prompt,
                        conditions=conditions,
                        height=args.size,
                        width=args.size,
                        num_inference_steps=steps,
                        guidance_scale=args.guidance_scale,
                        generator=g,
                        kv_cache=kv,
                    ).images[0]

                tag = f"n_cond={n_cond} steps={steps} kv={kv}"
                log.info("测量 %s ...", tag)
                try:
                    med, std, peak, out = time_one(gen_fn, args.repeats)
                except Exception as e:
                    log.error("配置 %s 失败:%s", tag, e)
                    continue
                results.append(BenchResult(steps, n_cond, kv, med, std, peak, [out]))
                log.info("  -> %.3fs ±%.3f | 峰值显存 %.2f GB", med, std, peak)

    _report(results, args)


def _report(results: List[BenchResult], args):
    """打印加速比表格 + 存生成图(供肉眼/后续 CLIP-FID 质量核对)。"""
    os.makedirs(args.out, exist_ok=True)
    # 建立 (n_cond, steps) -> {False: r, True: r} 便于算 speedup
    idx = {}
    for r in results:
        idx.setdefault((r.n_cond, r.steps), {})[r.kv_cache] = r

    print("\n" + "=" * 74)
    print(f"{'n_cond':>6} {'steps':>6} {'baseline(s)':>12} {'kvcache(s)':>12} "
          f"{'speedup':>8} {'mem base':>9} {'mem kv':>8}")
    print("-" * 74)
    for (n_cond, steps), pair in sorted(idx.items()):
        base, kv = pair.get(False), pair.get(True)
        if not (base and kv):
            continue
        speedup = base.wall_s / kv.wall_s if kv.wall_s else float("nan")
        print(f"{n_cond:>6} {steps:>6} {base.wall_s:>12.3f} {kv.wall_s:>12.3f} "
              f"{speedup:>7.2f}x {base.peak_mem_gb:>8.2f}G {kv.peak_mem_gb:>7.2f}G")
        # 存两张图供质量对比:独立训练的 LoRA 下两者应近乎一致
        base.samples[0].save(os.path.join(args.out, f"c{n_cond}_s{steps}_base.png"))
        kv.samples[0].save(os.path.join(args.out, f"c{n_cond}_s{steps}_kv.png"))
    print("=" * 74)
    print(f"生成图已存到 {args.out}/ —— 用独立训练权重时 base 与 kv 两图应几乎相同;")
    print("若明显不同,说明该 LoRA 未按 independent_condition 训练。")


if __name__ == "__main__":
    run()
