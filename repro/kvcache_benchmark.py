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

多卡 dispatch 说明(24 GB 卡必读):
  FLUX.1-dev bf16 全量 ≈ 33 GB(text_encoder 1.7 + text_encoder_2 9.5 + transformer 22.4
  + vae 0.3,GiB 口径),单张 24 GB 4090 装不下。经过远程机器 11 次失败排查
  (见 repro/REPRODUCE_FEATURE_REUSE_STATUS.md / repro/TROUBLESHOOTING.md),
  最终采用"单拦截点"方案:
    * cuda:0 是主场:encoders + vae + latents + scheduler 全在 cuda:0,
      generate() 内部逻辑完全不感知多卡;
    * cuda:1 只放 transformer(整块不拆,22.4 GiB 恰好放得下一张 4090);
    * 在 omini.transformer_forward 入口装"跨卡桥"(install_tx_bridge):
      白名单输入搬进 cuda:1,输出搬回 cuda:0。
  这消灭了此前所有 addmm / index_select / scheduler.step / vae.decode 跨卡报错,
  且不需要 patch encoder/vae 的输出(旧方案的补丁已删除——它们把张量推向
  cuda:1,反而是制造设备混乱的来源)。

前置条件:
  1. 在仓库根目录运行(保证 `import omini` 可用),或用 --repo-root 指定。
  2. LoRA 必须用 `independent_condition: true` 训练(train/script/train_feature_reuse.sh)。
     用未独立训练的权重也能跑、速度数字有效,但质量对比无意义(会掉),脚本会告警。

用法示例:
  python repro/kvcache_benchmark.py \
      --lora-repo runs/feature_reuse_canny/ckpt --lora-weight pytorch_lora_weights.safetensors \
      --adapter-name canny --condition-type canny \
      --steps 8,20,28 --conditions 1,2,3 --repeats 3 --image assets/vase_hq.jpg

  # 冒烟测试(拿现成 v1 权重先验证管线,质量不作数):
  python repro/kvcache_benchmark.py --lora-repo Yuanshi/OminiControl \
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


# ---- 关键 workaround:torch 2.8 + diffusers 0.38 在某些 GPU 上跑 VAE 的
#      conv2d 时 cuDNN 报 CUDNN_STATUS_NOT_INITIALIZED(同 Stage 2 训练时的
#      同一个 bug,只是这次 VAE 在 cuda:0 而非 cuda:1)。让 F.conv2d 走
#      原生 eager 路径即可解决,FLUX 推理不需要 cuDNN 加速。
#      必须在首次 cuda op 之前设置。 ----
import torch  # noqa: E402
torch.backends.cudnn.enabled = False
log.info("cudnn disabled (workaround for VAE CUDNN_STATUS_NOT_INITIALIZED)")


@dataclass
class BenchResult:
    steps: int
    n_cond: int
    kv_cache: bool
    wall_s: float          # 中位墙钟(秒/张)
    wall_std: float
    peak_mem_gb: float     # 峰值显存(所有 GPU 中的最大值)
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
    p.add_argument("--dispatch", default="auto", choices=["auto", "single", "2gpu"],
                   help="auto: GPU0 显存 >=40GB 用 single,否则 2gpu")
    p.add_argument("--gpu0", type=int, default=0,
                   help="2gpu dispatch 主场 GPU(encoders+vae+latents)。"
                        "若 cuda:0 被 defunct context 死锁可改用其它空卡")
    p.add_argument("--gpu1", type=int, default=1,
                   help="2gpu dispatch 变压器卡 GPU(transformer 整块)。"
                        "若 cuda:1 被 defunct context 死锁可改用其它空卡")
    p.add_argument("--no-independent-check", action="store_true",
                   help="跳过 independent_condition 权重检查(冒烟测试用)")
    p.add_argument("--out", default="repro/kvcache_results", help="输出目录")
    return p.parse_args()


# ======================================================================
# 跨卡工具(可被 notebook 直接 import 复用)
# ======================================================================

def _sync_all():
    """CUDA 是异步的:计时前后必须 synchronize,否则量到的是"提交时间"而非"执行时间"。
    多卡 dispatch 下要对【每张】用到的卡 sync,不带参的 synchronize 只同步当前卡。"""
    import torch
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)


def _move(obj, dev):
    """递归把 tensor / list / tuple 里的 tensor 搬到 dev;其它类型原样返回。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, list):
        return [_move(x, dev) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_move(x, dev) for x in obj)
    return obj


# 只搬这些 kwargs。WHY 白名单而不是"递归全搬":
#   cache_storage 是靠【list 对象身份】在多步之间共享的 KV 缓存容器
#   (write 时 append、read 时读同一个 list)。递归重建会把 write 写进副本、
#   read 读到空缓存 —— KV-Cache 直接失效且很难排查。
_TX_BRIDGE_KEYS = (
    "image_features", "text_features", "img_ids", "txt_ids",
    "pooled_projections", "timesteps", "guidances", "group_mask",
)


def install_tx_bridge():
    """给 omini.transformer_forward 装"跨卡桥":

    输入(白名单)搬到 transformer 所在卡,输出搬回主 latent 所在卡("主场",cuda:0)。
    这样 generate() 里的 scheduler.step / vae.decode / 随机数全程留在主场,
    整条管线只有这一个跨卡拦截点。幂等:重复调用(notebook 重跑 cell)不会套两层。
    """
    import torch
    import omini.pipeline.flux_omini as omini_mod

    if getattr(omini_mod.transformer_forward, "_is_tx_bridge", False):
        log.info("tx bridge 已安装,跳过")
        return
    orig = omini_mod.transformer_forward

    def bridge(transformer, *args, **kwargs):
        tx_dev = next(transformer.parameters()).device
        feats = kwargs.get("image_features") or (args[0] if args else None)
        home_dev = feats[0].device if feats else tx_dev
        # generate() 除 transformer 外全用 kwargs;万一有 positional 也一并搬
        args = tuple(_move(a, tx_dev) for a in args)
        for k in _TX_BRIDGE_KEYS:
            if k in kwargs:
                kwargs[k] = _move(kwargs[k], tx_dev)
        out = orig(transformer, *args, **kwargs)
        # 输出搬回主场:scheduler.step 需要 noise_pred 与 latents 同卡
        return _move(out, home_dev)

    bridge._is_tx_bridge = True
    omini_mod.transformer_forward = bridge
    log.info("已安装 tx bridge:输入 -> transformer 卡,输出 -> 主场卡")


def sweep_devices(pipe):
    """把每个子模块的所有 param/buffer 强行搬到该模块自己的 device。

    WHY:PEFT 的 load_lora_weights/set_adapters 在多卡 dispatch 下可能把新建的
    LoRA 权重放在 cuda:0(它以 text_encoder 所在卡当"模型 device"),导致
    transformer 内部 forward 撞设备(远程机器失败 #11)。必须在 set_adapters
    【之后】调用。"""
    moved = 0
    for module in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae):
        target = next(module.parameters()).device
        for p in module.parameters():
            if p.device != target:
                p.data = p.data.to(target)
                moved += 1
        for b in module.buffers():
            if b.device != target:
                b.data = b.data.to(target)
                moved += 1
    log.info("device 安全网:搬了 %d 个 param/buffer", moved)
    return moved


def load_pipeline(flux_path="black-forest-labs/FLUX.1-dev", dispatch="auto",
                  gpu0: int = 0, gpu1: int = 1):
    """加载 FLUX pipeline,按显存自动选择放置策略。

    single: 整个 pipeline .to("cuda")(需要单卡 >= 40 GB)。
    2gpu:   encoders+vae -> cuda:{gpu0},transformer 整块 -> cuda:{gpu1},并安装 tx bridge。
            关键:transformer 必须整块放一张卡;accelerate 的 device_map="balanced"
            会按权重把它切到多张卡,内部 matmul 直接跨卡报错(失败 #6)。

    Args:
        gpu0: 2gpu dispatch 主场 GPU(devices for text_encoder/text_encoder_2/vae/latents)。
              默认 0。当 cuda:0 被 defunct context 死锁时可改成其它空卡。
        gpu1: 2gpu dispatch 变压器卡(device for transformer 整块)。默认 1。
    """
    import torch
    from diffusers import FluxPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU;KV-Cache 基准必须在 GPU 上跑。")

    n_gpu = torch.cuda.device_count()
    gpu0_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    for d in range(n_gpu):
        log.info("GPU %d: %s | %.1f GB", d, torch.cuda.get_device_name(d),
                 torch.cuda.get_device_properties(d).total_memory / (1024 ** 3))

    if dispatch == "auto":
        dispatch = "single" if gpu0_gb >= 40 else "2gpu"
        log.info("dispatch=auto -> 选择 %s(GPU0 %.0f GB)", dispatch, gpu0_gb)

    if dispatch == "single":
        log.info("加载 FLUX pipeline: %s (bf16, 单卡)", flux_path)
        pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to("cuda")
        return pipe

    if dispatch != "2gpu":
        raise ValueError(f"未知 dispatch: {dispatch}")
    if n_gpu < 2:
        raise RuntimeError(
            f"2gpu 策略需要至少 2 张 GPU(当前 {n_gpu} 张)。"
            "单张小卡请看 repro/TROUBLESHOOTING.md 的量化退路方案。")
    if gpu0 == gpu1:
        raise ValueError(f"gpu0 == gpu1 == {gpu0},2gpu dispatch 必须用两张不同的卡")

    log.info("加载 FLUX pipeline: %s (bf16, 2gpu 手工 dispatch)", flux_path)
    pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(f"cuda:{gpu0}")
    pipe.text_encoder_2.to(f"cuda:{gpu0}")
    pipe.vae.to(f"cuda:{gpu0}")
    pipe.transformer.to(f"cuda:{gpu1}")
    log.info("dispatch 完成: cuda:%d = encoders+vae+latents(主场), cuda:%d = transformer",
             gpu0, gpu1)

    # generate() 用 pipe._execution_device 创建 latents/timesteps 等,必须是主场。
    # diffusers 取 components 里第一个 nn.Module 的 device,正常应命中主场的模块;
    # 若未来 diffusers 改了顺序命中 transformer,这里立刻报错而不是深处炸 addmm。
    exec_dev = str(pipe._execution_device)
    if exec_dev != f"cuda:{gpu0}":
        raise RuntimeError(
            f"pipe._execution_device 是 {exec_dev},预期 cuda:{gpu0}。"
            "diffusers 版本行为变化,见 repro/TROUBLESHOOTING.md §设备排查。")

    install_tx_bridge()
    return pipe


def attach_lora(pipe, lora_repo, lora_weight, adapter_name):
    """加载并激活 LoRA,随后做 device sweep(多卡下必须,单卡下无副作用)。"""
    log.info("加载 LoRA: %s :: %s", lora_repo, lora_weight)
    pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name=adapter_name)
    pipe.set_adapters([adapter_name])
    sweep_devices(pipe)
    return pipe


def build_conditions(n_cond, image_path, condition_type, adapter_name, size=512):
    """
    构造 n_cond 路条件分支。

    WHY 复用同一张 canny 图 + 同一 adapter:本脚本量的是【KV-Cache 省下的计算量】,
    这是纯粹的时间/显存指标,与条件图具体内容无关。因此用同一条件复制 n 份即可,
    既能真实反映多分支的计算规模,又避免为多路准备多套 LoRA。
    若要做多条件【质量】复现,应换成不同任务的条件图 + 各自的 adapter(见 train_multi_condition)。
    """
    from PIL import Image
    from omini.pipeline.flux_omini import Condition, convert_to_condition
    img = Image.open(image_path).convert("RGB").resize((size, size))
    cond_img = convert_to_condition(condition_type, img)
    return [Condition(cond_img, adapter_name) for _ in range(n_cond)]


def time_one(gen_fn, repeats: int):
    """跑 1 次 warmup(触发 kernel 编译/autotune)+ repeats 次计时,返回中位数与标准差。"""
    import statistics
    import torch

    # warmup —— 第一次调用包含 CUDA 图/kernel 编译开销,计进去会严重高估
    _ = gen_fn()
    _sync_all()

    times = []
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    for _ in range(repeats):
        _sync_all()
        t0 = time.perf_counter()
        out = gen_fn()
        _sync_all()
        times.append(time.perf_counter() - t0)
    peak = max(torch.cuda.max_memory_allocated(d)
               for d in range(torch.cuda.device_count())) / (1024 ** 3)
    med = statistics.median(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return med, std, peak, out


# ======================================================================
# CLI 主流程
# ======================================================================

def run():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.repo_root))

    try:
        import torch
        from omini.pipeline.flux_omini import generate
    except ImportError as e:
        log.error("导入失败,确认在仓库根目录且已 pip install -r requirements.txt:%s", e)
        raise

    pipe = load_pipeline(args.flux_path, dispatch=args.dispatch,
                         gpu0=args.gpu0, gpu1=args.gpu1)
    attach_lora(pipe, args.lora_repo, args.lora_weight, args.adapter_name)

    if not args.no_independent_check:
        log.warning(
            "请确认该 LoRA 用 independent_condition=true 训练。否则 kv_cache=True 生成质量会掉,"
            "速度数字仍有效但质量对比无意义。加 --no-independent-check 可静默此提示。")

    steps_list = [int(x) for x in args.steps.split(",")]
    cond_list = [int(x) for x in args.conditions.split(",")]

    results: List[BenchResult] = []
    for n_cond in cond_list:
        conditions = build_conditions(n_cond, args.image, args.condition_type,
                                      args.adapter_name, args.size)
        for steps in steps_list:
            for kv in (False, True):
                def gen_fn(kv=kv, steps=steps, conditions=conditions):
                    # generator 在主场 cuda:0:latents 由它产出,必须与 scheduler 同卡
                    g = torch.Generator(device="cuda:0").manual_seed(args.seed)
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

    _report(results, args.out)


def _report(results: List[BenchResult], out_dir):
    """打印加速比表格 + 存生成图(供肉眼/后续 CLIP-FID 质量核对)。"""
    os.makedirs(out_dir, exist_ok=True)
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
        base.samples[0].save(os.path.join(out_dir, f"c{n_cond}_s{steps}_base.png"))
        kv.samples[0].save(os.path.join(out_dir, f"c{n_cond}_s{steps}_kv.png"))
    print("=" * 74)
    print(f"生成图已存到 {out_dir}/ —— 用独立训练权重时 base 与 kv 两图应几乎相同;")
    print("若明显不同,说明该 LoRA 未按 independent_condition 训练。")


if __name__ == "__main__":
    run()
