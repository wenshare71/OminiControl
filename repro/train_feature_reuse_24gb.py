#!/usr/bin/env python3
"""
阶段二 · feature_reuse LoRA 训练启动器(24 GB 卡 2-GPU 拆分版)
================================================================

为什么需要这个文件:
  上游训练入口(omini/train_flux/trainer.py:67-69)把【整个】FluxPipeline
  `.to(device)` 搬上一张卡(bf16 全量 ≈ 33 GiB),24 GB 的 4090 在加载阶段
  就会 OOM,根本轮不到训练开始。

拆分思路(与推理侧 kvcache_benchmark.py 的布局【相反】,因为训练的主角变了):
  * cuda:0 —— transformer(22.4 GiB,要算梯度,必须和 LoRA/优化器同卡);
  * cuda:1 —— 冻结的 text_encoder / text_encoder_2 / vae(≈11.5 GiB,
    只在 no_grad 下做编码,天然可以放到别的卡);
  * 训练主路径只需把 encode_images / encode_prompt 的【输出】搬回 cuda:0,
    因为 training_step 里 transformer_forward 的其余输入本来就在 cuda:0;
  * 训练中的采样出图(test_function → generate)复用推理侧的 install_tx_bridge:
    此时 generate 的"主场"自动落在 vae 所在的 cuda:1(FluxPipeline.device 取
    组件里第一个 nn.Module = vae),桥把 transformer 输入搬到 cuda:0、输出搬回。

全程零改动上游文件 —— 所有调整都是运行时 monkey-patch,只对本进程生效。

用法(仓库根目录):
  python repro/train_feature_reuse_24gb.py                 # 用默认 feature_reuse.yaml
  OMINI_CONFIG=path/to.yaml python repro/train_feature_reuse_24gb.py
  OMINI_SPLIT=0 python repro/train_feature_reuse_24gb.py   # 大显存卡上强制走原始单卡路径

限制:拆分模式是单进程单"训练卡",与多卡 DDP 不兼容(每个 rank 需要独占 2 张卡,
上游脚本没这个映射)。要 DDP 就得上 ≥40GB 的卡走原始路径。
"""

import logging
import os
import sys
import time

# ---- 必须在 import torch 之前设置:24GB 卡 headroom 只有 ~1 GiB,
#      expandable_segments 能显著缓解显存碎片导致的"明明有余量却 OOM" ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMINI_CONFIG", "./train/config/feature_reuse.yaml")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# 仓库根目录 = 本文件的上上级;保证 `import omini` 与相对路径(runs/, cache/)一致
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "repro"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][train24gb] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train24gb")

import torch  # noqa: E402

TX_DEV = torch.device(os.environ.get("OMINI_TX_DEV", "cuda:0"))   # transformer + LoRA + 优化器
ENC_DEV = torch.device(os.environ.get("OMINI_ENC_DEV", "cuda:1"))  # 冻结的 encoders + vae


def _decide_split() -> bool:
    """OMINI_SPLIT=1/0 显式指定;auto 时按 GPU0 总显存判断(<40GB 装不下 33GB 全量)。"""
    flag = os.environ.get("OMINI_SPLIT", "auto").lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    total0_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total0_gb < 40


def _report_placement(pipe):
    for name in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        mod = getattr(pipe, name)
        dev = next(mod.parameters()).device
        n_gb = sum(p.numel() * p.element_size() for p in mod.parameters()) / 1024**3
        log.info("  %-16s -> %-7s (%.1f GiB)", name, dev, n_gb)
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        log.info("  cuda:%d 空闲 %.1f / %.1f GiB", d, free / 1024**3, total / 1024**3)


def _install_split_patches():
    import omini.train_flux.trainer as trainer_mod
    import omini.train_flux.train_spatial_alignment as tsa
    from diffusers.pipelines import FluxPipeline

    # ---- 1) 拦截整机 .to("cuda"):按上面的布局定向放置,避免 33GB 挤上一张卡 ----
    class _SplitFluxPipeline(FluxPipeline):
        def to(self, *args, **kwargs):
            dev = args[0] if args else kwargs.get("device")
            if dev is not None and str(dev).startswith("cuda"):
                self.transformer.to(TX_DEV)
                for n in ("text_encoder", "text_encoder_2", "vae"):
                    getattr(self, n).to(ENC_DEV)
                log.info("拆分放置完成:")
                _report_placement(self)
                return self
            return super().to(*args, **kwargs)

    trainer_mod.FluxPipeline = _SplitFluxPipeline

    # ---- 2) encode_images 输出搬回训练卡:编码在 vae 卡(pipeline.device=vae 卡)上算,
    #         但产物 x_0 / c_latents 要进 cuda:0 的 transformer_forward ----
    _orig_encode_images = trainer_mod.encode_images

    def _encode_images_bridge(pipeline, images):
        tokens, ids = _orig_encode_images(pipeline, images)
        return tokens.to(TX_DEV), ids.to(TX_DEV)

    trainer_mod.encode_images = _encode_images_bridge

    # ---- 3) encode_prompt 输出搬回训练卡(实例级 wrap,在模型 init 完成后挂)。
    #         调用方已传 device=pipe.device(= encoder 卡),内部自洽;只需搬输出。----
    _orig_model_init = trainer_mod.OminiModel.__init__

    def _init_then_wrap(self, *a, **kw):
        _orig_model_init(self, *a, **kw)
        pipe = self.flux_pipe
        _orig_ep = pipe.encode_prompt

        def _encode_prompt_bridge(*aa, **kk):
            out = _orig_ep(*aa, **kk)
            return tuple(
                x.to(TX_DEV) if torch.is_tensor(x) else x for x in out
            )

        pipe.encode_prompt = _encode_prompt_bridge

    trainer_mod.OminiModel.__init__ = _init_then_wrap

    # ---- 4) 钉死 Lightning 只用训练卡:2 张卡可见时 L.Trainer(devices="auto")
    #         会自作主张开 2 进程 DDP,把 33GB 模型加载两遍 → 必炸。
    #         trainer.py 运行期只用到 L.Trainer,其余属性透传真 lightning。----
    import lightning as _L_real

    class _LightningProxy:
        def __getattr__(self, name):
            return getattr(_L_real, name)

        def Trainer(self, **kw):
            kw["accelerator"] = "gpu"
            kw["devices"] = [TX_DEV.index]
            return _L_real.Trainer(**kw)

    trainer_mod.L = _LightningProxy()

    # ---- 5) 采样出图路径:复用推理侧跨卡桥(generate 主场在 vae 卡,
    #         transformer 输入搬到 cuda:0、输出搬回)----
    from kvcache_benchmark import install_tx_bridge
    install_tx_bridge()

    # ---- 6) test_function 里 torch.Generator(device=model.device) 会造 cuda:0 的
    #         generator,而采样 latents 在 cuda:1 → torch.randn 报 generator 设备
    #         不匹配。cpu generator 对任何目标卡都合法(diffusers randn_tensor 先在
    #         cpu 生成再搬),所以只 shim Generator,其余属性透传真 torch。----
    class _TorchCPUGenShim:
        def __getattr__(self, name):
            return getattr(torch, name)

        def Generator(self, device=None):
            return torch.Generator(device="cpu")

    tsa.torch = _TorchCPUGenShim()

    # ---- 7) 采样失败不许打断训练(采样只是监控手段,不是训练正确性的一部分)----
    _orig_test = tsa.test_function

    def _safe_test(model, save_path, file_name):
        try:
            _orig_test(model, save_path, file_name)
        except Exception as e:  # noqa: BLE001 —— 兜底监控路径,任何异常都只告警
            log.warning("采样出图失败(训练继续,见 TROUBLESHOOTING §7):%r", e)

    tsa.test_function = _safe_test
    log.info("2-GPU 拆分补丁已全部安装(transformer=%s, encoders/vae=%s)", TX_DEV, ENC_DEV)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 GPU。")
    split = _decide_split()
    if split and torch.cuda.device_count() < 2:
        raise RuntimeError(
            "拆分模式需要 >=2 张 GPU(transformer 22.4GiB + encoders 11.5GiB 单张 24GB "
            "放不下)。请检查 CUDA_VISIBLE_DEVICES 至少暴露 2 张卡。")

    log.info("配置文件:%s", os.environ["OMINI_CONFIG"])
    log.info("模式:%s", "2-GPU 拆分(24GB 卡)" if split else "原始单卡(大显存)")

    if split:
        _install_split_patches()

    import omini.train_flux.train_spatial_alignment as tsa
    t0 = time.time()
    try:
        tsa.main()
    except torch.cuda.OutOfMemoryError:
        log.error(
            "训练 OOM(已运行 %.0fs)。24GB 卡 headroom 只有 ~1GiB,按顺序尝试:"
            "1) 确认 cuda:0 完全空闲(nvidia-smi);"
            "2) yaml 里 condition_size/target_size 降到 [256,256];"
            "3) 换 yaml 注释里的精简版 target_modules;"
            "详见 repro/TROUBLESHOOTING.md §7。", time.time() - t0)
        raise


if __name__ == "__main__":
    main()
