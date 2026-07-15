# OminiControl2 · 特征复用复现 · 环境与排错记录

> 🔴 **2026-07-15:本文件 §1(机器环境)已全部过期,不要照做。**
> 机器重启后 `/root` 回滚,**conda 和 `/root/miniconda3/envs/omini` 已不存在**;
> 现在用 uv + `/root/omini-venv`,`HF_HOME` 也已迁到 Ceph 持久盘。
> **当前环境看 [`../MACHINE_STATUS.md`](../MACHINE_STATUS.md),重建步骤看 [`ENV_REBUILD.md`](ENV_REBUILD.md)。**
> §2 之后的复现记录与失败分析仍然有效。

> **2026-07-09 更新:失败 #11 及后续隐患已在本地修复。**
> 修复后的方案(2gpu dispatch + transformer_forward 跨卡桥 + LoRA device sweep)已落在
> `repro/kvcache_benchmark.py`;一键冒烟测试用 `repro/stage1_smoke_test.ipynb`;
> 报错分诊看 `repro/TROUBLESHOOTING.md`。本文件保留为原始失败记录,不再更新。

> 本文件记录阶段一(冒烟测试)的环境配置、踩过的坑、当前状态。
> 用于在另一台机器或后续会话中复现 / 接手。

---

## 1. 机器环境

### 1.1 硬件
| 项目 | 值 |
|---|---|
| 主机 | `aiplatform-bjy-ge47-388.idchb1az2.hb1.kwaidc.com` (内部 IDC) |
| CPU | 256 核 |
| 内存 | 1.0 TiB |
| 磁盘 | 7.0 T(已用 587 G,可用 6.5 T) |
| GPU | **8 × NVIDIA GeForce RTX 4090(每张 24 GB)**,driver 535.54.03,CUDA 12.2 |
| `nvcc` | 未装(只跑 inference 不需要) |

### 1.2 OS
- Linux 4.18.0-2.4.3.3.kwai.x86_64(KwaiOS / 类 RHEL 内核)
- 初始 Python:系统级 **3.8.10**(不符合项目要求)
- `conda` 未装;pip 24.0,pip index = 内部镜像 `pypi.corp.kuaishou.com`(Kuaishou 内部 PyPI)

### 1.3 创建的 conda 环境
- 位置:`/root/miniconda3/envs/omini`
- Python:**3.12.13**
- 安装方式:`bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 -u`,然后 `conda tos accept` 同意 main / r 通道
- 激活方式:`source train/setup_env.sh`(已写入仓库)

### 1.4 安装的关键包版本
| 包 | 版本 | 备注 |
|---|---|---|
| `torch` | **2.8.0+cu128** | 官方 `download.pytorch.org/whl/cu128` 装 |
| `diffusers` | **0.38.0** | 项目硬要求(其它版本会破 KV-Cache 的 `cache_idx` 设置) |
| `transformers` | **4.57.6** | 满足 `>=4.55,<5` |
| `huggingface_hub` | **0.36.2** | 满足 `<1.0`(transformers 仍 pin) |
| `peft` | 0.19.1 | |
| `accelerate` | 1.14.0 | ⚠️ 跟 diffusers 0.38 + `enable_sequential_cpu_offload` **不兼容**(见下) |
| `numpy` | 2.5.1 | |
| `Pillow` | 12.3.0 | |
| `opencv-python` | 5.0.0.93 | |

### 1.5 模型下载(已缓存到本地)
| 模型 | 大小 | 位置 |
|---|---|---|
| `black-forest-labs/FLUX.1-dev` | ~36 GB(bf16 全量) | `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/` |
| `Yuanshi/OminiControl::experimental/canny.safetensors` | ~1 GB | `~/.cache/huggingface/hub/models--Yuanshi--OminiControl/` |

### 1.6 HF 鉴权
- `huggingface-cli login --token <token>` 已写 `/root/.cache/huggingface/token`
- `whoami: wenshare`
- `FLUX.1-dev` gated=auto,访问已开门
- ⚠️ **安全提示**:token 明文存在机器上,本机非受控环境时建议撤销

### 1.7 git 状态
- 本仓库指向 fork:`https://github.com/wenshare71/OminiControl.git`
- 分支 `main`,HEAD `65d929e Fix multi-GPU training and align training stack with diffusers 0.38`

---

## 2. 复现目标(摘自 `REPRODUCE_FEATURE_REUSE.md`)

阶段一(本文件范围):用现成 v1 canny LoRA 跑通管线,验证 FLUX 加载 / LoRA 挂载 / `generate()` 调用链。
成功标准:脚本正常输出 baseline vs kvcache 的时间对比,无异常退出。

原始命令(在仓库根):
```bash
python repro/kvcache_benchmark.py \
  --lora-repo Yuanshi/OminiControl \
  --lora-weight experimental/canny.safetensors \
  --adapter-name canny --condition-type canny \
  --steps 8 --conditions 1 --repeats 2 \
  --no-independent-check
```

---

## 3. 失败链路(7 次失败 + 1 次部分解决)

按时间顺序,每次失败的根因 + 修法。

### 3.1 失败 #1 — `ImportError: cannot import name 'FluxPipeline'`
- 现象:系统 Python 3.8 + diffusers 0.29.2 太旧,`from diffusers.pipelines import FluxPipeline` 不存在
- 根因:仓库要求 Python 3.12 + diffusers 0.38,但机器上是 3.8.10 + 0.29.2
- 修法:见 §1,装 conda / 新环境 / 装 torch 2.8 + 项目 requirements

### 3.2 失败 #2 — `CUDA out of memory. Tried to allocate 90.00 MiB`
- 现象:加载 FLUX pipeline 报 OOM,GPU 0 上 process 占了 23.57 GB,只剩 74 MB
- 根因:FLUX.1-dev bf16 全量约 **36 GB**(text_encoder 1.7 + text_encoder_2 9.5 + transformer 23.8 + vae 0.3 + 调度器/tokenizer),24 GB 装不下
- 后续:放弃"全量单卡"

### 3.3 失败 #3 — `slow_conv2d_forward` 设备不一致
- 策略:diffusers `pipe.enable_model_cpu_offload()`
- 现象:`Expected all tensors to be on the same device, but got weight is on cuda:0, different from other tensors on cpu`
- 根因:CPU offload 钩子把 VAE 的 conv 权重搬到 GPU 但输入 latent 还在 CPU,跨设备
- 后续:换 `sequential_cpu_offload`

### 3.4 失败 #4 — `Cannot copy out of meta tensor; no data!`
- 策略:diffusers `pipe.enable_sequential_cpu_offload()`
- 现象:首次 forward 报"meta tensor"
- **根因**:`enable_sequential_cpu_offload` 在 diffusers 0.38 + accelerate 1.14 下,把全部 sub-module 移到了 **meta 设备**(不是 CPU)——这是 `accelerate.hooks` 的兼容 bug
  - 验证:用 `from_pretrained(..., low_cpu_mem_usage=False)` 加载后,模块在 CPU;调 `enable_sequential_cpu_offload` 后,**所有**模块都跳到 meta
  - 跟 `low_cpu_mem_usage` 默认值无关,`low_cpu_mem_usage=False` 也复现
- 后续:换多卡 `device_map`

### 3.5 失败 #5 — `NotImplementedError: auto not supported. Supported strategies are: balanced, cuda, cpu`
- 策略:accelerate `device_map="auto"`
- 根因:diffusers 0.38 不支持 `device_map="auto"`,仅 `balanced / cuda / cpu`
- 后续:换 `device_map="balanced"`

### 3.6 失败 #6 — `addmm` 跨卡
- 策略:accelerate `device_map="balanced"`(8 卡)
- 现象:`mat1 is on cuda:0, different from other tensors on cuda:1`
- 根因:accelerate 的 `balanced` 按权重**拆分** transformer 到多张卡(transformer 内部 matmul 跨卡)
- 后续:换手工 dispatch

### 3.7 失败 #7 — 手工 2 卡 dispatch,addmm 跨卡(text encoder 输出去向)
- 策略:`text_encoder/text_encoder_2/vae` → cuda:0,`transformer` → cuda:1
- 现象:transformer forward 报 `addmm device mismatch` (mat1 on cuda:0, mat2 on cuda:1)
- 根因:FLUX.1-dev 中 text encoder 输出的 hidden_states 传进 transformer 时,encoder 内部权重在 cuda:0、输入在 cuda:0,但 transformer 在 cuda:1 → addmm(W_cuda1, h_cuda0) 撞设备
- 修法:monkey-patch `text_encoder.forward` / `vae.encode` / `vae.decode`,把输出搬到 transformer.device
  - 第一次修失败:`'dict' object has no attribute 'pooler_output'` —— `BaseModelOutputWithPooling` 是 `OrderedDict` 子类,**没有 `.to()`**,被当成普通 dict 递归时丢属性
  - 修法:用 `transformers.utils.ModelOutput` 显式重建同类型
  - 第二次修失败:`'dict' object has no attribute 'latent_dist'` —— diffusers 的 `AutoencoderKLOutput` 基类是 `diffusers.utils.BaseOutput`,**不是** `transformers.ModelOutput`
  - 修法:同时 import `ModelOutput` 和 `BaseOutput` 并处理

### 3.8 失败 #8 — `index_select` 设备不一致
- 策略:类层面把 `pipe.device` 改成 cuda:1
- 现象:`index is on cuda:1, different from other tensors on cuda:0` —— 在 text encoder **内部**
- 根因:把 `pipe.device` 改 cuda:1 → `encode_prompt(..., device=cuda:1)` → `tokenizer.__call__` 输出的 `input_ids` 跑到 cuda:1 → `text_encoder(input_ids_cuda1)` 在 cuda:0 → embedding `index_select(emb_cuda0, idx_cuda1)` 撞设备
- 真正的矛盾:encoders 在 cuda:0 需要 `device == cuda:0` 的输入;transformer 在 cuda:1 需要 `device == cuda:1` 的输入。`pipe.device` 一个值两边抢
- 修法:把 `pipe.device` **保持默认 cuda:0**(给 encoders),单独覆盖 `_execution_device = cuda:1`(给 loop 里的 `c_timesteps` / `c_guidances` / `guidance` / `latents` / `latent_image_ids`)

### 3.9 失败 #9 — `prepare_latents() got multiple values for argument 'device'`
- 现象:覆盖 `prepare_latents` 把 `device` 塞到 kwargs,跟原函数 positional 的 `device` 冲突
- 修法:替换 `args[5]` 而非 kwargs

### 3.10 失败 #10 — `addmm` 跨卡(latents)
- 现象:在 cuda:1 的 transformer 报 `mat1 is on cuda:0`
- 根因:虽然 `_execution_device == cuda:1`,但 `pipe.device` 仍用第一个 nn.Module 的 device(此处是 `text_encoder == cuda:0`),其它路径(如 `transformer_kwargs` / `pipe._pack_latents` 之类)还会用 `pipe.device` 创建张量
- 修法:取消 `pipe.device` / `_execution_device` 类级覆盖,改在 **omini 的 `transformer_forward` 入口**包一层 wrapper,把 7 个 List[Tensor] 参数(image_features / text_features / img_ids / txt_ids / pooled_projections / timesteps / guidances)统一搬到 `transformer.device`

### 3.11 失败 #11 — `found at least two devices, cuda:0 and cuda:1`(transformer 内部)
- 现象:wrapper 已经把所有 List[Tensor] 搬到 cuda:1,但 transformer **内部**仍出现 cuda:0 张量
- 未完成定位:最可能是 **PEFT 的 `load_lora_weights` 把 LoRA 权重/buffer 留在了 cuda:0**(cuda:0 是 text_encoder 所在卡,PEFT 可能用了它的 device 作"模型 device")
- 已尝试的修法:加载 LoRA 后遍历 `pipe.transformer.parameters()` 和 `.buffers()`,把所有非 `transformer.device` 的张量 `.data = .to(transformer.device)`(**此修复未提交,因为被打断**)
- 需要:在 `pipe.set_adapters([...])` **之后**加上 sweep,见 `kvcache_benchmark.py` 修复点建议(§5)

---

## 4. 当前脚本状态(`repro/kvcache_benchmark.py`)

已实施且确认有效的部分(失败 #1-10 累积的):
1. 手工 2 卡 dispatch(text_encoder/text_encoder_2/vae → cuda:0;transformer → cuda:1)
2. monkey-patch `text_encoder.forward` / `text_encoder_2.forward` / `vae.encode` / `vae.decode`,输出搬到 transformer.device,正确处理 `ModelOutput` 和 `BaseOutput`
3. **类层面覆盖** `_execution_device` (但已回退,见下)
4. monkey-patch `pipe.prepare_latents`,把 device 替换为 cuda:1(已回退)
5. **类层面覆盖** `pipe.device` 为 cuda:1(已回退)
6. **入口 wrapper**:用 `omini.pipeline.flux_omini.transformer_forward` 的 wrapper 替换原函数,把 7 个 List[Tensor] 参数搬 transformer.device

**当前文件状态**:
- §1-7 在 `run()` 里
- Pylance 报告 1 个未解析导入(`from transformers.utils import ModelOutput` → 实际应该是 `transformers.utils.generic.ModelOutput`)+ 1 个 .images 类型告警(非阻塞)
- 7 个 List[Tensor] wrapper 是**当前**生效的方案;类级 device property 覆盖已回退

---

## 5. 推荐修复方向(留给后续接手者)

### 5.1 最小修复 — 1 行
**最可能的成功点**:在 `pipe.set_adapters([args.adapter_name])` **之后**插入"全量 sweep"——
```python
moved = 0
for module in [pipe.transformer]:
    target = module.device
    for p in module.parameters():
        if p.device != target:
            p.data = p.data.to(target); moved += 1
    for b in module.buffers():
        if b.device != target:
            b.data = b.data.to(target); moved += 1
log.info("device sweep: moved %d tensors", moved)
```
这个改动**未提交**(被打断)。预期它能解决失败 #11——PEFT 加的 LoRA buffer 留在 cuda:0。

### 5.2 退路方案(若 5.1 不行)
1. **降 accelerate**:试 `pip install accelerate==0.34.0`(老版本可能没有 §3.4 那个 meta-tensor bug),然后 `enable_sequential_cpu_offload()` 也许直接能跑
2. **换主仓 fp8 权重**:用 `black-forest-labs/FLUX.1-dev-fp8`(若存在,约 12 GB,单卡 24 GB 装得下)或社区 `flux1-dev-fp8.safetensors`
3. **bitsandbytes 4-bit 量化 transformer**:约 6 GB,放单卡

### 5.3 长期建议
- 当前仓库 24 GB 单卡路线不可行(36 GB 模型 vs 24 GB 显存);任何后续用 FLUX 的工作,都建议在脚本里显式声明 dispatch 策略,而不是依赖 diffusers 默认行为
- 阶段二训练(`feature_reuse.yaml`)本身有 batch_size=1 + grad-ckpt,峰值更低,可能反而能在单卡上跑;阶段一冒烟不需要在主训脚本里做同样修复
- `accelerate.hooks` 在 diffusers 0.38 下的行为有 bug,后续升级 diffusers 时要重新验证

---

## 6. 已被杀进程 / 释放资源

- 所有 `python repro/kvcache_benchmark.py` 子进程(7 次失败 + 1 次成功加载):已 `pkill -9` 清掉
- GPU 0-7 显存均 2 MiB(空闲)
- 保留运行:code-server IDE 进程、Pylance LSP(用户 IDE 用,**未杀**)

---

## 7. 已落到仓库的改动

| 文件 | 状态 |
|---|---|
| `repro/REPRODUCE_FEATURE_REUSE.md` | 原始复现计划,**未动** |
| `repro/kvcache_benchmark.py` | **改了**——手工 dispatch + encoders/vae forward patch + transformer_forward 入口 wrapper(7 个 List[Tensor] 搬运)+ `low_cpu_mem_usage=False` |
| `repro/REPRODUCE_FEATURE_REUSE_STATUS.md` | **新增**——本文档 |
| `train/setup_env.sh` | **新增**——激活 omini conda env 的便捷脚本 |
| `.vscode/settings.json` | **新增**——Pylance 指向 omini env |

git 状态:以下文件待 commit
```
M  repro/kvcache_benchmark.py
?? repro/REPRODUCE_FEATURE_REUSE_STATUS.md
?? train/setup_env.sh
?? .vscode/settings.json
```
