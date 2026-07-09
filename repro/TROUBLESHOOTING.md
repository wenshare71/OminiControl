# OminiControl2 · KV-Cache 复现 · 修复手册

> 配套文件:`repro/stage1_smoke_test.ipynb`(一键冒烟测试)、`repro/stage2_train.ipynb`(一键训练)、
> `repro/stage3_benchmark.ipynb`(一键正式测量)、`repro/kvcache_benchmark.py`(CLI 基准)、
> `repro/train_feature_reuse_24gb.py`(24GB 卡 2-GPU 训练启动器)、
> `repro/REPRODUCE_FEATURE_REUSE_STATUS.md`(远程机器 11 次失败的原始记录)。
> 本手册 = 失败记录的**结论版** + **新报错的分诊方法** + **退路方案**。

---

## 快速分诊表(拿报错关键词查)

| 报错关键词 | 章节 |
|---|---|
| `cannot import name 'FluxPipeline'` / `ModuleNotFoundError` | §1.1 |
| `torch.cuda.is_available() == False` / CUDA error | §1.2 |
| `cache_idx` / KV-Cache 不加速 | §1.3 |
| `CUDA out of memory` | §2 |
| `Expected all tensors to be on the same device` / `addmm` / `index_select` / `found at least two devices` | §3 + §4 |
| `Cannot copy out of meta tensor` | §3.2 |
| `'dict' object has no attribute 'pooler_output' / 'latent_dist'` | §3.3 |
| `401` / `GatedRepoError` | §6.1 |
| adapter 名字冲突 / 多 LoRA 不生效 | §6.2 |
| 生成图与条件无关 / 质量差 | §6.3 |
| 计时抖动大 / 加速比忽高忽低 | §6.4 |
| **训练**加载/第一步 OOM | §7.1 |
| Lightning 起了 2 个进程 / 模型加载两遍 | §7.2 |
| `采样出图失败(训练继续)` | §7.3 |
| 数据下载卡住 / `cache/t2i2m` | §7.4 |
| loss 不降 / 阶段三像素 MAE 偏大 | §7.5 |

---

## 1. 环境 / 导入类

### 1.1 `ImportError: cannot import name 'FluxPipeline'`
- **根因**(远程失败 #1):跑在系统 Python 3.8 + diffusers 0.29,不是 omini 环境。
- **修**:`source train/setup_env.sh`(或 `conda activate omini`)。Jupyter 里要确认**内核**指向
  `/root/miniconda3/envs/omini/bin/python`——notebook 第 0 个 cell 会打印 `sys.executable`,不对就换内核。

### 1.2 torch 看不到 GPU
- 驱动 CUDA(`nvidia-smi` 右上角,本机 12.2)必须能跑 torch wheel(cu128 向下兼容 12.x 驱动 ≥ 525)。
- 验证:`python -c "import torch; print(torch.cuda.device_count())"` 应输出 8。
- 若报 kernel/driver 错:重装 `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`
  (降到 cu126 wheel,对旧驱动更保守)。

### 1.3 `cache_idx` 报错,或 kv_cache=True 完全不加速
- **根因**:diffusers 不是 **0.38.0**。新版把 FLUX 注意力类改名,`generate()` 里给 attention 模块
  挂 `cache_idx` 的逻辑失配。
- **修**:`pip install diffusers==0.38.0`,并核对 `huggingface_hub<1.0`、`transformers>=4.55,<5`。

---

## 2. 显存类

### 2.1 `CUDA out of memory`(加载阶段)
- **根因**(远程失败 #2):FLUX.1-dev bf16 全量 ≈33 GiB,24 GB 单卡放不下;`pipe.to("cuda")` 必炸。
- **修**:用脚本自带的 `--dispatch 2gpu`(auto 会自动选)。显存账本:
  - cuda:0:text_encoder 1.7 + text_encoder_2 9.5 + vae 0.3 ≈ **11.5 GiB**
  - cuda:1:transformer 整块 ≈ **22.4 GiB**(4090 可用 ≈23.6 GiB,恰好放下)
- 加载前先 `nvidia-smi` 确认两张卡是空的;别的进程占了 1–2 GB 就会把 cuda:1 挤爆。

### 2.2 OOM 出现在生成阶段(加载没炸)
- 512×512 时 cuda:1 的激活余量只有约 1 GiB;**分辨率升到 1024 或条件数堆太多会爆**。
- 缓解顺序:降 `--size` → 减 `--conditions` → 换 §5.2 量化方案。

---

## 3. 跨设备类(历史 11 连败的结论)

**当前脚本(`kvcache_benchmark.py`)的三件套已把下面所有问题一次性修掉**;本节是为了
"再出现同类报错时知道往哪看"。

### 3.0 修复架构(先懂这个,后面都好排查)
```
cuda:0 = 主场:encoders、vae、latents、scheduler、generator —— generate() 全程只感知 cuda:0
cuda:1 = 只放 transformer(整块)
唯一跨卡点 = omini.transformer_forward 的"桥"(install_tx_bridge):
    输入(白名单 8 个 kwargs)→ 搬到 cuda:1
    输出(noise_pred)      → 搬回 cuda:0
LoRA 加载后 sweep_devices():param/buffer 全量归位
```
**不变式**:除 transformer 权重和桥内部的中间量外,**任何张量都不应该出现在 cuda:1**。
排查任何 device 报错时,先问"是谁把张量带出了主场"。

### 3.1 `enable_model_cpu_offload` → VAE `slow_conv2d` 设备不一致(失败 #3)
- offload 钩子和 omini 自定义 `generate()` 不兼容。**结论:不要用任何 offload,用 2gpu dispatch。**

### 3.2 `Cannot copy out of meta tensor`(失败 #4)
- `enable_sequential_cpu_offload` 在 diffusers 0.38 + accelerate 1.14 组合下把全部模块移到 meta 设备(accelerate.hooks 的 bug)。
- **结论:同上,不要用 sequential offload。** 若非用不可,先降 `accelerate==0.34.0`(§5.1)。

### 3.3 `'dict' object has no attribute 'pooler_output' / 'latent_dist'`(失败 #7 的两个子坑)
- 只在"手工递归搬运模型输出"时出现:`transformers` 的 `ModelOutput` 和 `diffusers` 的 `BaseOutput`
  都是 OrderedDict 子类、**没有 `.to()`**,被当普通 dict 重建就丢属性。
- **现状:新脚本已删除 encoder/vae 输出补丁(不再需要),此坑理论上不会再出现。**
  若你要自己写类似搬运,记得同时 `isinstance(obj, (ModelOutput, BaseOutput))` 并用 `obj.__class__(**...)` 重建。

### 3.4 `device_map="auto"` / `"balanced"`(失败 #5、#6)
- 0.38 不支持 `auto`;`balanced` 会**按权重把 transformer 拆到多卡**,内部 matmul 必炸。
- **结论:FLUX transformer 必须整块放一张卡,不要用 accelerate 的自动切分。**

### 3.5 改 `pipe.device` / `_execution_device`(失败 #8、#9、#10)
- encoders 要求输入在 cuda:0,transformer 要求输入在 cuda:1,`pipe.device` 一个值两边抢,怎么改都顾此失彼。
- **结论:`pipe.device` 保持 cuda:0 不动**,跨卡只在 transformer_forward 桥里做。脚本里有断言:
  `pipe._execution_device != cuda:0` 会在加载时直接报错。

### 3.6 `found at least two devices, cuda:0 and cuda:1`(失败 #11 及其真正的双重根因)
1. **PEFT 把 LoRA 权重留在 cuda:0**(`load_lora_weights` 用 text_encoder 的卡当"模型 device")
   → 已修:`attach_lora()` 在 `set_adapters` 后调用 `sweep_devices()`。
2. **旧 wrapper 只搬输入不搬输出**:noise_pred 留在 cuda:1,`scheduler.step(noise_pred, t, latents)`
   与 cuda:0 的 latents 相撞;同理 `vae.decode` 输入跨卡
   → 已修:桥的返回值统一搬回主场 cuda:0。
- 若这类报错再现,按 §4 工具箱定位"是哪个张量越界"。

### 3.7 千万不要递归搬 `cache_storage`
- KV 缓存靠 **list 对象身份**在多步之间共享(write 时 `append`,read 时读同一个 list)。
  任何"把 kwargs 全部深拷贝式搬运"的写法都会让 write 写进副本、read 读到空缓存——
  表现为 read 步 `IndexError` 或输出全错。桥用**白名单**(8 个 kwargs)就是为了绕开它。

---

## 4. 设备排查工具箱(出现新的 device 报错时按顺序用)

```python
# 4.1 每个大模块在哪张卡
for name in ("text_encoder", "text_encoder_2", "vae", "transformer"):
    m = getattr(pipe, name)
    print(f"{name:>16}: {next(m.parameters()).device}")

# 4.2 找"越界"的 param/buffer(定位 LoRA 残留一类问题)
for name in ("text_encoder", "text_encoder_2", "vae", "transformer"):
    m = getattr(pipe, name)
    home = next(m.parameters()).device
    bad = [(n, str(p.device)) for n, p in m.named_parameters() if p.device != home]
    bad += [(n, str(b.device)) for n, b in m.named_buffers() if b.device != home]
    print(name, "越界数:", len(bad), bad[:5])

# 4.3 pipeline 主场是否还在 cuda:0
print("pipe._execution_device =", pipe._execution_device)

# 4.4 桥是否在位
import omini.pipeline.flux_omini as om
print("tx bridge:", getattr(om.transformer_forward, "_is_tx_bridge", False))
```

分诊逻辑:
- 4.2 有越界 → 再跑一次 `sweep_devices(pipe)`;若 sweep 后又出现,说明有代码在 forward 期间**新建**张量,用报错 traceback 定位是哪一层;
- 4.2 干净但仍报 device 错 → 是**激活值**越界:看 traceback 落在 scheduler/vae(输出没搬回,查 4.4 桥是否被覆盖)还是 transformer 内部(输入没搬进,检查是否有新 kwargs 不在白名单里);
- 4.4 是 False → 说明有人 reload 了 `flux_omini` 模块或重新 import,重跑 `install_tx_bridge()`。

---

## 5. 退路方案(2gpu 方案彻底走不通时)

### 5.1 降 accelerate + sequential offload(只用于"能出图"验证)
```bash
pip install accelerate==0.34.0
```
```python
pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
```
- 老版 accelerate 可能没有 §3.2 的 meta-tensor bug(未在本机验证)。
- ⚠️ **不能用于计时**:每层权重按需 H2D 搬运,墙钟被 PCIe 主导,baseline 和 kv 的对比全被搬运噪声淹没。

### 5.2 bitsandbytes NF4 量化 transformer(单卡方案)
```bash
pip install bitsandbytes
```
```python
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16)
tx = FluxTransformer2DModel.from_pretrained(FLUX_PATH, subfolder="transformer",
                                            quantization_config=q, torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained(FLUX_PATH, transformer=tx,
                                    torch_dtype=torch.bfloat16).to("cuda")
```
- transformer 降到 ≈6.8 GiB,全 pipeline 单卡 24 GB 放得下,**零跨卡问题**。
- ⚠️ 口径注意:量化后绝对速度不可与论文比,但 **baseline vs kv 的加速比仍然成立**(两边同样量化);
  LoRA 挂到 4bit 底座依赖 PEFT 支持,冒烟阶段可先不挂 LoRA 只验证 kv 开关。

### 5.3 换小显存友好的底座
- 若只为验证 KV-Cache 机制,可用 `FLUX.1-schnell`(同架构、免 gated;guidance_embeds 行为不同,
  `guidance_scale` 需按 schnell 口径调整为 0)。质量/速度数字不与 dev 口径混用。

---

## 6. 其它常见坑

### 6.1 `401` / `GatedRepoError`
- 先网页上对 FLUX.1-dev 点 Agree,再 `huggingface-cli login`。
- 内网机器走内部 pip 镜像不影响 HF 下载;若 HF 也被代理,设 `HF_ENDPOINT`。
- ⚠️ token 明文存在 `~/.cache/huggingface/token`,非受控机器用完记得撤销。

### 6.2 多 LoRA / adapter 名冲突
- 重复 `load_lora_weights` 同名 adapter 会抛 ValueError:notebook 里已 try-catch 成"重新激活 + sweep"。
- 多 LoRA 必须 `pipe.set_adapters([...])` 显式激活,否则只有最后一个生效(README Note)。

### 6.3 生成图与条件无关 / 质量差
- 冒烟阶段用 v1 权重 + kv_cache=True 质量差是**预期**(权重不是独立条件训练的),只看速度。
- `guidance_scale` 保持 3.5(dev 蒸馏值);subject 类任务才需要 `image_guidance_scale > 1`。

### 6.4 计时抖动大
- 加大 `--repeats`;确认没有其他进程占卡(`nvidia-smi`);确认没开 offload;
- 2gpu 模式下 D2D 拷贝极小(每步 ~几百 KB),不构成抖动来源;抖动大多来自共享机器的邻居负载。

---

## 7. 训练阶段(stage2_train.ipynb / train_feature_reuse_24gb.py)

### 7.0 训练侧拆分与推理侧【相反】,别拿 §3 的布局套
- 训练主角是 transformer(要梯度,必须和 LoRA/优化器同卡)→ **cuda:0 = transformer**,
  **cuda:1 = 冻结的 encoders + vae**;推理侧正好反过来。
- 上游 `trainer.py:67-69` 把整个 pipeline(≈33 GiB)`.to()` 上一张卡,24 GB 卡
  **加载即 OOM** —— 这就是必须用 `repro/train_feature_reuse_24gb.py` 启动的原因。
  它在运行时 patch:定向放置、encode_images/encode_prompt 输出搬回 cuda:0、
  钉死 Lightning 单卡、采样路径复用推理侧跨卡桥。零改动上游文件。

### 7.1 加载阶段 OOM / `CUDA out of memory` 出现在训练第一步
- 先确认走的是启动器而不是 `train_feature_reuse.sh` 直起(日志开头应有
  `[train24gb] 模式:2-GPU 拆分`)。
- cuda:0 必须**几乎全空**(需 ≥23 GiB):`nvidia-smi` 清掉占卡进程,包括还没关的阶段一 notebook 内核。
- 仍 OOM 按顺序:yaml 里 `condition_size`/`target_size` 降到 `[256,256]` →
  换 yaml 注释里的精简版 `target_modules` → 只能上大显存卡。
- 启动器已默认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 抗碎片,别删。

### 7.2 Lightning 起了 2 个进程 / 模型被加载两遍
- 2 卡可见时 `L.Trainer(devices="auto")` 会自作主张开 DDP,把 33 GB 模型加载两遍。
  启动器已把 devices 钉成训练卡单卡;若绕过启动器自己起训练,必须
  `CUDA_VISIBLE_DEVICES` 只露一张卡(那又回到 33 GB 单卡 OOM)。
  拆分模式与多卡 DDP 不兼容,见启动器 docstring。

### 7.3 日志里 `采样出图失败(训练继续)`
- 采样(test_function → generate)只是监控手段,启动器已兜底不让它打断训练。
- 常见原因:采样瞬间两张卡同时高水位 → 偶发 OOM,过几个 interval 自己恢复;
  持续失败则把报错串到 §3/§4 分诊(采样路径与推理共用跨卡桥,主场在 vae 卡)。

### 7.4 数据下载卡住 / `cache/t2i2m` 报错
- 首次启动 `load_dataset` 要下 2 个 shard(数 GiB)+ `num_proc=32` 解包,十几分钟
  没有 Loss 行是正常的;日志停在 dataset 相关行超过 30 分钟才算异常。
- 磁盘满会表现为 webdataset 解包错误:`df -h .` 确认 ≥15 GiB 空闲后删 `cache/t2i2m` 重来。

### 7.5 loss 不降 / 样图不贴合 canny / 阶段三像素 MAE 偏大
- Prodigy(lr=1)前几百步波动大属正常,看 500 步以上的趋势。
- 确认用的是 `feature_reuse.yaml`(`independent_condition: true`);拿错 config 训出来的
  权重在阶段三的症状是:速度数字正常,但 base/kv 两图差异大(像素 MAE 高)。

---

## 附:三件套修复与失败编号对照

| 修复 | 位置 | 消灭的失败 |
|---|---|---|
| 2gpu 手工 dispatch(transformer 整块) | `load_pipeline()` | #2 OOM、#5/#6 device_map |
| 跨卡桥(输入搬进、**输出搬回**) | `install_tx_bridge()` | #7–#10 addmm/index_select、scheduler/vae 跨卡 |
| LoRA device sweep | `attach_lora()` → `sweep_devices()` | #11 PEFT 残留 |
| (删除)encoder/vae 输出补丁 | —— | #7 的 ModelOutput/BaseOutput 子坑不再可能发生 |
