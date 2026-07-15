# OminiControl2 · 特征复用(Feature Reuse / KV-Cache)复现计划

> 目标:在**一台远程 GPU 机器**上,复现 OminiControl2 论文中 **Feature Reuse(KV-Cache)** 的效率结论——
> 在同一 seed / prompt / 条件图下,只切换 `kv_cache` 开关,测量**墙钟加速比**、**峰值显存**,并验证**生成质量保持不变**。
>
> 配套脚本:`repro/kvcache_benchmark.py`(与本文件同目录,会随仓库一起同步到远程机器)。

---

## 0. 先读:两个"数字口径"必须分清

复现前最容易踩的坑是拿错误的期望值去对标,导致误判"复现失败"。

| 来源 | 声称 | 适用场景 |
|---|---|---|
| 仓库 README(2026-07-02) | **≈ 1.5×** 端到端加速 | **单条件**、8 步、`kv_cache=True` |
| 论文 OminiControl2(arXiv 2503.08280) | **5.9× / >90%** 开销降低 | **多条件**、且专指**条件分支处理开销**的相对降低 |

**关键**:KV-Cache 省下的计算量 ∝ `条件分支数 × (推理步数 − 1)`。

- 单条件 8 步 → 只能看到 ≈ 1.5×(与 README 一致,这就是"对的")。
- 要接近论文的大数字,**必须扫多条件**(见阶段三的 `--conditions 1,2,3`)。

> 只测单条件就下"复现不出 5.9×"的结论 = 用错了对标场景。

---

## 1. 原理速览(为什么 KV-Cache 能成立)

条件分支在推理主循环里有两个"永远不变"的量(源码 `omini/pipeline/flux_omini.py`):

- **条件 token 内容**:在去噪循环**外**只 VAE 编码一次(`generate()` 第 653–654 行)。
- **时间步调制**:条件的 timestep 恒为 `0`(第 660 行 `c_timesteps.append(torch.zeros([1]))`),整个去噪过程条件图**从不加噪**。

但"内容固定 + t=0"还**不够**让 Q/K/V 恒定:

- **默认模式**(`independent_condition=false`):条件的 Q 能注意到 image 的 K/V,而 image 每步去噪都在变 → 条件 hidden state 从第 1 层起被"染上"时间步依赖 → **Q/K/V 随步数变,无法缓存**。
- **独立模式**(`independent_condition=true`):训练/推理时切断条件→(text+image)注意力(`group_mask[2:, :2] = False`,第 714–715 行),条件只看自己 → **每层每步 Q/K/V 逐字节相同** → 第 0 步 `write` 算一次并缓存,后续步 `read` 直接复用(第 730–731 行)。

这就是为什么**必须用 `independent_condition: true` 训练的 LoRA**,`kv_cache=True` 才既快又不掉质量。

---

## 2. 前置条件核查(按"最便宜的失败点优先")

在敲任何训练/推理命令前,先把下面几项卡住:

```bash
# 2.1 硬件
nvidia-smi                       # 记录 GPU 型号、显存、驱动版本
#   FLUX.1-dev bf16 推理:≈ 24–32 GB 起步
#   feature_reuse 训练(512, bs=1, grad-ckpt on):建议 ≥ 40 GB 单卡更稳
df -h .                          # 磁盘:FLUX 权重 ≈ 24 GB + 数据集,预留 ≥ 60 GB

# 2.2 CUDA 与 torch 匹配(远程复现头号翻车点)
nvcc --version                   # 或 nvidia-smi 右上角 CUDA 版本
#   驱动 CUDA 需能配 torch 的 wheel(默认示例用 cu128 → CUDA 12.8)

# 2.3 门禁模型:FLUX.1-dev 是 gated
#   先在 https://huggingface.co/black-forest-labs/FLUX.1-dev 网页点 "Agree"
huggingface-cli login            # 贴入有该模型访问权的 token
```

> 未过门禁 → 后续所有下载 401;CUDA 不匹配 → import torch 或 kernel 直接崩。这两项几分钟内就能失败,放最前面。

---

## 3. 环境搭建

> 📌 **在快手 IDC 那台机器(`aiplatform-bjy-ge47-391`)上,别照抄本节。**
> 那台机器上 conda 已随重启消失,且 `/home` 是 Ceph 网络盘 —— venv 建错位置会卡 70 分钟。
> 直接用:`source train/setup_env.sh`(venv 缺失会自动重建)。
> 机器现状见 [`../MACHINE_STATUS.md`](../MACHINE_STATUS.md),踩坑详情见 [`ENV_REBUILD.md`](ENV_REBUILD.md)。
> 本节保留为**通用**搭建流程,供其它机器参考。

```bash
conda create -n omini python=3.12 -y
conda activate omini

# torch 单独装,index-url 按你的 CUDA 选(下面是 cu128 示例)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖:diffusers 必须严格 ==0.38.0,huggingface_hub 必须 <1.0
pip install -r requirements.txt
```

**版本锁定(强烈建议)**:代码对 diffusers 版本敏感(新版 FLUX 注意力类改名会破坏 KV-Cache 的 `cache_idx` 设置)。装完后固化:

```bash
pip freeze > repro/requirements.lock.txt   # 复现出问题时的对照基线
```

关键 pin:`diffusers==0.38.0`、`huggingface_hub<1.0`、`transformers>=4.55,<5`、`torch==2.8.0`。

---

## 4. 阶段一 · 冒烟测试(证明管线通,约 10 分钟)

先用**现成的 v1 canny 权重**跑通全链路(FLUX 加载 → LoRA 挂载 → `generate()`)。
此阶段**只验证管线和速度,质量不作数**(v1 权重不是独立训练的)。

**推荐方式(24 GB 卡的机器)**:直接从上到下运行 `repro/stage1_smoke_test.ipynb`——
内置环境自检、2 卡 dispatch(FLUX bf16 ≈33 GB 单张 24 GB 卡放不下)、跨卡桥和 LoRA device sweep。
报错按 `repro/TROUBLESHOOTING.md` 分诊。

CLI 等价命令:

```bash
# 在仓库根目录运行(--dispatch auto 会按显存自动选 single / 2gpu)
python repro/kvcache_benchmark.py \
  --lora-repo Yuanshi/OminiControl \
  --lora-weight experimental/canny.safetensors \
  --adapter-name canny --condition-type canny \
  --steps 8 --conditions 1 --repeats 2 \
  --no-independent-check
```

通过标准:脚本正常输出一行 baseline vs kvcache 的时间对比、无异常退出。

---

## 5. 阶段二 · 训练 independent_condition LoRA(质量复现的前提)

HF 上**没有提供** feature-reuse 的预训练权重,必须自己训。配置 `train/config/feature_reuse.yaml` 已就绪(`independent_condition: true`、canny、512×512)。

**推荐方式(24 GB 卡的机器)**:从上到下运行 `repro/stage2_train.ipynb`——后台启动训练、
监控 loss/显存/ckpt、预览采样图、手动停止,全部一键。
⚠️ 上游训练入口把整个 pipeline(≈33 GiB)搬上一张卡,24 GB 卡**加载即 OOM**,
notebook 用 `repro/train_feature_reuse_24gb.py` 启动:自动做 2 卡拆分
(cuda:0 = transformer+LoRA+优化器,cuda:1 = 冻结 encoders+vae,与推理侧布局**相反**),
零改动上游文件。报错按 `repro/TROUBLESHOOTING.md` §7 分诊。

CLI 等价命令(≥40 GB 单卡则用原始脚本):

```bash
# 24 GB 卡(2 卡拆分,自动判断)
python repro/train_feature_reuse_24gb.py

# ≥40 GB 单卡(原始路径;数据首启动时自动下载,或先跑 5.1)
bash train/script/data_download/data_download2.sh   # 5.1 可选预下载
bash train/script/train_feature_reuse.sh            # 5.2 起训;可选 export WANDB_API_KEY=...
```

要点:

- 默认 `feature_reuse.yaml` 只挂 2 个数据 shard(`data_000045/046`)。先小规模确认 loss 下降、`sample_interval` 出的样图合理,再在 yaml 里解注释加数据做完整训练。
- 优化器默认 Prodigy(`lr: 1`),batch size 建议 1,用 `accumulate_grad_batches` 模拟更大 batch。
- 训练产物(LoRA safetensors)保存在 `runs/<run 名>/ckpt/` 下——记下**目录名**和**权重文件名**,阶段三要用。

> 注:README 明确 "Feature reuse 会略微降低质量、增加训练时间"——这是方法本身的 trade-off,属预期。

---

## 6. 阶段三 · 正式测量(用阶段二训出的权重)

**推荐方式**:从上到下运行 `repro/stage3_benchmark.ipynb`——自动找最新 ckpt、
扫 (条件数×步数) 网格、单配置 OOM 自动跳过、输出加速比表 + 像素 MAE 质量核对 + 并排预览。

CLI 等价命令:

```bash
python repro/kvcache_benchmark.py \
  --lora-repo runs/<你的run名>/ckpt \
  --lora-weight <权重文件名>.safetensors \
  --adapter-name canny --condition-type canny \
  --steps 8,20,28 --conditions 1,2,3 --repeats 3 \
  --out repro/kvcache_results
```

**输出**:一张 `(条件数 × 步数)` 的对照表(baseline 墙钟 / kvcache 墙钟 / 加速比 / 两者峰值显存),
并把每个配置的 `base` 与 `kv` 生成图存到 `repro/kvcache_results/`。

### 脚本的三个关键设计(以及为什么)

| 设计 | 原因 |
|---|---|
| **扫 `--conditions`** | 能否接近论文 5.9× 的唯一关键:收益 ∝ 条件数 × 步数。只测单条件必然只有 ≈1.5×。 |
| **先 warmup 再计时 + `cuda.synchronize()`** | CUDA 异步:首次调用含 kernel 编译开销;不同步会量到"提交时间"而非"执行时间"。GPU 计时两大坑。 |
| **base/kv 两图落盘对比** | 独立训练下 KV-Cache 是精确复用,两图应几乎一致;若明显不同,说明 LoRA 没训成独立条件——免费的正确性自检。 |

---

## 7. 阶段四(可选)· 质量指标 FID / CLIP

阶段三的质量核对只是**像素级肉眼对比**。若论文里你要复现的是 FID/CLIP-Score 表,需要:

- 一个评测集(如 COCO 子集或论文所用测试集);
- `pip install clean-fid open_clip_torch`;
- 对 baseline 与 kv_cache 各生成一批图,分别算 FID(vs 真实图)与 CLIP-Score(vs prompt)。

> 需要的话让我把这段加进 `kvcache_benchmark.py`,直接产出"速度 + 质量"完整对照表。

---

## 8. 预期结果与判读

| 观察项 | 预期 | 判读 |
|---|---|---|
| 单条件 8 步加速 | ≈ 1.5× | ✅ 与仓库自测吻合即成功 |
| 加速比随条件数/步数 | 单调上升,多条件明显 > 1.5× | ✅ 复现出论文趋势 |
| base vs kv 生成图 | 独立权重下近乎一致 | ❌ 明显不同 → LoRA 未按独立条件训成 |
| 峰值显存 | kv_cache 略增(存 K/V 缓存) | 属正常;换到的是时间 |

---

## 9. 常见坑 / 排查

- **`import transformers` 报错** → `huggingface_hub` 装成了 ≥1.0,降到 `<1.0`。
- **`cache_idx` 相关报错 / KV-Cache 不生效** → diffusers 不是 0.38.0,换回精确版本。
- **generate 忽略条件、出图与条件无关** → 若在 FLUX.1-dev 上跑 subject 类任务需 `image_guidance_scale>1`;canny 空间任务一般不需要,但确认 `guidance_scale=3.5` 未改(dev 蒸馏值,非自由超参)。
- **多 LoRA 只有最后一个生效** → 必须 `pipe.set_adapters([...])` 显式激活(见 README Note)。
- **计时抖动大** → 加大 `--repeats`,并确认机器无其他占卡进程(`nvidia-smi`)。

---

## 10. 附录 · 关键源码位置(`omini/pipeline/flux_omini.py`)

| 机制 | 位置 |
|---|---|
| 条件循环外只编码一次 | `generate()` 第 653–654 行 |
| 条件 timestep 恒为 0 | 第 660 行 |
| `group_mask` 切断条件→image | 第 710–715 行 |
| KV-Cache write/read 切换 | 第 730–734 行 |
| 条件分支不重算(read 模式) | `transformer_forward` 入参 `image_features=[latents] + (c_latents if use_cond else [])` 第 738 行 |
| 训练侧独立条件开关 | `train/config/feature_reuse.yaml` → `model.independent_condition: true` |

---

### 一句话路线

**过门禁装环境 → 冒烟测通管线 → 训独立条件 LoRA → 扫(条件数×步数)量加速比 →(可选)加 FID/CLIP。**
单条件 ≈1.5× 即对;多条件那几行才是论文标题数字的来源。
