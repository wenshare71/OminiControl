# OminiControl 运行环境状态报告

**版本**: v2(**取代 v1;v1 的核心结论已被实证推翻,见 §5**)
**更新时间**: 2026-07-15
**机器**: `aiplatform-bjy-ge47-391`(快手内部 IDC)
**全链路重建步骤与踩坑详情**: 见 [`repro/ENV_REBUILD.md`](repro/ENV_REBUILD.md)

---

## 0. 当前状态:✅ 环境可用

**驱动一行没动**,环境已从零重建完成,FLUX 权重已就位,可直接跑推理与训练。

```bash
cd /home/wuwenxuan03/OminiControl
source train/setup_env.sh     # venv 若因重启丢失会自动重建
```

---

## 1. 硬件

| 项目 | 值 |
|---|---|
| **GPU** | 8 × NVIDIA GeForce RTX 4090(24 GB × 8,sm_89) |
| **NVIDIA Driver** | **535.54.03** ✅ **够用,无需升级**(理由见 §5) |
| **CUDA Toolkit** | 未装(**不需要** —— PyTorch 的 pip wheel 自带 CUDA 运行时) |
| **OS / 内核** | Ubuntu 20.04.6 LTS / `4.18.0-2.4.3.3.kwai.x86_64`(快手定制) |

> ⚠️ **注意 pipeline 实际只用 2 张卡,不是 8 张。**
> `kvcache_benchmark.py:219` 的 `dispatch="auto"` 是二选一,不是"摊到 8 卡":
> `single`(GPU0 ≥ 40 GB)/ `2gpu`(否则)。4090 是 24 GB → 走 `2gpu`:
> `cuda:0` = encoders+vae+latents,`cuda:1` = **transformer 整块**,其余 6 张闲置。
> **这是刻意的**:accelerate 的 `device_map="balanced"` 会把 transformer 按权重切到多张卡,
> 内部 matmul 直接跨卡报错(历史失败 #6)。详见 `repro/TROUBLESHOOTING.md` §3.4。

**由此得出的显存预算**(决定了能塞几张条件图):

```
4090 单卡              23.99 GiB   (24564 MiB)
FLUX transformer bf16  -22.17 GiB   (23.8 GB 十进制;12B 参数 × 2 字节)
                       ──────────
剩给 KV cache + 激活     1.82 GiB
```

这解释了 stage3 的观测:512px 条件每张 KV ≈ 0.7 GiB → 1 张塞得下,2 张 OOM。

---

## 2. 存储布局(**本机最关键的认知**)

| 挂载点 | 类型 | 容量 | 重启后 | 放什么 |
|---|---|---|---|---|
| `/` `/root` | overlay(**本地盘**) | 7.0 T(可用 6.8 T) | ❌ **回滚** | venv、uv cache |
| `/home/wuwenxuan03` | **Ceph 网络盘** | 1.0 T(可用 945 G) | ✅ **持久** | 仓库、权重、数据、日志 |

🔴 **绝对不要把 venv / pip 缓存建在 `/home/wuwenxuan03`。**
Ceph 上装 28 个包要 **70 分钟**(56 万次小文件写,每次一轮网络往返);
放本地盘 `/root` 是 **176 毫秒**。差 2 万倍。详见 `repro/ENV_REBUILD.md` §2.2。

🔴 **权重/数据必须放 Ceph**,否则每次重启要重下 34 GB。

---

## 3. 软件栈(已验证)

| 组件 | 版本 | 位置 / 备注 |
|---|---|---|
| uv | 0.11.28 | `/root/.local/bin/uv` ← **重启即失** |
| Python | 3.12.13 | `/root/omini-venv` ← **重启即失,可自动重建** |
| **torch** | **2.8.0+cu128** | `cuda.is_available()=True`,8 卡可见,bf16 matmul 通过 |
| torchvision | 0.23.0+cu128 | |
| **diffusers** | **0.38.0** | 项目硬要求(其它版本会破 KV-Cache 的 `cache_idx`) |
| transformers | 4.57.6 | 满足 `>=4.55,<5` |
| peft | 0.19.1 | |
| torchao | (已装,有警告) | ⚠️ 版本过新,C++ 扩展不加载 → 回退纯 Python。**对 KV-Cache 复现无影响** |

## 4. 资产(全部在 Ceph 上,重启后仍在)

| 资产 | 大小 | 位置 |
|---|---|---|
| FLUX.1-dev | ~34 GB(`du` 显示 32G = GiB,对得上) | `$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/` |
| 自训 LoRA ckpt | 5 个档位 | `runs/20260713-111742/ckpt/{1000..5000}/default.safetensors` |
| stage2 训练数据 | 9.0 G | `OminiControl/cache/t2i2m`(**不是权重**,是 text-to-image-2M 分片) |

`HF_HOME=/home/wuwenxuan03/.cache/huggingface`(由 `train/setup_env.sh` 设置)

---

## 5. ⛔ v1 报告的核心结论是错的(存档,以免重蹈)

> **v1(2026-07-14)判定**:
> "问题 3: 显卡驱动与 PyTorch 版本不兼容 ⚠️ **核心阻塞** —— 驱动 535 最高支持 CUDA 12.2,
> cu128 需要驱动 ≥ 580。**必须在 driver ≥ 580 上才能装 PyTorch 2.8 + cu128。**"
> 并因 `/lib/modules` 不存在、无 kernel header、无 DKMS,进一步判定"沙箱内无解",
> 建议**升级驱动 / 找运维 / 改用云 GPU**。

**这个结论是错的。** 两个独立错误叠加而成:

**错误 A:那条报错跟驱动无关。**
v1 引用的失败命令用的是 **cu121 索引**。PyTorch 自 2.5 起停止向 cu121 发布 wheel,
该索引最高只有 2.4.1 —— 所以"找不到 2.8.0"是**索引选错了**,不是驱动拦截。
torch 2.8 只存在于 **cu126 / cu128** 索引。

**错误 B:580 是"捆绑版本",不是"最低要求"。**
580 是随 CUDA 12.8 Toolkit 一起发布的驱动版本;**运行** cu128 程序所需的最低驱动是另一回事:

> **CUDA Minor Version Compatibility (MVC)**:驱动 **≥ R525** 即可运行**任意 CUDA 12.x** 运行时。

补充两点:
- PyTorch 的 pip wheel **自带 CUDA 运行时** → "没有 nvcc / 没装 Toolkit"完全不影响。
- RTX 4090 = sm_89,cu128 wheel **直接带 sm_89 的 SASS** → 连 PTX JIT 都不需要。

`535.54.03 ≥ 525` → **合法**。

**实证**:驱动**未做任何改动**,`torch 2.8.0+cu128` 装上即可用 ——
`available: True` / `gpu count: 8` / `capability: (8, 9)` / `bf16 matmul: OK 98142.07`。

### v1 的真正失误

它把**一个真问题**(Ceph 让装包变得不可能)和**一个假问题**(驱动太老)混为一谈,
并把真问题的症状(装不上包)**归因给了假问题**。
真正的阻塞 —— `/home` 是网络盘 —— v1 **从头到尾没有发现**。

> **教训**:任何"必须升级底层基础设施(驱动/内核/OS)"的结论,动手前**必须先跑一个能证伪它的廉价实验**。
> 这里那个实验只要 30 秒:换 cu128 索引装一次,然后 `torch.cuda.is_available()`。
> 30 秒,省下几天 —— 以及一次在无 header 的定制内核上升驱动可能引发的事故。

---

## 6. 已知的非阻塞警告

| 警告 | 说明 |
|---|---|
| `torchao: Skipping import of cpp extensions ... upgrade to torch >= 2.11.0` | 版本过新,回退纯 Python。**KV-Cache 路径不用 torchao,无影响** |
| `SyntaxWarning: invalid escape sequence '\.'` | torchao 内部,纯噪音 |
| 脚本里 `torch.backends.cudnn.enabled = False` | **刻意为之**,torch 2.8 + diffusers 0.38 的 VAE `conv_in` workaround |

---

## 7. 安全

⚠️ **HF token 明文风险**:`hf auth login` 会把 token 明文写进 `$HF_HOME/token`,
而本机 `HF_HOME` 位于 **Ceph 共享持久盘**。因此推荐 `export HF_TOKEN=hf_xxx`(只存活于当前 shell,不落盘),
`train/setup_env.sh` 也刻意**不写** token。

🔴 **用完这台机器,去 https://huggingface.co/settings/tokens 撤销 token**;
若曾用过 `hf auth login`,另需 `rm $HF_HOME/token`。

---

## 附:文档版本

| | v1 | v2(本文件) |
|---|---|---|
| 日期 | 2026-07-14 | 2026-07-15 |
| 核心结论 | ❌ "驱动太老,必须升 580,沙箱内无解" | ✅ "驱动够用,真凶是 Ceph,已修复" |
| 环境状态 | ❌ 阻塞 | ✅ 可用 |
| 生成模型 | Claude Opus 4.7 | Claude Opus 4.8 |
