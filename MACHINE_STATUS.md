# OminiControl 项目运行环境诊断报告

**生成时间**: 2026-07-14
**诊断人**: Claude Code
**目标项目**: OminiControl (Diffusion Transformer 控制框架,基于 FLUX)

---

## 1. 机器硬件信息

| 项目 | 值 |
|---|---|
| **GPU** | 8 × NVIDIA GeForce RTX 4090 (24 GB × 8) |
| **GPU 索引** | 0, 1, 2, 3, 4, 5, 6, 7 (nvidia-smi 全部可见) |
| **GPU 当前状态** | 全部 On,空闲,无占用 (2 MiB / 24564 MiB) |
| **NVIDIA Driver** | **535.54.03** (2023-06-06 发布) |
| **Driver 支持的最高 CUDA** | **12.2** |
| **CUDA Toolkit 安装情况** | 仅装了 nsight-compute 12.1,无 nvcc / 无 toolkit |
| **磁盘** | 7.0 T 总量,267 G 已用,6.8 T 可用 (overlay 挂载) |
| **OS** | Ubuntu 20.04.6 LTS (Focal Fossa) |
| **Kernel** | **4.18.0-2.4.3.3.kwai.x86_64** ⚠️ (快手定制内核) |

---

## 2. Python / 包管理器现状

| 工具 | 状态 | 路径 |
|---|---|---|
| **系统 Python** | ✅ 3.8.10 | `/usr/bin/python3` |
| **系统 pip** | ✅ 24.0 (for py3.8) | `/usr/local/bin/pip` |
| **uv 管理的 Python 3.12** | ✅ 解释器在 | `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12` |
| **uv 命令行** | ❌ 不可用 | PATH 中找不到 `uv` |
| **conda / miniconda / anaconda** | ❌ 未安装 | `/root/miniconda3` / `/root/anaconda3` 均不存在 |
| **PATH 中 `/video/anaconda/bin`** | ❌ 路径在但目录不存在 | (残留 PATH 条目) |
| **`pip3.12` / `pip3`** | ✅ uv 缓存的 pip 在 `.venv/bin` | `/home/wuwenxuan03/OminiControl/.venv/bin/pip3.12` |

---

## 3. 目标项目结构

```
/home/wuwenxuan03/
├── ominicontrol/                 # ❌ 空目录(只有 . 和 ..)
├── OminiControl/                 # ✅ 实际项目
│   ├── .venv/                    # ⚠️ 存在但几乎为空
│   ├── CLAUDE.md                 # ✅ 项目说明
│   ├── requirements.txt          # 依赖清单
│   ├── README.md
│   ├── omini/                    # 核心代码 (pipeline / train_flux)
│   ├── train/                    # 训练脚本 + YAML 配置
│   ├── examples/                 # Jupyter notebook
│   ├── repro/                    # 复现脚本
│   ├── assets/
│   ├── cache/
│   ├── kvcache_results/
│   ├── runs/
│   └── scripts/
└── bagel/.venv_bagel/            # 另一个项目的 venv(不相关)
```

### 3.1 `.venv` 当前内容

```
.venv/
├── bin/
│   ├── activate / activate.fish / activate.csh ...
│   ├── pip3, pip3.12 (→ uv 管理的 pip)
│   └── python → /root/.local/share/uv/python/cpython-3.12.../bin/python3.12
├── lib/python3.12/site-packages/
│   ├── pip
│   ├── pip-25.0.1.dist-info
│   ├── _virtualenv.pth
│   └── _virtualenv.py            ← 除此之外什么都没有
├── lib64 → lib
├── CACHEDIR.TAG
├── .gitignore
└── pyvenv.cfg                    # 创建者: uv 0.11.28
```

**结论**: `.venv` 只装了 pip 本身,**torch / diffusers / transformers / peft / gradio / jupyter / torchao 全部未安装**。

### 3.2 目标依赖 (来自 `requirements.txt`)

```text
# Validated stack (July 2026): diffusers 0.38.0, python 3.12, torch 2.8 (cu12x).
transformers>=4.55,<5
diffusers==0.38.0
huggingface_hub<1.0
peft
opencv-python
protobuf
sentencepiece
gradio
jupyter
torchao
```

训练额外 (`train/requirements.txt`): `lightning`, `datasets`, `prodigyopt`, `wandb`, `torchvision`。

---

## 4. 遇到的问题

### 问题 1: 关键工具链缺失
- ❌ 没有 `uv` 命令(只有它下载的 Python 解释器残留)
- ❌ 没有 `conda` / `miniconda` / `anaconda`
- ❌ `.venv` 是个空壳,没有任何项目依赖
- ⚠️ PATH 里有个不存在的 `/video/anaconda/bin`,是脏配置

### 问题 2: Python 版本不匹配
- 系统 Python 是 **3.8.10**,太老,多数现代 ML 库(pydantic 2、diffusers 0.38、torch 2.8)都已不支持
- 目标需要 **3.12** —— 只有 uv 之前下载的解释器在用,需要重新挂到 venv

### 问题 3: 显卡驱动与 PyTorch 版本不兼容 ⚠️ **核心阻塞**
当前 Driver **535.54.03**,对应 CUDA runtime 最高 **12.2**。

PyTorch 2.8 在官方源 (https://download.pytorch.org/whl/) 的可用情况:

| Wheel | Driver 最低要求 | 实际能装到的 PyTorch 版本范围 | 我们能否用 |
|---|---|---|---|
| `cu118` | ≥ 520 | 最高 2.4.x (PyTorch 2.5+ 已停供) | ❌ 没 2.8 |
| `cu121` | ≥ 530 | **最高 2.4.1** (实测 pip 报错确认) | ❌ 没 2.8 |
| `cu124` | ≥ 550 | 2.4+ | ❌ driver 太老 (535 < 550) |
| `cu126` | ≥ 560 | 2.5+ | ❌ driver 太老 |
| `cu128` | ≥ 580 | 2.5+ | ❌ driver 太老 |

**实际报错**:
```
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
Looking in indexes: https://download.pytorch.org/whl/cu121
ERROR: Could not find a version that satisfies the requirement torch==2.8.0
(from versions: 2.1.0+cu121, 2.1.1+cu121, 2.1.2+cu121, 2.2.0+cu121,
 2.2.1+cu121, 2.2.2+cu121, 2.3.0+cu121, 2.3.1+cu121, 2.4.0+cu121, 2.4.1+cu121)
```

**结论**: 必须在 driver ≥ 580 (推荐 580.95.05) 上才能装 PyTorch 2.8 + cu128,即 OminiControl 文档里"validated stack"。

### 问题 4: 升级 driver 存在硬性卡点 ⚠️
环境是 **Ubuntu 20.04 + 快手定制内核 4.18.0-2.4.3.3.kwai.x86_64**。升级 driver 需要编译 kernel module,需要 kernel headers,但:

| 路径 | 检查结果 |
|---|---|
| `/lib/modules/` | ❌ **目录不存在** |
| `/lib/modules/$(uname -r)/` | ❌ 不存在 |
| `/usr/src/linux-headers-*` | ❌ 无任何 header |
| `/boot/` | 存在但是空的 (Apr 15 2020 创建) |
| `dkms status` | ❌ 命令无输出 (DKMS 未装 / 不可用) |
| `apt-cache search linux-headers-4.18.0-2.4.3.3.kwai` | ❌ 官方源没有 |

**实测 driver 信息**:
```
/proc/driver/nvidia/version → NVRM version: NVIDIA UNIX x86_64 Kernel Module
                              535.54.03  Tue Jun  6 22:20:39 UTC 2023
                              GCC version:  gcc version 4.8.5
```

**解读**:
1. `/lib/modules` 不存在 + 容器化的 OS 信息 → **极可能是容器/sandbox,driver 由宿主机 bind-mount 进来**,在沙箱里**没有权限/无能力改 driver**
2. 即便有权限,快手定制内核 `4.18.0-2.4.3.3.kwai` 在标准 Ubuntu apt 源里也没有对应 header 包,需要从快手内部源获取,或编译 header from `/proc/config.gz`
3. 当前 driver 是 2023 年 6 月发布的(基于 GCC 4.8.5 / Red Hat 工具链),看起来是某次环境初始化时 runfile 装的,之后没动过

### 问题 5: 降级 PyTorch 不可接受
若降级到 torch 2.4.1+cu121:
- diffusers 必须降到 ~0.31(0.32 改了 FLUX pipeline 的 transformer 块结构)
- OminiControl 的 `flux_omini.py` 用 monkey-patch 替换 forward,大概率无法在新结构上工作
- **用户已明确拒绝降级路线**

---

## 5. 关键路径与决策点

### 5.1 环境类型判定
请确认以下任意一项:

- [ ] 你是这台机器的物理 owner / 有 root sudo + 能 reboot?
- [ ] 你是通过容器/SSH 进去的,driver 是宿主机共享?
- [ ] `/lib/modules` 是否真的不存在,还是 sandbox 限制了 `ls`?

### 5.2 决策矩阵

| 你的情况 | 推荐路线 |
|---|---|
| 有 sudo,能 reboot,header 可获取 | 走 NVIDIA runfile 升级到 580.95.05,然后 `pip install torch==2.8.0+cu128` |
| 标准 Ubuntu 内核,无 header 问题 | 走 NVIDIA 官方 apt 源装 `nvidia-driver-580` |
| 容器/sandbox,driver 由宿主给 | 找运维升级宿主机 driver,本沙箱内无解 |
| 不想动 driver,接受降级 | 走 torch 2.4.1+cu121 + diffusers 0.31 路线(用户已拒绝) |
| header 拿不到(快手定制内核) | 走 DKMS runfile + 找运维拿 kwai 内部 header 包 |

---

## 6. 已尝试的解决方案记录

| 尝试 | 结果 |
|---|---|
| `pip install torch==2.8.0 --index-url .../cu121` | ❌ 报错:该 wheel 不存在,cu121 最高 2.4.1 |
| 探测 `conda` / `uv` | ❌ 均不可用 |
| 探测 `uv pip install` | ❌ uv 命令不存在 |
| 探测 conda env 列表 | ❌ 无 conda |

---

## 7. 下一步建议 (按优先级)

### 🟢 推荐路径 (有 sudo 时)
```bash
# 1. 升级 driver 到 580.95.05
cd ~/nvidia-drv
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/580.95.05/NVIDIA-Linux-x86_64-580.95.05.run
sudo ./NVIDIA-Linux-x86_64-580.95.05.run --dkms

# 2. 重启后验证
sudo reboot
nvidia-smi   # 期望: Driver Version: 580.x, CUDA Version: 12.8

# 3. 装 PyTorch
cd /home/wuwenxuan03/OminiControl
source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. 装项目依赖
pip install -r requirements.txt
pip install -r train/requirements.txt
```

### 🟡 备选 (快手定制内核环境)
联系快手运维获取:
- `linux-headers-4.18.0-2.4.3.3.kwai.x86_64` (或 DKMS 兼容的 source)
- 宿主机 driver 升级协助(若是容器)

### 🔴 兜底 (都不行)
考虑使用云 GPU (如 autodl / 恒源云) 跑 OminiControl,本地只做代码编辑。

---

## 8. 联系信息

- 文档版本: v1
- 生成工具: Claude Code (claude.ai/code)
- 模型: Claude Opus 4.7
- 工作目录: `/home/wuwenxuan03`
