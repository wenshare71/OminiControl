# 从零重建运行环境 · 全链路与踩坑记录

> **机器**: `aiplatform-bjy-ge47-391`(快手内部 IDC,8 × RTX 4090)
> **日期**: 2026-07-15
> **背景**: 机器重启后 `/root` 回滚,环境全部消失,从零重建。
> **结论先行**: 环境重建成功,**驱动一行没动**。上一版诊断报告 (`MACHINE_STATUS.md` v1)
> 判定的"核心阻塞 = 驱动太老必须升到 580"是**误诊**;真正的阻塞是它从未发现的
> **`/home` 是 Ceph 网络盘**。详见 §2.1 / §2.2。

---

## 0. 先懂这个:本机的存储模型

**这是全篇最重要的一张表。** 后面 80% 的坑都是它推导出来的:

| 挂载点 | 类型 | 容量 | 重启后 | 小文件写 | 该放什么 |
|---|---|---|---|---|---|
| `/` `/root` | overlay(**本地盘**) | 7.0 T(可用 6.8 T) | ❌ **回滚到基础镜像** | ✅ 快 | venv、uv cache、临时文件 |
| `/home/wuwenxuan03` | **Ceph 网络盘** | 1.0 T(可用 945 G) | ✅ **持久** | ❌ **灾难级慢** | 仓库、模型权重、训练数据、日志 |

`df -h` 里的原始证据:

```
Filesystem                                                        Size  Used Avail Use%
overlay                                                           7.0T  267G  6.8T   4%  /
10.80.201.41,10.80.202.44,10.80.202.47,10.48.58.105,10.48.57.49:/mmu_ssd/wuwenxuan03
                                                                  1.0T   80G  945G   8%  /home/wuwenxuan03
```

`/root` 会回滚的**实证**(不是猜):重启后 `ls -la /root/.cache/huggingface/` 里的目录
时间戳是 **2024 年 4 月 / 5 月**,且 `token` 文件不存在 —— 这就是基础镜像烧进去的原始内容,
之前所有改动全没了。

### 由此推出的两条铁律

1. **venv 必须建在 `/root`(本地盘)** —— 建在 Ceph 上装包要 **70 分钟**,本地盘 **176 毫秒**(§2.2)。
   代价是每次重启要重建,但重建只要几秒,`train/setup_env.sh` 已自动化。
2. **权重/数据必须放 `/home/wuwenxuan03`(Ceph)** —— 34 GB 的 FLUX 不能每次重启重下。
   Ceph 对**少量大文件顺序读写**很正常,只对**海量小文件**灾难。venv 恰好是后者,权重恰好是前者。

---

## 1. 全链路(可直接复制,约 15 分钟 + 下载时间)

```bash
# ── 1) 装 uv(/root 上,重启会没,重装很快) ──────────────────
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version                      # 期望 0.11.28+

# ── 2) 建 venv —— 注意路径是 /root,不是仓库里! ─────────────
# WHY: 见 §0 铁律 1。建在 /home/wuwenxuan03/OminiControl/.venv 会卡 70 分钟。
uv venv --python 3.12 /root/omini-venv
source /root/omini-venv/bin/activate
python -V                         # 期望 Python 3.12.13

# ── 3) 装 torch(cu128) ────────────────────────────────────
# WHY cu128 而不是 cu121: torch 2.8 只在 cu126/cu128 索引里存在,cu121 最高只到 2.4.1。
# WHY 驱动 535 能跑 cu128: CUDA Minor Version Compatibility,见 §2.1。
uv pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
# 期望: Installed 28 packages in 176ms

# ── 4) 验 CUDA(装依赖前先验,失败早发现) ────────────────────
python -c "
import torch
print('torch:', torch.__version__, '| built-cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())
print('gpu0:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
print('bf16 matmul: OK', (x @ x).float().abs().sum().item())
"
# 期望: available: True | gpus: 8 | RTX 4090 | capability: (8, 9) | bf16 matmul: OK

# ── 5) 装项目依赖 ──────────────────────────────────────────
# WHY 两个 -r 写在同一条命令: uv 一次解析。分两条跑第二次会重新解析,
#     可能把 cu128 的 torch 换成 PyPI 上的通用版(CUDA 就废了)。
cd /home/wuwenxuan03/OminiControl
uv pip install -r requirements.txt -r train/requirements.txt

# ── 6) 确认 torch 没被上一步冲掉 ────────────────────────────
python -c "
import torch, diffusers, transformers, peft
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('diffusers:', diffusers.__version__, '| transformers:', transformers.__version__, '| peft:', peft.__version__)
"
# 期望: torch 2.8.0+cu128 / cuda True / diffusers 0.38.0 / transformers 4.57.6 / peft 0.19.1

# ── 7) HF 缓存指向 Ceph + 鉴权 ─────────────────────────────
export HF_HOME=/home/wuwenxuan03/.cache/huggingface   # WHY: 见 §0 铁律 2
export HF_TOKEN=hf_xxxxxxxx                           # WHY 不用 hf auth login: 见 §4 安全

# ── 8) 下 FLUX 权重(~34 GB) ───────────────────────────────
uv pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # WHY: 见 §2.6,不导出就是静默无效
nohup python -c 'from huggingface_hub import snapshot_download; print("DONE:", snapshot_download("black-forest-labs/FLUX.1-dev", ignore_patterns=["flux1-dev.safetensors","ae.safetensors","*.gguf"], max_workers=8))' > /home/wuwenxuan03/flux_dl.log 2>&1 &
echo "PID: $!"
```

### 为什么 `ignore_patterns` 能省掉 24 GB

FLUX.1-dev 仓库把权重**存了两份**,格式不同、内容一样:

| 文件 | 大小 | 格式 | 要吗 |
|---|---|---|---|
| `transformer/*.safetensors` | ~23.8 GB | diffusers 分片 | ✅ **要** |
| `flux1-dev.safetensors` | ~23.8 GB | 单文件(ComfyUI 用) | ❌ 重复 |
| `vae/*.safetensors` | ~335 MB | diffusers | ✅ **要** |
| `ae.safetensors` | ~335 MB | 单文件 | ❌ 重复 |

全量 ~58 GB → 加 `ignore_patterns` 后 **~34 GB**(transformer 23.8 + T5-XXL 9.5 + CLIP/VAE/tokenizer ~0.6)。

> 📌 `du -sh` 报 **32G** 不是缺文件:`du` 显示的是 **GiB**,34 GB(十进制) = 31.7 GiB。对得上。

---

## 2. 踩过的坑(症状 → 误判 → 真因 → 修复)

### 2.1 ⛔ 上一版诊断报告的核心结论是错的:"驱动 535 太老,必须升到 580"

| | |
|---|---|
| **症状** | `pip install torch==2.8.0 --index-url .../cu121` → `ERROR: Could not find a version ... (from versions: 2.1.0+cu121 ... 2.4.1+cu121)` |
| **v1 报告的结论** | 驱动 535 只支持到 CUDA 12.2,cu128 需要驱动 ≥ 580 → **核心阻塞**,必须升驱动。又发现 `/lib/modules` 不存在、无 header、`/boot` 空、无 DKMS → 判定"沙箱内无解",建议找运维或改用云 GPU。 |
| **真因** | **两个独立错误叠在一起,得出了一个双重错误的结论。** |

**错误 A —— 报错根本不是驱动引起的。**
那条命令用的是 **cu121 索引**。PyTorch 从 2.5 起就不再往 cu121 发布 wheel 了,该索引最高只有 2.4.1。
所以"找不到 2.8.0"是**索引选错**,不是驱动拦截。torch 2.8 只存在于 **cu126 / cu128** 索引。
换 cu128 索引,同一台机器、同一个驱动,**装上就能用**。

**错误 B —— 580 这个数字被用错了地方。**
580 是"**CUDA 12.8 Toolkit 捆绑发布**的驱动版本",不是"**运行** cu128 程序所需的最低驱动"。
两者差着一个 **CUDA Minor Version Compatibility (MVC)**:

> 驱动 **≥ R525** 即可运行**任意 CUDA 12.x** 运行时。

再加两条:
- PyTorch 的 pip wheel **自带 CUDA 运行时**,不依赖系统 CUDA Toolkit —— 所以"没有 nvcc"完全不影响。
- RTX 4090 = **sm_89**;cu128 的 wheel 里**直接带 sm_89 的 SASS**,连 PTX JIT 都不需要。

`535.54.03 ≥ 525` → 合法。

**实证(最硬的证据)**:驱动**一行没动**,`torch 2.8.0+cu128` 装上后
`cuda.is_available() = True`,8 卡全见,bf16 matmul 正常出数。**v1 的核心结论就此作废。**

> ⚠️ **这个误诊的代价**:v1 报告推荐的动作是"跑 runfile 升驱动 / 找运维 / 换云 GPU"。
> 在一台快手定制内核(`4.18.0-2.4.3.3.kwai`)、无 header、大概率是容器的机器上升驱动,
> 轻则白折腾几天,重则把机器搞挂。**而实际需要做的只是把 `cu121` 改成 `cu128`。**

---

### 2.2 🔥 真正的阻塞(v1 从未发现):uv 装包卡 70 分钟

| | |
|---|---|
| **症状** | `uv pip install` 在 `[27/28] sympy==1.14.0` 停住,进度条一小时不动。伴随警告:`Failed to hardlink files; falling back to full copy ... cache and target directories are on different filesystems` |
| **误判 1(我的)** | "sympy 有 2000+ 个小 .py 文件" → **错**。`du` 的报错暴露 uv 当时其实在写 `torch/include/ATen/ops/`(2000+ 个小 `.h`)。**uv 是并行安装的,`[27/28] sympy` 这个标签具有误导性**,它不代表当前在装什么。 |
| **误判 2(我的)** | "`/home` 满了" → **错**。`df` 显示 945 G 可用,用量 8%。 |
| **真因** | **`/home/wuwenxuan03` 是 Ceph 网络盘。** `/proc/25605/io` 摊牌:`syscw: 560895` —— **56 万次写系统调用**,每一次都是一轮网络往返。12.5 GB 数据(`wchar: 12564144920`)就这么以 ~4 KB/s 的有效速率爬。 |
| **修复** | **把 venv 和 uv cache 放到同一块本地盘**(`/root/omini-venv` + `/root/.cache/uv`)。同盘 → hardlink 生效 → 不再逐文件拷贝。 |

**修复前后**:

```
Ceph  (/home/wuwenxuan03/OminiControl/.venv):  ~70 分钟,还没装完
本地盘 (/root/omini-venv):                      Installed 28 packages in 176ms
```

**差 2 万倍。** 而且**零重新下载** —— 3 GB 的 wheel 早就在本地 uv cache 里了,慢的从来不是网络,是**往 Ceph 上落 56 万个小文件**。

> 💡 **一句话记住**:hardlink 警告 (`different filesystems`) 出现时,在 Ceph 上它不是"性能建议",
> 是**致命错误的前兆**。看到就立刻停,别等。

### 2.2.1 附带坑:删掉那个死 venv 本身也会卡

`rm -rf` 2 万个小文件 = 2 万次网络往返,同样卡死。正确姿势:

```bash
mv .venv .venv.dead            # 同盘 rename,瞬间完成
nohup rm -rf .venv.dead &      # 扔后台慢慢删,不挡事
```

---

### 2.3 监控本身会拖慢被监控的东西

`watch du -sh .venv` 看着人畜无害,实际上**每轮都重扫整棵树**,在慢网络盘上**跟安装抢 I/O**。
正确做法是读 `/proc/PID/io` —— 内核计数器,**零文件系统开销**:

```bash
PID=<真实数字>    # ⚠️ 注意:echo "PID: $!" 只是打印,并不会赋值给 $PID
prev=0
while kill -0 $PID 2>/dev/null; do
  w=$(awk '/^write_bytes/{print $2}' /proc/$PID/io)
  printf '[%s] 已写 %6d MB | 速率 %4d MB/s\n' \
    "$(date +%H:%M:%S)" "$((w/1048576))" "$(((w-prev)/1048576/5))"
  prev=$w; sleep 5
done
```

**例外**:`du` 只扫 `blobs/`(27 个文件)是便宜的,可以用来看下载进度。贵的是扫整棵树。

---

### 2.4 ❌ 我的监控指标选错了:`rchar` 看不到网络下载

| | |
|---|---|
| **症状** | 下载已经跑了 8 分钟,`rchar` 死死钉在 **26 MB**,速率恒为 0 —— 看起来完全卡死。 |
| **误判(我的)** | "你在看错的 PID" → **错**,PID 78607 从头到尾都是对的。 |
| **真因** | **`rchar` 只统计 `read()` / `pread()`,不统计 socket 的 `recv()`。** Python 从网络收数据走的是 `recv()`,所以永远不进 `rchar`。那 26 MB 是 Python **import 时读自己的库文件**。 |
| **修复** | 看 **`write_bytes`** —— 下载的数据最终要 `write()` 落到 Ceph 上,这个计数器如实反映进度。 |

**`/proc/PID/io` 字段速查**(踩一次坑换来的):

| 字段 | 含义 | 用来干嘛 |
|---|---|---|
| `rchar` | `read()`/`pread()` 的字节数 | ⚠️ **不含网络**,别拿来看下载 |
| `wchar` | `write()` 的字节数 | 看写入总量 |
| `write_bytes` | 真正提交到存储层的字节数 | ✅ **看下载/安装进度就用它** |
| `read_bytes` | 真正从块设备读的字节 | `0` = 全部命中 page cache,正常 |
| `syscw` | 写系统调用**次数** | ✅ **小文件地狱的探针**(56 万 = 出事了) |

**顺带**:`ps` 的 STAT 列 `Sl+` = 可中断睡眠 + 多线程 + 前台 = **健康**。
要担心的是 `D`(不可中断睡眠,通常是 I/O 卡死)。

---

### 2.5 "进度条卡在 30%" —— 是误读,不是故障

```
Fetching 27 files:  30%|██▉       | 8/27 [02:38<08:06, 25.60s/it]
```

**这个进度条数的是「文件数」,不是「字节数」。** 已完成的 8 个是 config / tokenizer / CLIP 这类小文件;
**还在飞的 5 个是 10 GB 级的 transformer 分片 + 9.5 GB 的 T5-XXL**。
大文件不下完,进度条一格都不会动 —— 卡十几分钟完全正常。

同理,`[02:38<08:06]` 这个 ETA 是**按文件数外推**的,严重低估,**别信**。

**判断是否真卡死,看这两个(任选)**:

```bash
# a) 分片在长大吗(blobs 只有 ~27 个文件,du 很便宜)
watch -n 10 'du -sh $HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/blobs'

# b) .incomplete 的大小和 mtime 在动吗
ls -la $HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/blobs/*.incomplete
```

健康的样子:5 个 `.incomplete`(1.1G / 474M / 576M / 374M / 1.2G),mtime 是**刚刚**。

> 💡 **另一个反直觉的信号**:`du` 报 `cannot access '.../.tmpSLwObg': No such file or directory`
> 看着像故障,其实是**正在积极写入的证明** —— uv/HF 都用"先写 `.tmp` 再 rename"的原子写模式,
> `du` 只是撞上了那个瞬间。

---

### 2.6 下载太慢(8.5 MB/s)→ 换 hf_transfer

**先定位瓶颈**:8.5 MB/s **不是 Ceph 的锅** —— §2.2 那 12 GB 之所以慢是因为 56 万次小文件写;
大文件顺序写 Ceph 能跑几百 MB/s。所以瓶颈在 **Python `requests` 的下载路径**。

```bash
kill <旧PID>
uv pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # ⚠️ 必须在启动进程「之前」、「同一个 shell」里导出,否则静默无效
# 重跑同一条 snapshot_download 命令
```

- **风险几乎为零**:HF **支持断点续传**,已下的 GB 数不会白费,重跑接着下。
- **代价**:hf_transfer **没有逐文件进度条**,日志会更朴素 —— 所以别再拿进度条判断死活,看 `write_bytes`。
- **兜底**:真报错就 `unset HF_HUB_ENABLE_HF_TRANSFER`,退回原来那条(慢但能用)的路,下载照样续传。

---

### 2.7 终端出现 `dquote>` 是怎么回事

粘贴 heredoc(`cat > x.py <<'EOF' ... EOF`)时断了 → `EOF` 没被识别 → 后面的 Python 代码被当成 shell 解析
→ 引号不配对 → zsh 等你补上收尾的 `"`,于是一直显示 `dquote>`。

**修复**:`Ctrl+C`,然后改用**粘贴安全**的单行写法 —— 外层单引号、内层双引号,没有多行状态可丢:

```bash
nohup python -c 'print("hello")' > log 2>&1 &
```

---

### 2.8 torchao 的警告(无害,但要知道为什么)

```
Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0 (found 2.8.0+cu128)
SyntaxWarning: invalid escape sequence '\.'  (torchao/quantization/quant_api.py:1745)
```

装到了一个**过新**的 torchao,它的 C++ 扩展加载不了,回退到纯 Python 实现。

**对 KV-Cache 复现零影响 —— 这条路径根本不用 torchao。** 只有将来走 FP8/NF4 量化(方案二)才需要,
届时把 torchao 钉到匹配 torch 2.8 的版本即可。第二条 `SyntaxWarning` 纯噪音。

---

## 3. 重启之后怎么恢复

**重启会毁掉**:`/root/omini-venv`、`/root/.local/bin/uv`、所有 `export`、`~/.bashrc` 里的改动。
**重启不会碰**:仓库、FLUX 权重、`cache/t2i2m` 训练数据、`runs/` 里的 LoRA ckpt(都在 Ceph 上)。

所以恢复只有一条命令:

```bash
cd /home/wuwenxuan03/OminiControl
source train/setup_env.sh          # 缺 venv 会自动重建(约 1~2 分钟),已有就秒进
```

**权重不用重下** —— 它们在 `HF_HOME=/home/wuwenxuan03/.cache/huggingface`(Ceph)。
这正是 §0 铁律 2 的全部意义。

> 📌 `OminiControl/cache/` 那 9.0 G 是 **stage2 的训练数据**(`cache/t2i2m`,text-to-image-2M 分片),
> 不是模型权重。见 `omini/train_flux/train_token_integration.py:107` 的 `cache_dir="cache/t2i2m"`。
> 它在 Ceph 上,重启后**还在**,不用重下。

---

## 4. 安全

⚠️ **HF token 是明文存储的。**

| 方式 | token 落在哪 | 评价 |
|---|---|---|
| `hf auth login` | `$HF_HOME/token` **明文** | ❌ 本机 `HF_HOME` 在 **Ceph 共享持久盘**上,等于把明文 token 写进公司集群的持久存储 |
| `export HF_TOKEN=hf_xxx` | 只在当前 shell 的环境变量里 | ✅ **推荐**,不落盘,shell 关掉就没 |

因此 **`train/setup_env.sh` 里刻意不写 `HF_TOKEN`**。权重下完之后本地推理**根本不需要 token**;
真要再下东西时临时 `export` 一次就行。

🔴 **用完这台机器,记得去 HF 后台撤销 token**(https://huggingface.co/settings/tokens)。
如果当初用过 `hf auth login`,还要顺手 `rm $HF_HOME/token`。

---

## 5. 最终环境快照

| 项目 | 值 |
|---|---|
| 机器 | `aiplatform-bjy-ge47-391`,8 × RTX 4090 (24 GB) |
| 驱动 | **535.54.03(未改动)** |
| OS / 内核 | Ubuntu 20.04.6 / `4.18.0-2.4.3.3.kwai.x86_64` |
| venv | `/root/omini-venv`(Python 3.12.13,uv 0.11.28 创建)**← 重启即失,可自动重建** |
| torch | **2.8.0+cu128**(`cuda.is_available()=True`,8 卡,sm_89,bf16 OK) |
| torchvision | 0.23.0+cu128 |
| diffusers / transformers / peft | **0.38.0** / 4.57.6 / 0.19.1 |
| `HF_HOME` | `/home/wuwenxuan03/.cache/huggingface`(Ceph,**持久**) |
| FLUX.1-dev | ~34 GB,已缓存 |
| LoRA ckpt | `runs/20260713-111742/ckpt/{1000..5000}/default.safetensors`(自训,Ceph,**持久**) |
| 训练数据 | `OminiControl/cache/t2i2m`(9.0 G,Ceph,**持久**) |

---

## 附:这次误诊给的教训

v1 报告把**一个真问题**(Ceph 让装包变得不可能)和**一个假问题**(驱动太老)混在了一起,
并且把真问题的症状(装不上包)**归因给了假问题**。

它做对的:客观事实(硬件、路径、版本)基本准确。
它做错的:**看到一条报错,直接跳到了一个听起来合理的因果,没有验证这条因果链的每一环。**
—— cu121 装不上 2.8,只证明了"cu121 索引里没有 2.8",不证明"驱动不行"。

> **可复用的判据**:任何"必须升级底层基础设施(驱动/内核/OS)"的结论,
> 在动手之前**必须先有一个能证伪它的廉价实验**。这里那个实验只要 30 秒:
> `uv pip install torch==2.8.0 --index-url .../cu128` 然后 `torch.cuda.is_available()`。
> 30 秒,省下几天。
