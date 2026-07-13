# 阶段二 · 训练 independent_condition LoRA(一键版)
#
# **目标**:训出 `independent_condition: true` 的 canny LoRA —— 这是 KV-Cache
# 质量复现的前提(阶段一用的 v1 权重不是独立训练的,只能验证速度)。
#
# **前提**:阶段一(`stage1_smoke_test.ipynb`)已跑通;用同一个 `omini` 内核。
#
# **本 notebook 做什么**:
# 1. 环境自检(双卡空闲显存 / 磁盘 / 配置文件);
# 2. 后台启动训练(`repro/train_feature_reuse_24gb.py`);
# 3. 反复运行监控 cell 看 loss / 显存 / checkpoint;
# 4. 预览训练过程中的采样图;
# 5. 需要时手动停止。
#
# **24 GB 卡说明**:上游训练入口把整个 FluxPipeline(≈33 GiB)搬上一张卡,4090
# 加载即 OOM。启动器 `train_feature_reuse_24gb.py` 会自动检测并做 2 卡拆分
# (cuda:0 = transformer+LoRA+优化器,cuda:1 = 冻结的 encoders+vae),零改动上游文件。
# **注意布局与阶段一推理相反**(训练的主角是 transformer,必须和梯度/优化器同卡)。
#
# **时间/空间预期**:首次启动会下载 2 个数据 shard(数 GiB,到 `cache/t2i2m/`);
# 每 100 步出一张采样图、每 1000 步存一个 ckpt(`runs/<时间戳>/ckpt/<步数>/default.safetensors`);
# 小规模验证跑到 2000–5000 步、确认 loss 下降且样图对齐 canny 即可去阶段三。


# Cell 0 · 环境自检(任何一项不过都会给出 TROUBLESHOOTING 章节指引)
import os, shutil, sys

def fail(msg, section):
    raise RuntimeError(f"{msg}\n→ 处理办法见 repro/TROUBLESHOOTING.md {section}")

# 仓库根目录自适应:cwd 或其父目录
for cand in (os.getcwd(), os.path.dirname(os.getcwd())):
    if os.path.exists(os.path.join(cand, "omini", "pipeline", "flux_omini.py")):
        REPO_ROOT = cand; break
else:
    fail("找不到 OminiControl 仓库根目录(需包含 omini/pipeline/flux_omini.py)", "§1")
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
print(f"repo root: {REPO_ROOT}")
print(f"python:    {sys.executable}")   # 应指向 omini 环境;不对说明内核选错了(§1)

if sys.version_info < (3, 10):
    fail(f"Python {sys.version.split()[0]} 过旧,应为 omini 环境的 3.12", "§1")

import torch
if not torch.cuda.is_available():
    fail("torch 看不到 GPU", "§1")

# 显存:拆分模式要求 cuda:0 几乎全空(transformer 22.4GiB + 激活 ~1GiB)
n_gpu = torch.cuda.device_count()
total0 = torch.cuda.get_device_properties(0).total_memory / 1024**3
need_split = total0 < 40
if need_split and n_gpu < 2:
    fail(f"GPU0 只有 {total0:.0f}GB(<40GB 需 2 卡拆分),但只见到 {n_gpu} 张卡", "§7")
for d, need in ([(0, 23.0), (1, 12.0)] if need_split else [(0, 36.0)]):
    free = torch.cuda.mem_get_info(d)[0] / 1024**3
    print(f"cuda:{d} 空闲 {free:.1f} GiB(需 ≥{need})")
    if free < need:
        fail(f"cuda:{d} 空闲显存不足,先 nvidia-smi 清掉占卡进程", "§7")

# 磁盘:数据 shard + ckpt + 采样图
free_disk = shutil.disk_usage(REPO_ROOT).free / 1024**3
print(f"磁盘空闲 {free_disk:.0f} GiB(建议 ≥30)")
if free_disk < 15:
    fail("磁盘空间不足,数据 shard 下载会失败", "§7")

CONFIG_PATH = "train/config/feature_reuse.yaml"
if not os.path.exists(CONFIG_PATH):
    fail(f"缺少 {CONFIG_PATH}", "§7")

from huggingface_hub import get_token
if not get_token():
    fail("未登录 HuggingFace(FLUX.1-dev 是 gated 模型)", "§6")
print("\n环境自检通过 ✔  模式:", "2-GPU 拆分" if need_split else "单卡")


# Cell 1 · 配置
CONFIG_PATH   = "train/config/feature_reuse.yaml"  # independent_condition: true 已就绪
WANDB_API_KEY = ""                                 # 可选:填了就开 wandb 可视化
LOG_DIR       = "runs/logs"                        # 训练日志目录

# 传给训练进程的额外环境变量(一般不用动;OMINI_SPLIT=auto 会按显存自动选)
TRAIN_ENV = {
    "OMINI_CONFIG": CONFIG_PATH,
    # "OMINI_SPLIT": "1",          # 强制拆分 / "0" 强制单卡
    # "CUDA_VISIBLE_DEVICES": "0,1",  # 机器上有人占卡时,挑两张空卡
}
print("配置就绪")


# Cell 2 · 后台启动训练(重复运行本 cell 不会起第二个进程)
import os, subprocess, sys, time

def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except (OSError, TypeError):
        return False

if _alive(globals().get("TRAIN_PID")):
    print(f"训练进程已在跑(PID={TRAIN_PID}),不重复启动;日志:{TRAIN_LOG}")
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    TRAIN_LOG = os.path.join(LOG_DIR, time.strftime("train_%Y%m%d-%H%M%S.log"))
    env = {**os.environ, **TRAIN_ENV}
    if WANDB_API_KEY:
        env["WANDB_API_KEY"] = WANDB_API_KEY
    # WHY:trainer.py:312 的 print() 在 stdout 走文件时是 block-buffered
    #      (4KB 才 flush),监控时看不到 loss。PYTHONUNBUFFERED=1 让 Python
    #      走 line-buffered,每行立即写盘,实时监控体验大幅改善。
    env["PYTHONUNBUFFERED"] = "1"
    # WHY 后台子进程而非在 notebook 内跑:训练要跑数小时,内核重启不该杀掉它;
    # 且 33GB 模型留在训练进程里,notebook 内核保持干净。
    logf = open(TRAIN_LOG, "w")
    proc = subprocess.Popen(
        [sys.executable, "repro/train_feature_reuse_24gb.py"],
        stdout=logf, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,   # 脱离内核会话,重启内核不影响训练
    )
    TRAIN_PID = proc.pid
    with open(os.path.join(LOG_DIR, "train.pid"), "w") as f:
        f.write(str(TRAIN_PID))
    print(f"训练已启动:PID={TRAIN_PID}\n日志:{TRAIN_LOG}")
    print("首次启动要先下载数据 shard + 加载 33GB 模型,几分钟内没有 Loss 行是正常的。")


# Cell 3 · 监控(可反复运行;重启内核后也能直接运行 —— 自己从 pid 文件恢复状态)
import glob, os

def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except (OSError, TypeError):
        return False

pid_file = os.path.join(LOG_DIR, "train.pid")
pid = globals().get("TRAIN_PID") or (int(open(pid_file).read()) if os.path.exists(pid_file) else None)
alive = bool(pid) and _alive(pid)
print(f"进程:PID={pid} {'存活 ✔' if alive else '已退出 ✘(看日志末尾找原因,TROUBLESHOOTING §7)'}")

logs = sorted(glob.glob(os.path.join(LOG_DIR, "train_*.log")))
if logs:
    tail = open(logs[-1]).readlines()
    loss_lines = [l for l in tail if "Loss:" in l]
    print(f"\n--- 日志末尾({logs[-1]})---")
    print("".join(tail[-12:]))
    if loss_lines:
        print(f"--- 最近 5 条 loss(共 {len(loss_lines)} 条,每 10 步一条)---")
        print("".join(loss_lines[-5:]))

# GPU 占用一览
import torch
for d in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(d)
    print(f"cuda:{d} 已用 {(total-free)/1024**3:5.1f} / {total/1024**3:.1f} GiB")

# checkpoint 一览(每 1000 步一个)
ckpts = sorted(glob.glob("runs/*/ckpt/*/default.safetensors"), key=os.path.getmtime)
print(f"\ncheckpoints({len(ckpts)} 个):")
for c in ckpts[-5:]:
    print("  ", c)


# Cell 4 · 采样图预览(每 100 步训练会自动出一张,肉眼看是否逐渐对齐 canny 结构)
import glob, os
from IPython.display import display
from PIL import Image

samples = sorted(glob.glob("runs/*/output/lora_*_canny_*.jpg"), key=os.path.getmtime)
if not samples:
    print("还没有采样图(sample_interval=100,启动初期属正常;若长期没有看日志里的 WARN)")
for p in samples[-4:]:
    print(p)
    display(Image.open(p).reduce(2))


# Cell 5 · 停止训练(手动操作:先把 DRY_RUN 改成 False 再运行)
DRY_RUN = True

import os, signal
pid_file = os.path.join(LOG_DIR, "train.pid")
pid = globals().get("TRAIN_PID") or (int(open(pid_file).read()) if os.path.exists(pid_file) else None)
if not pid:
    print("没有记录在案的训练进程")
elif DRY_RUN:
    print(f"DRY_RUN=True,只演习:将会 SIGTERM 进程组 {pid}(最近一次 ckpt 之后的进度会丢)")
else:
    os.killpg(os.getpgid(pid), signal.SIGTERM)   # start_new_session 起的进程组,连 dataloader worker 一起停
    print(f"已发送 SIGTERM 给进程组 {pid}")


## 判读与下一步
#
# | 观察项 | 预期 | 说明 |
# |---|---|---|
# | Loss 曲线 | 前几百步明显下降后放缓 | Prodigy(lr=1)自适应,起步波动大属正常 |
# | 采样图 | 随步数逐渐贴合 canny 轮廓 | `runs/<run>/output/`,100 步一张 |
# | cuda:0 占用 | ≈23 GiB(拆分模式) | 逼近上限是设计如此;OOM 见 TROUBLESHOOTING §7 |
# | ckpt | 每 1000 步一个 | `runs/<run>/ckpt/<步数>/default.safetensors` |
#
# - **跑多久**:小规模验证 2000–5000 步(默认只挂 2 个数据 shard)。要更接近论文
#   质量,在 yaml 里解注释更多 shard 并加大步数。
# - **质量预期**:README 明确 feature reuse 会略微降质、训练更慢 —— 方法本身的
#   trade-off,不是你训错了。
# - **下一步**:有了 ckpt 就去 `repro/stage3_benchmark.ipynb` 做正式测量
#   (它会自动找最新的 ckpt)。
