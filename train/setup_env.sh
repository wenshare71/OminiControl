#!/usr/bin/env bash
# OminiControl 环境激活脚本(venv 缺失时自动重建)。
#
# 用法 —— 必须 source,不能直接执行(否则 activate 只作用于子进程,当前 shell 拿不到):
#
#   cd /home/wuwenxuan03/OminiControl
#   source train/setup_env.sh
#   python repro/probe_max_conditions.py
#
# 可调环境变量:
#   OMINI_VENV            venv 位置(默认 /root/omini-venv)
#   OMINI_FORCE_REBUILD=1 强制删掉重建
#
# ── 为什么这个脚本要能「自愈」 ───────────────────────────────────────────
# 本机 /root 是 overlay,**每次重启回滚到基础镜像** —— venv 和 uv 一起消失。
# 而 /home/wuwenxuan03 是 Ceph 持久盘,仓库和权重活着。
# 所以:脚本住在仓库里(活得下来),venv 建在 /root(每次重建,1~2 分钟)。
# 重启后的完整恢复流程就只剩这一条 `source`。
#
# ── 为什么 venv 绝不能建在 Ceph 上 ──────────────────────────────────────
# 试过,装 28 个包花了 70 分钟还没完。真因:uv 发现 cache 和目标不同盘 → hardlink 失效
# → 退化成逐文件拷贝 → 56 万次写系统调用,每次一轮网络往返。
# 放本地盘:176 毫秒。差 2 万倍。详见 repro/ENV_REBUILD.md §2.2。
#
# ── 为什么这里没有 HF_TOKEN ─────────────────────────────────────────────
# HF_HOME 在 Ceph 共享持久盘上,`hf auth login` 会把 token **明文**写进公司集群的持久存储。
# 权重下完后本地推理不需要 token;真要再下东西时临时 `export HF_TOKEN=hf_xxx` 即可
# (只存活于当前 shell,不落盘)。详见 repro/ENV_REBUILD.md §4。
# ────────────────────────────────────────────────────────────────────────

# 定位仓库根目录。WHY 用 eval 取 zsh 的脚本路径:${(%):-%x} 是 zsh 专有语法,
# 直接写进来会让 bash 在**解析阶段**就报错(哪怕那个分支根本不会执行),eval 可以推迟解析。
if [ -n "${ZSH_VERSION:-}" ]; then
  _omini_src="$(eval 'echo ${(%):-%x}')"
else
  _omini_src="${BASH_SOURCE[0]:-$0}"
fi
OMINI_ROOT="$(cd "$(dirname "$_omini_src")/.." 2>/dev/null && pwd)"
unset _omini_src

if [ ! -f "$OMINI_ROOT/requirements.txt" ]; then
  echo "[omini] ✗ 定位仓库根目录失败(得到 '$OMINI_ROOT')。请在仓库根目录 source 本脚本。"
  return 1 2>/dev/null || exit 1
fi

OMINI_VENV="${OMINI_VENV:-/root/omini-venv}"

# WHY 显式钉住 UV_CACHE_DIR:它必须和 OMINI_VENV 在**同一块盘**上,hardlink 才生效。
# 这正是 70 分钟 → 176 毫秒的那个开关。改 OMINI_VENV 时务必同步改这里。
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"

# 权重/数据放 Ceph 持久盘 —— 否则每次重启要重下 34 GB 的 FLUX。
export HF_HOME="${HF_HOME:-/home/wuwenxuan03/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # Rust 下载器,比 requests 快 3~5 倍

if [ "${OMINI_FORCE_REBUILD:-0}" = "1" ] && [ -d "$OMINI_VENV" ]; then
  echo "[omini] OMINI_FORCE_REBUILD=1 → 删除 $OMINI_VENV"
  rm -rf "$OMINI_VENV"
fi

# ── 自愈:venv 不在就重建 ───────────────────────────────────────────────
if [ ! -x "$OMINI_VENV/bin/python" ]; then
  echo "[omini] venv 不存在(大概率刚重启过)→ 重建中,约 1~2 分钟..."

  # uv 也住在 /root,同样会被重启抹掉,先确保它在。
  if ! command -v uv >/dev/null 2>&1; then
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "[omini] uv 也没了 → 重新安装"
    curl -LsSf https://astral.sh/uv/install.sh | sh || {
      echo "[omini] ✗ uv 安装失败,检查网络"; return 1 2>/dev/null || exit 1; }
    . "$HOME/.local/bin/env"
  fi
  echo "[omini] uv $(uv --version 2>/dev/null | awk '{print $2}')"

  uv venv --python 3.12 "$OMINI_VENV" || {
    echo "[omini] ✗ 创建 venv 失败"; return 1 2>/dev/null || exit 1; }
  . "$OMINI_VENV/bin/activate"

  # WHY torch 单独先装、且指定 cu128 索引:
  #   - torch 2.8 **只存在于** cu126/cu128 索引;cu121 最高到 2.4.1(v1 报告就栽在这)。
  #   - 驱动 535.54.03 能跑 cu128:CUDA Minor Version Compatibility,驱动 ≥ R525 即可跑任意 12.x。
  #     580 是「CUDA 12.8 捆绑发布的驱动版本」,不是「运行 cu128 的最低驱动」。别再被这个数字骗了。
  echo "[omini] 装 torch 2.8.0+cu128 ..."
  uv pip install torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu128 || {
    echo "[omini] ✗ torch 安装失败"; return 1 2>/dev/null || exit 1; }

  # WHY 两个 -r 必须写在同一条命令里:uv 只解析一次。分成两条跑,第二次会重新解析依赖,
  # 有可能把 cu128 的 torch 替换成 PyPI 上的通用版 —— CUDA 就废了,而且报错还很隐蔽。
  echo "[omini] 装项目依赖 ..."
  uv pip install -r "$OMINI_ROOT/requirements.txt" \
                 -r "$OMINI_ROOT/train/requirements.txt" || {
    echo "[omini] ✗ 依赖安装失败"; return 1 2>/dev/null || exit 1; }

  echo "[omini] ✓ 重建完成"
else
  . "$OMINI_VENV/bin/activate"
fi

# ── 自检 ────────────────────────────────────────────────────────────────
echo "[omini] python  : $(python -V 2>&1) @ $(command -v python)"
echo "[omini] HF_HOME : $HF_HOME"
python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    print(f"[omini] torch   : {torch.__version__} | cuda={ok} | gpus={torch.cuda.device_count()}")
    if ok:
        print(f"[omini] gpu0    : {torch.cuda.get_device_name(0)} "
              f"| capability={torch.cuda.get_device_capability(0)}")
    else:
        # WHY 这里要显式喊出来:CUDA 不可用时脚本仍会「成功」返回,
        # 不喊的话下一步跑 benchmark 才炸,排查成本高得多。
        print("[omini] ⚠️  CUDA 不可用!先跑 nvidia-smi 确认驱动,再看 repro/ENV_REBUILD.md §2.1")
except Exception as e:
    print(f"[omini] ⚠️  torch 导入失败: {type(e).__name__}: {e}", file=sys.stderr)
PY
