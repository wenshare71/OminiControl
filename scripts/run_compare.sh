#!/bin/bash
# 对比两个推理结果目录，生成 HTML，并把查看链接写入本脚本同级目录下的 compare_url.txt。
#
# 用法: 直接修改下面的 DIR_A / DIR_B 两个变量，然后 `bash run_compare.sh`。
#
# 可用环境变量覆盖:
#   PYTHON       python 解释器路径(默认 /opt/conda/bin/python)
#   PROMPT_FILE  prompt jsonl(默认 ketu_v3 real badcase 测试集)
#   SUFFIX       图片命名后缀(默认 1)

set -euo pipefail

# ====== 在这里手动修改要对比的两个结果目录 ======
DIR_A="/kaimm-distill/train_logs/fengzipeng/ketu_v3_real_scm/real_train_meanflow_5step_jvp_fp32_k0.5_from_pcm2w_10step_endt_0init_input_r/checkpoint-11000/results_transformer_ema"
DIR_B="/kaimm-distill/train_logs/fengzipeng/ketu_v3_real_scm/real_train_meanflow_5step_jvp_fp32_k0_from_pcm2w_10step_endt_0init_input_r/checkpoint-11000/results_transformer_ema"   # 右侧
# ===============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON="${PYTHON:-/opt/conda/bin/python}"
PROMPT_FILE="${PROMPT_FILE:-/kaimm-distill/fengzipeng/datasets/ketu_v3_testset/pe_info_real_badcase0525.jsonl}"
SUFFIX="${SUFFIX:-1}"

if [ ! -d "$DIR_A" ]; then
    echo "[ERROR] 目录不存在: $DIR_A"; exit 1
fi
if [ ! -d "$DIR_B" ]; then
    echo "[ERROR] 目录不存在: $DIR_B"; exit 1
fi

OUT="$(python "$SCRIPT_DIR/upload_compare.py" \
    --save_path "$DIR_B" \
    --teacher_path "$DIR_A" \
    --prompt_file "$PROMPT_FILE" \
    --suffix "$SUFFIX")"
echo "$OUT"

URL="$(printf '%s\n' "$OUT" | sed -n 's/^show url: //p' | tail -n 1)"
if [ -z "$URL" ]; then
    echo "[ERROR] 未能从 upload_compare.py 输出中解析到 URL"
    exit 1
fi

printf '%s\n' "$URL" > "$SCRIPT_DIR/compare_url.txt"
echo "URL 已写入: $SCRIPT_DIR/compare_url.txt"
