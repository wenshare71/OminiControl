#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repro/download_subject_lora.py
==============================
单独下载官方 subject LoRA(Yuanshi/OminiControl,公开仓库、不 gated),
落到 HF cache 里,之后 load_lora_weights / attach_lora 在 HF_HUB_OFFLINE=1
下也能直接命中,无需再联网。

用法(远程机器先 source train/setup_env.sh 保证 HF_HOME 正确):
    python repro/download_subject_lora.py                 # 默认 subject_512
    python repro/download_subject_lora.py --which 1024    # subject_1024_beta
    python repro/download_subject_lora.py --which all
    HF_ENDPOINT=https://hf-mirror.com python repro/download_subject_lora.py  # 直连不通时走镜像
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "Yuanshi/OminiControl"
FILES = {
    "512":  "omini/subject_512.safetensors",
    "1024": "omini/subject_1024_beta.safetensors",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--which", choices=[*FILES, "all"], default="512")
    args = ap.parse_args()

    # WHY: 下载脚本必须允许联网;若外层 shell 残留了 offline 开关会直接 401/LocalEntryNotFound
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.pop(k, None):
            print(f"[DEBUG] 已临时移除环境变量 {k}(下载需联网)")

    names = list(FILES) if args.which == "all" else [args.which]
    for n in names:
        fn = FILES[n]
        print(f"[INFO] 下载 {REPO_ID} :: {fn} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=fn)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[INFO] 完成 -> {path}  ({size_mb:.1f} MB)")

    print("\n之后即使 HF_HUB_OFFLINE=1,下面的调用也会直接命中 cache:")
    print('  attach_lora(pipe, "Yuanshi/OminiControl", "omini/subject_512.safetensors", "subject")')
    return 0


if __name__ == "__main__":
    sys.exit(main())
