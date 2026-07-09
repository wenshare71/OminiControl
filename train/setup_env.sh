#!/usr/bin/env bash
# Activate the OminiControl conda env (Python 3.12 + torch 2.8/cu128 + diffusers 0.38).
# Source this file (do not execute) in any new shell before running notebooks or training:
#
#   source train/setup_env.sh
#   python -c "from omini.pipeline.flux_omini import generate"
#   jupyter nbconvert --to notebook --execute examples/subject.ipynb
#
export CONDA_ROOT=/root/miniconda3
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate omini
else
  export PATH="$CONDA_ROOT/envs/omini/bin:$PATH"
fi
echo "[omini] using $(python -V) at $(which python)"
echo "[omini] torch $(python -c 'import torch; print(torch.__version__, "cuda="+torch.version.cuda)')"
